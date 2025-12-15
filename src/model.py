import torch
import torch.nn as nn

from torch import Tensor
from pathlib import Path


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, x_hat, x):
        # x shape is (Batch, Time, 1)
        # We need (Batch, Time) for torch.stft

        # Squeeze the last dimension (channels)
        x_hat = x_hat.squeeze(-1)
        x = x.squeeze(-1)

        x_hat = x_hat.float()
        x = x.float()

        loss = 0.0
        for n_fft in [512, 1024, 2048]:
            hop = n_fft // 4

            # Spectrogram of target
            S_x = torch.stft(x, n_fft, hop_length=hop, return_complex=True)
            S_x = torch.view_as_real(S_x).pow(2).sum(-1).sqrt()

            # Spectrogram of recon
            S_hat = torch.stft(x_hat, n_fft, hop_length=hop, return_complex=True)
            S_hat = torch.view_as_real(S_hat).pow(2).sum(-1).sqrt()

            loss += self.loss(S_hat, S_x)

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, stft_weight=1.0, l1_weight=0.1):
        super().__init__()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.l1_loss = nn.L1Loss()
        self.stft_weight = stft_weight
        self.l1_weight = l1_weight

    def forward(self, x_hat, x):
        return self.stft_weight * self.stft_loss(
            x_hat, x
        ) + self.l1_weight * self.l1_loss(x_hat, x)


class AudioDownsampler(nn.Module):
    def __init__(self):
        super().__init__()
        # Total Stride: 4 * 4 * 4 * 5 = 320
        # 80,000 samples -> 250 time steps
        self.conv = nn.Sequential(
            # Block 1: Stride 4
            nn.Conv1d(1, 64, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            # Block 2: Stride 4
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            # Block 3: Stride 4
            nn.Conv1d(128, 256, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            # Block 4: Stride 5
            nn.Conv1d(256, 128, kernel_size=9, stride=5, padding=4),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Expects (B, T, 1) -> Conv needs (B, 1, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class AudioEncoder(nn.Module):
    """Sequence-to-sequence encoder - outputs full sequence of latents."""

    def __init__(self, input_features, hidden_dim, latent_dim, num_layers=2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Project bidirectional output to latent dim
        self.to_latent = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, T', input_features)
        out, _ = self.encoder(x)  # (B, T', 2*hidden_dim)
        z_seq = self.to_latent(out)  # (B, T', latent_dim)
        return z_seq


class AudioDecoder(nn.Module):
    """Sequence-to-sequence decoder - processes latent sequence."""

    def __init__(self, hidden_dim, latent_dim, num_layers):
        super().__init__()

        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, z_seq):
        # z_seq: (B, T', latent_dim)
        x = self.from_latent(z_seq)  # (B, T', hidden_dim)
        x, _ = self.lstm(x)  # (B, T', hidden_dim)
        return x


class AudioUpsampler(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Must mirror Downsampler strides: 5 -> 4 -> 4 -> 4
        self.net = nn.Sequential(
            # Upsample 5
            nn.ConvTranspose1d(
                hidden_dim, 256, kernel_size=9, stride=5, padding=4, output_padding=4
            ),  # Note 1
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            # Upsample 4
            nn.ConvTranspose1d(
                256, 128, kernel_size=9, stride=4, padding=4, output_padding=3
            ),  # Note 2
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            # Upsample 4
            nn.ConvTranspose1d(
                128, 64, kernel_size=9, stride=4, padding=4, output_padding=3
            ),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            # Upsample 4
            nn.ConvTranspose1d(
                64, 1, kernel_size=9, stride=4, padding=4, output_padding=3
            ),
            nn.Tanh(),
        )

    def forward(self, x, output_length):
        # x: (B, T', H) -> (B, H, T')
        x = x.transpose(1, 2)
        x = self.net(x)
        return x[:, :, :output_length].transpose(1, 2)


class AudioAutoEncoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.downsampler = AudioDownsampler()
        self.encoder = AudioEncoder(
            input_features=input_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.decoder = AudioDecoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.upsampler = AudioUpsampler(hidden_dim=hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)

        x_d = self.downsampler(x)  # (B, T', 128)
        z_seq = self.encoder(x_d)  # (B, T', latent_dim) - sequence of latents!

        x_dec = self.decoder(z_seq)  # (B, T', hidden_dim)

        x_hat = self.upsampler(x_dec, T)
        return x_hat, z_seq
