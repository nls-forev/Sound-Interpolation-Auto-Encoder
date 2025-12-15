import torch
import torch.nn as nn

from torch import Tensor
from ..utils.utils import ESC50
from pathlib import Path
from .model import AudioAutoEncoder, CombinedLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


def train(
    model: nn.Module,
    train_data,
    epochs: int,
    batch_size: int = 16,
    steps_per_epoch: int = 100,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=1000,
        cycle_momentum=False,
    )

    loss_fn = CombinedLoss().to("cuda")
    scaler = GradScaler(enabled=(device.type == "cuda"))

    batch_gen = train_data.batch_gen(batch_size)

    for epoch in range(epochs):
        epoch_loss = 0.0

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
        )

        for step in pbar:
            x_np, _ = next(batch_gen)
            x = torch.from_numpy(x_np).float().to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                x_hat, _ = model(x)
                loss = loss_fn(x_hat, x)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch [{epoch+1}/{epochs}]  avg_loss={avg_loss:.6f}")

    train_data.stop = True


if __name__ == "__main__":

    ROOT_DIR = Path("/run/media/klasta/New Volume/Apps/Datasets/ESC-50")
    AUDIO_DIR = ROOT_DIR / "audio"
    CSV_PATH = ROOT_DIR / "esc50.csv"

    train_gen = ESC50(
        CSV_PATH,
        AUDIO_DIR,
        folds=[1, 2, 3, 4],
        inputLength=5,
        audio_rate=16000,
        mix=False,
        normalize=True,
    )

    test_gen = ESC50(
        CSV_PATH,
        AUDIO_DIR,
        folds=[5],
        randomize=False,
        random_crop=False,
        inputLength=5,
        mix=False,
        normalize=True,
    )

    autoEncoder = (
        AudioAutoEncoder(
            input_features=128,
            hidden_dim=256,
            latent_dim=512,
            num_layers=2,
        )
        .train()
        .to("cuda")
    )

    train(
        model=autoEncoder,
        train_data=train_gen,
        epochs=30,
        batch_size=16,
        steps_per_epoch=100,
        lr=1e-3,
    )

    torch.save(autoEncoder.state_dict(), "audio_autoencoder.pt")
    autoEncoder.eval()
