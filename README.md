# Audio AutoEncoder

A PyTorch implementation of an audio autoencoder for the ESC-50 dataset. Uses a sequence-to-sequence architecture with downsampling/upsampling and LSTM layers.

## Architecture

The model consists of four main components:

- **AudioDownsampler**: Convolutional layers that reduce the input sequence length by a factor of 320
- **AudioEncoder**: Bidirectional LSTM that encodes the downsampled audio into a latent representation
- **AudioDecoder**: LSTM that decodes the latent sequence back to a feature representation
- **AudioUpsampler**: Transposed convolutional layers that reconstruct the original audio length

## Loss Function

Uses a combined loss with two components:

- Multi-resolution STFT loss (weighted at 1.0)
- L1 reconstruction loss (weighted at 0.1)

The STFT loss computes spectrograms at three different resolutions (512, 1024, 2048 FFT sizes) for better frequency domain matching.

## Training

The training script uses:

- AdamW optimizer with weight decay
- OneCycleLR scheduler
- Mixed precision training with gradient scaling
- Gradient clipping at 1.0

Example usage:
```python
autoEncoder = AudioAutoEncoder(
    input_features=128,
    hidden_dim=256,
    latent_dim=512,
    num_layers=2,
)

train(
    model=autoEncoder,
    train_data=train_gen,
    epochs=30,
    batch_size=16,
    steps_per_epoch=100,
    lr=1e-3,
)
```

## Requirements

- PyTorch
- tqdm
- numpy

## Dataset

Expects ESC-50 dataset with:
- Audio files in a directory
- CSV metadata file
- 16kHz sample rate
- 5-second clips

## Model Parameters

- Input features: 128
- Hidden dimensions: 256
- Latent dimensions: 512
- Number of LSTM layers: 2

The model processes 80,000 sample inputs (5 seconds at 16kHz) and compresses them to 250 timesteps in the latent space.