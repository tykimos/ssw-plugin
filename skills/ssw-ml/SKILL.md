---
name: ssw-plugin:ssw-ml
description: "Machine learning for solar physics using preprocessed SSW data. Use when Claude needs to: (1) train deep learning models on solar EUV images, (2) build solar flare prediction models, (3) perform image-to-image translation between solar instruments, (4) detect/segment coronal holes or active regions, (5) create PyTorch/TensorFlow dataloaders for FITS files, (6) evaluate ML models on solar observation data. Triggers: 'solar ML', 'solar deep learning', 'flare prediction', 'coronal hole detection', 'instrument translation', 'solar image segmentation', 'FITS dataloader', 'solar neural network', '태양 ML', '태양 딥러닝', '플레어 예측', 'solar AI'"
---

# SSW-ML

Machine learning workflows for solar physics research. Build, train, and evaluate deep learning models using preprocessed solar observation data from ssw-prep.

## Setup

```bash
pip install git+https://github.com/sswlab/ssw-tools
pip install torch torchvision  # or tensorflow
```

## Common ML Tasks in Solar Physics

| Task | Architecture | Input → Output |
|------|-------------|----------------|
| Instrument Translation | U-Net, Pix2Pix | Image A → Image B (e.g., STEREO→SDO) |
| Super Resolution | SRCNN, EDSR | Low-res → High-res EUV |
| Flare Prediction | CNN+LSTM, ResNet | Time series → Flare class |
| Coronal Hole Segmentation | U-Net, SegNet | EUV image → Binary mask |
| Active Region Classification | ResNet, EfficientNet | EUV patch → Class label |
| Image Generation/Filling | GAN, Diffusion | Partial → Complete solar disk |

## FITS DataLoader (PyTorch)

```python
import torch
from torch.utils.data import Dataset, DataLoader
from sunpy.map import Map
from pathlib import Path
import numpy as np

class SolarFITSDataset(Dataset):
    """PyTorch Dataset for preprocessed solar FITS files."""

    def __init__(self, fits_dir, wavelengths=None, transform=None):
        self.fits_dir = Path(fits_dir)
        self.transform = transform
        pattern = '*.fits'
        self.files = sorted(self.fits_dir.glob(pattern))

        if wavelengths:
            self.files = [f for f in self.files
                         if any(f'{wl}A' in f.name for wl in wavelengths)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        smap = Map(str(self.files[idx]))
        data = smap.data.astype(np.float32)

        # Normalize to [0, 1]
        data = np.clip(data, 0, None)
        data = np.log1p(data)  # log scaling for high dynamic range
        data = data / data.max() if data.max() > 0 else data

        tensor = torch.from_numpy(data).unsqueeze(0)  # [1, H, W]

        if self.transform:
            tensor = self.transform(tensor)

        metadata = {
            'wavelength': float(smap.wavelength.value),
            'date': str(smap.date.isot),
            'filename': self.files[idx].name
        }
        return tensor, metadata

# Usage
dataset = SolarFITSDataset('./prep_data/', wavelengths=[171, 193, 304])
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
```

## Paired Multi-Wavelength DataLoader

```python
class PairedSolarDataset(Dataset):
    """Dataset for paired multi-wavelength observations (e.g., 171A ↔ 304A)."""

    def __init__(self, input_dir, target_dir, transform=None):
        self.input_files = sorted(Path(input_dir).glob('*.fits'))
        self.target_files = sorted(Path(target_dir).glob('*.fits'))
        self.transform = transform
        assert len(self.input_files) == len(self.target_files)

    def __len__(self):
        return len(self.input_files)

    def _load(self, path):
        data = Map(str(path)).data.astype(np.float32)
        data = np.clip(data, 0, None)
        data = np.log1p(data)
        data = data / (data.max() + 1e-8)
        return torch.from_numpy(data).unsqueeze(0)

    def __getitem__(self, idx):
        inp = self._load(self.input_files[idx])
        tgt = self._load(self.target_files[idx])
        if self.transform:
            inp, tgt = self.transform(inp), self.transform(tgt)
        return inp, tgt
```

## U-Net for Image-to-Image Translation

Reference: InstrumentToInstrument (https://github.com/RobertJaro/InstrumentToInstrument/)

```python
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class SolarUNet(nn.Module):
    """U-Net for solar image translation (e.g., STEREO 171A → SDO 193A)."""

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        for f in features:
            self.downs.append(UNetBlock(in_channels, f))
            in_channels = f

        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(UNetBlock(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)
```

## Training Loop

```python
def train_solar_model(model, train_loader, val_loader, epochs=50, lr=1e-4,
                      device='cuda', save_dir='./checkpoints/'):
    Path(save_dir).mkdir(exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.MSELoss()

    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')

    return model
```

## Evaluation Metrics

```python
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_solar_model(model, test_loader, device='cuda'):
    model.eval()
    metrics = {'mse': [], 'mae': [], 'psnr': [], 'ssim': []}

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            for i in range(outputs.shape[0]):
                pred = outputs[i, 0].cpu().numpy()
                true = targets[i, 0].cpu().numpy()

                metrics['mse'].append(np.mean((pred - true) ** 2))
                metrics['mae'].append(np.mean(np.abs(pred - true)))
                metrics['psnr'].append(psnr(true, pred, data_range=true.max() - true.min()))
                metrics['ssim'].append(ssim(true, pred, data_range=true.max() - true.min()))

    for k, v in metrics.items():
        print(f'{k.upper()}: {np.mean(v):.6f} ± {np.std(v):.6f}')
    return metrics
```

## Inference

```python
def predict(model, fits_path, device='cuda'):
    from sunpy.map import Map

    smap = Map(fits_path)
    data = smap.data.astype(np.float32)
    data = np.clip(data, 0, None)
    data = np.log1p(data)
    data = data / (data.max() + 1e-8)

    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    result = output[0, 0].cpu().numpy()
    result = np.expm1(result * np.log1p(smap.data.max()))  # inverse transform
    return result
```

For task-specific architectures and advanced training strategies, see `references/architectures.md`.

## Related Skills

- **ssw-download**: Download raw solar observation data
- **ssw-prep**: Preprocess data for ML (required before training)
- **ssw-viz**: Visualize model inputs, outputs, and evaluation results
