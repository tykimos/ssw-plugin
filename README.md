# SSW Plugin for Claude Code

Sun and Space Weather toolkit for Claude Code. Download, preprocess, visualize, and apply machine learning to solar observation data from SDO, STEREO, and Solar Orbiter missions.

## Installation

```bash
claude plugin add https://github.com/tykimos/ssw-plugin.git
```

### Prerequisites

```bash
pip install git+https://github.com/sswlab/ssw-tools
pip install sunpy matplotlib astropy aiapy
pip install torch torchvision  # for ML tasks
```

## Skills Overview

| Skill | Command | Description |
|-------|---------|-------------|
| **ssw-download** | `/ssw-plugin:ssw-download` | Download solar observation data (SDO, STEREO, Solar Orbiter) |
| **ssw-prep** | `/ssw-plugin:ssw-prep` | Preprocess raw FITS data into ML-ready format |
| **ssw-ml** | `/ssw-plugin:ssw-ml` | Train and evaluate deep learning models on solar data |
| **ssw-viz** | `/ssw-plugin:ssw-viz` | Visualize solar images, ML results, and analysis |

## Workflow

```
1. Download          2. Preprocess        3. Train ML          4. Visualize
   ssw-download  -->    ssw-prep     -->    ssw-ml       -->    ssw-viz
   (Raw FITS)        (ML-ready FITS)     (Model + Pred)     (Plots & Anim)
```

---

## Skill 1: ssw-download

Download solar observation data from multiple space missions.

### Supported Missions

| Mission | Instrument | Wavelengths | Registration |
|---------|-----------|-------------|-------------|
| **SDO** | AIA | 94, 131, 171, 193, 211, 304, 335 A | JSOC (free, required) |
| **STEREO-A/B** | SECCHI/EUVI | 171, 304 A | Not required |
| **Solar Orbiter** | EUI/FSI | 174, 304 A | Not required |

### Data Availability

| Mission | Period | Notes |
|---------|--------|-------|
| SDO/AIA | 2010 - present | Continuous, 12s cadence |
| STEREO-A | 2006 - present | Active |
| STEREO-B | 2006 - 2014 | Contact lost |
| Solar Orbiter/EUI | 2020 - present | Intermittent |

### Usage Examples

**Solar Orbiter (EUI 174A + 304A pair)**

```python
from ssw_tools.download_data.solo_down import run_solo
from datetime import datetime

sd = datetime.strptime("2024-06-01T00:00", "%Y-%m-%dT%H:%M")
run_solo(sd, None, delta_hours=12, out_path='./solo_data/',
         level=1, tolerance_min=15, cadence_min=1440)
```

**STEREO (EUVI 171A + 304A pair)**

```python
from ssw_tools.download_data.stereo_down import run_stereo
from datetime import datetime

sd = datetime.strptime("2024-06-06T00:00", "%Y-%m-%dT%H:%M")
run_stereo(sd, None, delta_hours=12, out_path='./stereo_data/',
           level=1, tolerance_min=15, cadence_min=1440)
```

**SDO/AIA (via SunPy Fido)**

> Requires free JSOC registration: http://jsoc.stanford.edu/ajax/register_email.html

```python
from sunpy.net import Fido, attrs as a
import astropy.units as u

result = Fido.search(
    a.Time('2024-01-01T00:00', '2024-01-01T00:01'),
    a.jsoc.Series('aia.lev1_euv_12s'),
    a.jsoc.Wavelength(193*u.AA),
    a.jsoc.Segment('image'),
    a.jsoc.Notify('your@email.com')  # JSOC registered email
)
files = Fido.fetch(result, path='./sdo_data/')
```

**CLI Interface**

```bash
python -m ssw_tools.download_data.main \
    --target solo \
    --start_date 2024-06-01T00:00 \
    --delta_hours 12 \
    --tolerance_min 15 \
    --cadence_min 1440
```

Targets: `solo`, `stereo-a`, `stereo-b`

### Download Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_date` | Search start (datetime) | required |
| `end_date` | Search end (None = same as start) | `None` |
| `delta_hours` | Search window +/- hours | `12` |
| `out_path` | Output directory | required |
| `level` | Processing level (1 or 2) | `1` |
| `tolerance_min` | Max gap between wavelength pairs (min) | `15` |
| `cadence_min` | Time step for series (1440 = daily) | `1440` |

---

## Skill 2: ssw-prep

Preprocess SDO/AIA Level 1 FITS data into standardized ML-ready format.

### Pipeline

```
Raw AIA Level 1 FITS (4096x4096, variable orientation/brightness)
    |
    +-- 1. Pointing Correction    --> Fix spacecraft orientation errors
    +-- 2. Registration           --> North-up, center disk, resample to fixed size
    +-- 3. Degradation Correction --> Compensate sensor aging over mission lifetime
    +-- 4. Exposure Normalization --> DN -> DN/s (standardize brightness)
    |
    v
ML-Ready (1024x1024, float32, DN/s, north-up, centered)
```

### Quick Start

```python
from sunpy.map import Map
import astropy.units as u
from aiapy.calibrate.util import get_correction_table, get_pointing_table
from ssw_tools.prep.sdo_aia import aia_prep_ml

# Load raw FITS
aia_map = Map('aia_lev1_file.fits')

# Fetch calibration tables
pointing_table = get_pointing_table(aia_map.date - 6*u.h, aia_map.date + 6*u.h)
correction_table = get_correction_table()

# Preprocess
prep_map = aia_prep_ml(
    aia_map,
    pointing_table=pointing_table,
    correction_table=correction_table,
    resolution=1024,
    padding_factor=0.1
)

# Save
prep_map.save('prep_193A.fits', overwrite=True)
```

### Batch Processing

```python
from pathlib import Path
from sunpy.map import Map
import astropy.units as u
from aiapy.calibrate.util import get_correction_table, get_pointing_table
from ssw_tools.prep.sdo_aia import aia_prep_ml
from loguru import logger

input_dir = Path('./raw/')
output_dir = Path('./prep/')
output_dir.mkdir(exist_ok=True)

correction_table = get_correction_table()  # fetch once, reuse

for fits_file in sorted(input_dir.glob('*.fits')):
    try:
        m = Map(str(fits_file))
        pt = get_pointing_table(m.date - 6*u.h, m.date + 6*u.h)
        prep = aia_prep_ml(m, pt, correction_table, resolution=1024, padding_factor=0.1)
        prep.save(str(output_dir / f'prep_{m.wavelength.value:.0f}A_{m.date.isot}.fits'),
                  overwrite=True)
        logger.info(f'Done: {fits_file.name}')
    except Exception as e:
        logger.error(f'Failed: {fits_file.name}: {e}')
```

### Preprocessing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `aia_map` | SunPy Map of AIA Level 1 data | required |
| `pointing_table` | Pointing calibration table | `None` |
| `correction_table` | Degradation correction table | `None` |
| `resolution` | Output pixel size (512 / 1024 / 2048) | `1024` |
| `padding_factor` | Extra space around solar disk (0.1 = 10%) | `0.1` |

### Output Format

| Property | Value |
|----------|-------|
| Data type | float32 |
| Units | DN/s (data number per second) |
| Orientation | Solar north up (CROTA2=0) |
| Centering | Solar disk centered in frame |
| Metadata | FITS header updated with calibration info |

---

## Skill 3: ssw-ml

Build, train, and evaluate deep learning models on preprocessed solar data.

### Common ML Tasks

| Task | Architecture | Input / Output |
|------|-------------|----------------|
| Instrument Translation | U-Net, Pix2Pix | Image A -> Image B (e.g., STEREO -> SDO) |
| Super Resolution | SRCNN, EDSR | Low-res -> High-res EUV |
| Flare Prediction | CNN+LSTM, ResNet | Time series -> Flare class |
| Coronal Hole Segmentation | U-Net, SegNet | EUV image -> Binary mask |
| Active Region Classification | ResNet, EfficientNet | EUV patch -> Class label |
| Image Generation/Filling | GAN, Diffusion | Partial -> Complete solar disk |

### FITS DataLoader (PyTorch)

```python
import torch
from torch.utils.data import Dataset, DataLoader
from sunpy.map import Map
from pathlib import Path
import numpy as np

class SolarFITSDataset(Dataset):
    def __init__(self, fits_dir, wavelengths=None, transform=None):
        self.fits_dir = Path(fits_dir)
        self.transform = transform
        self.files = sorted(self.fits_dir.glob('*.fits'))
        if wavelengths:
            self.files = [f for f in self.files
                         if any(f'{wl}A' in f.name for wl in wavelengths)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        smap = Map(str(self.files[idx]))
        data = smap.data.astype(np.float32)
        data = np.clip(data, 0, None)
        data = np.log1p(data)
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

### Paired Multi-Wavelength DataLoader

```python
class PairedSolarDataset(Dataset):
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

### U-Net Model

Reference: [InstrumentToInstrument](https://github.com/RobertJaro/InstrumentToInstrument/)

```python
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
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        for f in features:
            self.downs.append(UNetBlock(in_channels, f))
            in_channels = f

        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)

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

### Training Loop

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

### Evaluation Metrics

```python
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
        print(f'{k.upper()}: {np.mean(v):.6f} +/- {np.std(v):.6f}')
    return metrics
```

### Inference

```python
def predict(model, fits_path, device='cuda'):
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
    result = np.expm1(result * np.log1p(smap.data.max()))
    return result
```

---

## Skill 4: ssw-viz

Visualize solar observation data, preprocessing results, and ML model outputs.

### Single Image Display

```python
import matplotlib.pyplot as plt
from sunpy.map import Map
from astropy.visualization import ImageNormalize, AsinhStretch

smap = Map('preprocessed.fits')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection=smap)
smap.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5000, stretch=AsinhStretch(0.01)))
smap.draw_limb(axes=ax, color='white', linewidth=0.5)
smap.draw_grid(axes=ax, color='white', linewidth=0.5, alpha=0.3)
ax.set_title(f'SDO/AIA {smap.wavelength} - {smap.date.iso[:19]}')
plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label='DN/s')
plt.savefig('solar_image.png', dpi=300, bbox_inches='tight')
```

### Wavelength-Specific Colormaps

SunPy auto-selects the correct colormap, or specify manually:

| Wavelength | Colormap | Feature |
|------------|----------|---------|
| 94 A | `sdoaia094` | Flare plasma (green) |
| 131 A | `sdoaia131` | Flare/transition (teal) |
| 171 A | `sdoaia171` | Corona (gold) |
| 193 A | `sdoaia193` | Corona (bronze) |
| 211 A | `sdoaia211` | Active regions (purple) |
| 304 A | `sdoaia304` | Chromosphere (red) |
| 335 A | `sdoaia335` | Active regions (blue) |

### Normalization Options

```python
from astropy.visualization import (
    ImageNormalize, AsinhStretch, LogStretch, SqrtStretch, HistEqStretch
)

# Best for solar EUV (reveals faint coronal features)
norm = ImageNormalize(vmin=0, vmax=5000, stretch=AsinhStretch(0.01))

# High dynamic range
norm = ImageNormalize(vmin=1, vmax=10000, stretch=LogStretch())

# Maximize local contrast
norm = ImageNormalize(stretch=HistEqStretch(smap.data))
```

### Multi-Wavelength Panel

```python
wavelengths = [171, 193, 211, 304]
files = [f'prep_{wl}A.fits' for wl in wavelengths]
maps = [Map(f) for f in files]

fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                          subplot_kw={'projection': maps[0]})

for ax, smap in zip(axes.flatten(), maps):
    smap.plot(axes=ax, norm=ImageNormalize(vmin=0, vmax=5000,
              stretch=AsinhStretch(0.01)))
    smap.draw_limb(axes=ax, color='white', linewidth=0.5)
    ax.set_title(f'{smap.wavelength}')

plt.suptitle(f'Multi-Wavelength - {maps[0].date.iso[:10]}', fontsize=16)
plt.tight_layout()
plt.savefig('multi_wavelength.png', dpi=300, bbox_inches='tight')
```

### Before/After Preprocessing Comparison

```python
fig = plt.figure(figsize=(16, 8))
raw, prep = Map('raw.fits'), Map('prep.fits')

ax1 = fig.add_subplot(121, projection=raw)
raw.plot(axes=ax1)
ax1.set_title('Raw Level 1')

ax2 = fig.add_subplot(122, projection=prep)
prep.plot(axes=ax2, norm=ImageNormalize(vmin=0, vmax=5000, stretch=AsinhStretch(0.01)))
ax2.set_title('ML-Preprocessed')

plt.savefig('before_after.png', dpi=300, bbox_inches='tight')
```

### ML Model Output Comparison

```python
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

input_data = Map('input_171A.fits').data
target_data = Map('target_193A.fits').data
prediction = np.load('model_output.npy')

norm = ImageNormalize(vmin=0, stretch=AsinhStretch(0.01))

axes[0].imshow(input_data, origin='lower', cmap='sdoaia171', norm=norm)
axes[0].set_title('Input (171A)')

axes[1].imshow(prediction, origin='lower', cmap='sdoaia193', norm=norm)
axes[1].set_title('Prediction (193A)')

axes[2].imshow(target_data, origin='lower', cmap='sdoaia193', norm=norm)
axes[2].set_title('Ground Truth (193A)')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig('ml_comparison.png', dpi=300, bbox_inches='tight')
```

### Difference / Error Map

```python
diff = prediction - target_data

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(diff, origin='lower', cmap='RdBu_r', vmin=-500, vmax=500)
ax.set_title('Prediction Error (Pred - Truth)')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, label='DN/s difference')
plt.savefig('error_map.png', dpi=300, bbox_inches='tight')
```

### Time-Lapse Animation

```python
import matplotlib.animation as animation
from pathlib import Path

fits_files = sorted(Path('./prep_data/').glob('*.fits'))
maps = [Map(str(f)) for f in fits_files]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection=maps[0])
norm = ImageNormalize(vmin=0, vmax=5000, stretch=AsinhStretch(0.01))
maps[0].plot(axes=ax, norm=norm)

def update(frame):
    ax.clear()
    maps[frame].plot(axes=ax, norm=norm)
    ax.set_title(f'{maps[frame].date.iso[:19]}')

ani = animation.FuncAnimation(fig, update, frames=len(maps), interval=200)
ani.save('timelapse.mp4', writer='ffmpeg', dpi=150)
```

### Pixel Distribution

```python
data = Map('prep.fits').data.flatten()
data = data[data > 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(data, bins=200, color='steelblue', edgecolor='none')
axes[0].set_xlabel('DN/s'); axes[0].set_title('Linear')

axes[1].hist(np.log10(data), bins=200, color='coral', edgecolor='none')
axes[1].set_xlabel('log10(DN/s)'); axes[1].set_title('Log Scale')

plt.tight_layout()
plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
```

---

## Natural Language Usage

You can also invoke skills by simply describing what you need in natural language:

| What you say | Skill triggered |
|-------------|----------------|
| "Download Solar Orbiter data for June 2024" | ssw-download |
| "SDO 193A 데이터 다운로드해줘" | ssw-download |
| "Preprocess these AIA FITS files for ML" | ssw-prep |
| "태양 데이터 전처리해줘" | ssw-prep |
| "Train a U-Net on the solar data" | ssw-ml |
| "태양 이미지 시각화해줘" | ssw-viz |
| "Show me a multi-wavelength comparison" | ssw-viz |

---

## Project Structure

```
ssw-plugin/
+-- .claude-plugin/
|   +-- plugin.json
+-- skills/
|   +-- ssw-download/
|   |   +-- SKILL.md
|   +-- ssw-prep/
|   |   +-- SKILL.md
|   +-- ssw-ml/
|   |   +-- SKILL.md
|   +-- ssw-viz/
|       +-- SKILL.md
+-- README.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| [ssw-tools](https://github.com/sswlab/ssw-tools) | Solar data download & preprocessing |
| [SunPy](https://sunpy.org/) | Solar data I/O, maps, coordinate systems |
| [astropy](https://www.astropy.org/) | FITS handling, units, visualization |
| [aiapy](https://aiapy.readthedocs.io/) | AIA-specific calibration tables |
| [matplotlib](https://matplotlib.org/) | Plotting and animation |
| [PyTorch](https://pytorch.org/) | Deep learning models |
| [scikit-image](https://scikit-image.org/) | Evaluation metrics (SSIM, PSNR) |

## License

MIT
