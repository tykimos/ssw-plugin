---
name: ssw-plugin:ssw-viz
description: "Solar observation data visualization for EUV imagery, ML results, and analysis. Use when Claude needs to: (1) display solar EUV images from FITS files, (2) create multi-wavelength comparison panels, (3) make before/after preprocessing comparisons, (4) visualize ML model predictions on solar data, (5) create solar time-lapse animations, (6) plot intensity distributions of solar images. Triggers: 'solar visualization', 'solar image display', 'FITS visualization', 'EUV image plot', 'multi-wavelength comparison', 'solar animation', 'sun image', '태양 이미지 시각화', '태양 시각화', 'solar plot'"
---

# SSW-Viz

Visualization patterns for solar observation data, preprocessing results, and ML model outputs. Uses SunPy Map, matplotlib, and astropy visualization tools.

## Setup

```bash
pip install sunpy matplotlib astropy
```

## Single Image Display

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

## Wavelength-Specific Colormaps

SunPy auto-selects correct colormaps, or specify manually:

```python
smap.plot(cmap='sdoaia171')   # 171A - corona (gold)
smap.plot(cmap='sdoaia193')   # 193A - corona (bronze)
smap.plot(cmap='sdoaia304')   # 304A - chromosphere (red)
smap.plot(cmap='sdoaia211')   # 211A - active regions (purple)
smap.plot(cmap='sdoaia094')   # 94A - flare plasma (green)
smap.plot(cmap='sdoaia131')   # 131A - flare/transition (teal)
smap.plot(cmap='sdoaia335')   # 335A - active regions (blue)
```

## Normalization Options

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

## Multi-Wavelength Panel

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

## Before/After Preprocessing

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

## ML Model Output Comparison

```python
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

input_data = Map('input_171A.fits').data
target_data = Map('target_193A.fits').data
prediction = np.load('model_output.npy')  # or from model inference

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

## Difference / Error Map

```python
diff = prediction - target_data

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(diff, origin='lower', cmap='RdBu_r', vmin=-500, vmax=500)
ax.set_title('Prediction Error (Pred - Truth)')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, label='DN/s difference')
plt.savefig('error_map.png', dpi=300, bbox_inches='tight')
```

## Time-Lapse Animation

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

## Pixel Distribution

```python
import numpy as np

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

## Related Skills

- **ssw-download**: Download solar observation data
- **ssw-prep**: Preprocess raw data for ML
- **ssw-ml**: Train and evaluate ML models
