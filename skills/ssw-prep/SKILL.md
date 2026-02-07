---
name: ssw-plugin:ssw-prep
description: "SDO/AIA solar data ML preprocessing pipeline. Use when Claude needs to: (1) preprocess AIA Level 1 FITS data for machine learning, (2) calibrate solar images (pointing, degradation, exposure), (3) register and normalize solar disk images, (4) batch convert raw FITS to ML-ready format, (5) standardize solar observation data for neural network training. Triggers: 'AIA preprocessing', 'solar data prep', 'FITS preprocessing', 'aia_prep_ml', 'ML-ready solar data', 'calibrate AIA', 'solar image registration', '태양 데이터 전처리', 'AIA 보정', 'ML 전처리'"
---

# SSW-Prep

SDO/AIA solar data preprocessing pipeline from ssw-tools (https://github.com/sswlab/ssw-tools). Transforms Level 1 FITS data into standardized ML-ready format.

## Setup

```bash
pip install git+https://github.com/sswlab/ssw-tools
```

## Pipeline Overview

```
Raw AIA Level 1 FITS (4096x4096, variable orientation/brightness)
    │
    ├─ 1. Pointing Correction   → Fix spacecraft orientation errors
    ├─ 2. Registration           → North-up, center disk, resample to fixed size
    ├─ 3. Degradation Correction → Compensate sensor aging over mission lifetime
    └─ 4. Exposure Normalization → DN → DN/s (standardize brightness)
    │
    ▼
ML-Ready (1024x1024, float32, DN/s, north-up, centered)
```

## Quick Start

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
    resolution=1024,        # output size: 512, 1024, or 2048
    padding_factor=0.1      # solar disk padding: 0.1=10%, 0.2=20%
)

# Save
prep_map.save('prep_193A.fits', overwrite=True)
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `aia_map` | SunPy Map of AIA Level 1 data | required |
| `pointing_table` | Pointing calibration table | None |
| `correction_table` | Degradation correction table | None |
| `resolution` | Output pixel size (512/1024/2048) | 1024 |
| `padding_factor` | Extra space around solar disk | 0.1 |

**Resolution formula**: `resolution/2 = (1 + padding_factor) × R_sun_pixels`

## Batch Processing

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

## Output Format

- **Data type**: float32
- **Units**: DN/s (data number per second)
- **Orientation**: Solar north up (CROTA2=0)
- **Centering**: Solar disk centered in frame
- **Metadata**: FITS header updated with calibration info

For pipeline stage details, custom configurations, and helper functions, see `references/pipeline_details.md`.

## Related Skills

- **ssw-download**: Download raw solar observation data
- **ssw-ml**: Train/inference ML models on preprocessed data
- **ssw-viz**: Visualize raw and preprocessed solar images
