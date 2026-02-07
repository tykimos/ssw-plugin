---
name: ssw-plugin:ssw-download
description: "Solar observation data downloader for SDO, STEREO, and Solar Orbiter missions. Use when Claude needs to: (1) download SDO/AIA EUV data from JSOC, (2) download STEREO/SECCHI/EUVI data, (3) download Solar Orbiter/EUI/FSI data, (4) create multi-date time series of solar observations, (5) pair multi-wavelength observations. Triggers: 'solar data download', 'SDO download', 'AIA data', 'STEREO data', 'Solar Orbiter data', 'FITS download', 'sun observation data', 'EUI data', 'EUVI data', '태양 데이터 다운로드', '태양 관측 데이터'"
---

# SSW-Download

Download solar observation data from multiple space missions using ssw-tools (https://github.com/sswlab/ssw-tools).

## Setup

```bash
pip install git+https://github.com/sswlab/ssw-tools
```

## Supported Missions

| Mission | Instrument | Wavelengths | Function | Source |
|---------|-----------|-------------|----------|--------|
| Solar Orbiter | EUI/FSI | 174A, 304A | `run_solo()` | SOAR (no registration) |
| STEREO-A/B | SECCHI/EUVI | 171A, 304A | `run_stereo()` | VSO (no registration) |
| SDO | AIA | 94,131,171,193,211,304,335A | SunPy Fido | JSOC (free registration required) |

## Quick Start

### Solar Orbiter (EUI 174A + 304A pair)

```python
from ssw_tools.download_data.solo_down import run_solo
from datetime import datetime

sd = datetime.strptime("2024-06-01T00:00", "%Y-%m-%dT%H:%M")
run_solo(sd, None, delta_hours=12, out_path='./solo_data/',
         level=1, tolerance_min=15, cadence_min=1440)
```

### STEREO (EUVI 171A + 304A pair)

```python
from ssw_tools.download_data.stereo_down import run_stereo
from datetime import datetime

sd = datetime.strptime("2024-06-06T00:00", "%Y-%m-%dT%H:%M")
run_stereo(sd, None, delta_hours=12, out_path='./stereo_data/',
           level=1, tolerance_min=15, cadence_min=1440)
```

### SDO/AIA (via SunPy Fido)

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

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_date` | Search start (datetime) | required |
| `end_date` | Search end (None = same as start) | None |
| `delta_hours` | Search window +/- hours | 12 |
| `out_path` | Output directory | required |
| `level` | Processing level (1 or 2) | 1 |
| `tolerance_min` | Max gap between wavelength pairs | 15 |
| `cadence_min` | Time step for series (1440=daily) | 1440 |

## CLI Interface

```bash
python -m ssw_tools.download_data.main \
    --target solo \
    --start_date 2024-06-01T00:00 \
    --delta_hours 12 \
    --tolerance_min 15 \
    --cadence_min 1440
```

Targets: `solo`, `stereo-a`, `stereo-b`

For multi-date series, batch downloads, and troubleshooting, see `references/download_details.md`.

## Data Availability

- **SDO/AIA**: 2010-present (continuous, 12s cadence)
- **STEREO-A**: 2006-present
- **STEREO-B**: 2006-2014 (contact lost)
- **Solar Orbiter/EUI**: 2020-present (intermittent)

## Related Skills

- **ssw-prep**: Preprocess downloaded data for ML
- **ssw-ml**: Train/inference ML models on solar data
- **ssw-viz**: Visualize solar imagery
