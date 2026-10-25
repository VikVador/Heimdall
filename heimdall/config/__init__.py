r"""Global paths."""

__all__ = [
    "PATH_SCRATCH",
    "PATH_PROJECT",
    "PATH_MODELS",
    "PATH_ANALYSIS",
    "OSN_STORAGE",
    "PATH_OSN",
    "PATH_DATASETS",
    "PATH_MASKS",
    "PATH_STATS",
]

import os

from pathlib import Path

# fmt: off
#
# ----- Main Folders
#
PATH_SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/vm3158"))
PATH_PROJECT = PATH_SCRATCH / "heimdall"

# ----- Subfolders
#
PATH_MODELS   = PATH_PROJECT / "models"
PATH_ANALYSIS = Path(__file__).parents[2] / "experiments" / "analysis"

# ----- Datasets
#
# Anonymous connection to the public bucket
OSN_STORAGE = {"anon": True, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}}

# Creating paths to the datasets on the OSN storage
PATH_OSN    = "s3://m2lines-pubs/Datasets"

GLORYS      = f"{PATH_OSN}/Glorys"
TEMPERATURE = f"{PATH_OSN}/Satellites/Temperature"
SALINITY    = f"{PATH_OSN}/Satellites/Salinity"
WIND        = f"{PATH_OSN}/Satellites/Wind"
CHLOROPHYLL = f"{PATH_OSN}/Satellites/Chlorophyll"

FOLDERS = {
    "glorys"   : GLORYS,
    "sst"      : TEMPERATURE,
    "sss_asc"  : SALINITY,
    "sss_des"  : SALINITY,
    "wind_asc" : WIND,
    "wind_des" : WIND,
    "chl"      : CHLOROPHYLL,
}

# Creating path to data files
PATH_MASKS    = {modality: f"{folder}/mask.zarr" for modality, folder in FOLDERS.items()}
PATH_STATS    = {modality: f"{folder}/statistics.zarr" for modality, folder in FOLDERS.items()}

PATH_DATASETS = {
    "glorys"   : f"{GLORYS}/cmems_glorys_2000-2025_daily_4320x2041x50.zarr",
    "sst"      : f"{TEMPERATURE}/cmems_temperature_2000-2025_daily_3600x1600.zarr",
    "sss_asc"  : f"{SALINITY}/cmems_salinity-asc_2010-2025_daily_1800x837.zarr",
    "sss_des"  : f"{SALINITY}/cmems_salinity-des_2010-2025_daily_1800x837.zarr",
    "wind_asc" : f"{WIND}/cmems_wind-asc_2000-2025_daily_1440x720.zarr",
    "wind_des" : f"{WIND}/cmems_wind-des_2000-2025_daily_1440x720.zarr",
    "chl"      : f"{CHLOROPHYLL}/cmems_chlorophyll_2000-2025_daily_8640x4320.zarr",
}
