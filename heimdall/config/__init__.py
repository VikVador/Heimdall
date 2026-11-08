r"""Global paths."""

__all__ = [
    "PATH_SCRATCH",
    "PATH_PROJECT",
    "PATH_MODELS",
    "PATH_LATENTS",
    "PATH_LATENT_DATASETS",
    "PATH_ANALYSIS",
    "DATA_SOURCE",
    "OSN_STORAGE",
    "PATH_OSN",
    "PATH_GLORYS_DAYS",
    "GLORYS_DAYS_COARSENING",
    "PATH_DATASETS",
    "PATH_MASKS",
    "PATH_STATS",
    "PATH_EXCLUDED_DAYS",
    "PATH_ARTEFACTS",
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
PATH_LATENTS  = PATH_PROJECT / "latents"
PATH_ANALYSIS = Path(__file__).parents[2] / "experiments" / "analysis"

# ----- Datasets
#
# Where the datasets are read: "osn" (the public bucket, from anywhere) or "torch" (its copy on the scratch of Torch,
# much faster to read); the copy has the same folders and files as the bucket
DATA_SOURCE = "torch"

# Anonymous connection to the public bucket
OSN_STORAGE = {"anon": True, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}}

# Creating paths to the datasets, on the OSN storage or on Torch
PATH_BUCKET = "s3://m2lines-pubs/Datasets"
PATH_OSN    = PATH_BUCKET if DATA_SOURCE == "osn" else str(PATH_PROJECT / "datasets" / "osn")

# The store of GLORYS (12 TB) stays on OSN: on Torch, its days already preprocessed with a coarsening of 4 are read
# instead, the other days and coarsenings coming from OSN
PATH_GLORYS_DAYS       = None if DATA_SOURCE == "osn" else PATH_PROJECT / "datasets" / "glorys-c4"
GLORYS_DAYS_COARSENING = 4

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
PATH_MASKS          = {modality: f"{folder}/mask.zarr" for modality, folder in FOLDERS.items()}
PATH_STATS          = {modality: f"{folder}/statistics-{modality}.json" for modality, folder in FOLDERS.items()}
PATH_EXCLUDED_DAYS  = {modality: f"{folder}/excluded-days-{modality}.json" for modality, folder in FOLDERS.items()}

# Columns of GLORYS with values out of their physical bounds on many days (fixed artefacts of the model), the
# observations keeping their gaps (their contaminations by ice or RFI are transient)
PATH_ARTEFACTS = {"glorys": f"{GLORYS}/artefacts-glorys.npy"}

PATH_DATASETS = {
    "glorys"   : f"{PATH_BUCKET}/Glorys/cmems_glorys_2000-2025_daily_4320x2041x50.zarr",
    "sst"      : f"{TEMPERATURE}/cmems_temperature_2000-2025_daily_3600x1600.zarr",
    "sss_asc"  : f"{SALINITY}/cmems_salinity-asc_2010-2025_daily_1800x837.zarr",
    "sss_des"  : f"{SALINITY}/cmems_salinity-des_2010-2025_daily_1800x837.zarr",
    "wind_asc" : f"{WIND}/cmems_wind-asc_2000-2025_daily_1440x720.zarr",
    "wind_des" : f"{WIND}/cmems_wind-des_2000-2025_daily_1440x720.zarr",
    "chl"      : f"{CHLOROPHYLL}/cmems_chlorophyll_2000-2025_daily_8640x4320.zarr",
}

# ----- Latents
#
# Latents of each dataset, encoded by its trained encoder on the scratch of Torch (temporary: the scratch is purged
# after 60 days without access)
PATH_LATENT_DATASETS = {
    "glorys"   : PATH_LATENTS / "ENC_glorys_lc128_tk25515__D60D"  / "best-validation",
    "sst"      : PATH_LATENTS / "ENC_sst_lc128_tk5650__F439"      / "best-validation",
    "sss_asc"  : PATH_LATENTS / "ENC_sss_asc_lc128_tk5876__00CA"  / "best-validation",
    "sss_des"  : PATH_LATENTS / "ENC_sss_des_lc128_tk5876__0836"  / "best-validation",
    "wind_asc" : PATH_LATENTS / "ENC_wind_asc_lc128_tk3780__038B" / "best-validation",
    "wind_des" : PATH_LATENTS / "ENC_wind_des_lc128_tk3780__991F" / "best-validation",
    "chl"      : PATH_LATENTS / "ENC_chl_lc128_tk9045__940E"      / "best-validation",
}
