r"""Global paths."""

__all__ = [
    "PATH_SCRATCH",
    "PATH_PROJECT",
    "PATH_DATASETS",
    "PATH_MODELS",
    "PATH_ANALYSIS",
    "PATH_GLORYS",
    "PATH_OBSERVATIONS",
    "PATH_MASK",
    "PATH_STATS",
]

import os

from pathlib import Path

# fmt: off
#
# ----- Main Folders
#
# Scratch of Torch ($SCRATCH, all-flash, purged when not accessed for 60 days), code and configurations living in $HOME
PATH_SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/vm3158"))
PATH_PROJECT = PATH_SCRATCH / "heimdall"

# ----- Subfolders
#
PATH_DATASETS = PATH_PROJECT / "datasets"
PATH_MODELS   = PATH_PROJECT / "models"
PATH_ANALYSIS = Path(__file__).parents[2] / "experiments" / "analysis"

# ----- Datasets
#
# TODO | GLORYS daily states of the small subset, a zarr store of dimensions (time, depth, latitude, longitude)
PATH_GLORYS = PATH_DATASETS / "glorys" / "glorys.zarr"

# TODO | Daily L3 products of the Copernicus Marine Data Store, regridded on GLORYS, one zarr store each (time, latitude, longitude)
PATH_OBSERVATIONS = {
    "sst" : PATH_DATASETS / "observations" / "sst.zarr",
    "sla" : PATH_DATASETS / "observations" / "sla.zarr",
}

# TODO | Land/sea mask of GLORYS (depth, latitude, longitude), 1 on sea and 0 on land
PATH_MASK = PATH_DATASETS / "structure" / "mask.zarr"

# TODO | Mean and standard deviation of every variable (statistic, [depth]), computed over the training period
PATH_STATS = PATH_DATASETS / "statistics" / "statistics.zarr"
