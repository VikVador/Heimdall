r"""Information about our datasets, the GLORYS states and the surface observations."""

__all__ = [
    "VARIABLES_CLIPPING",
    "DATASET_DATES_TRAINING",
    "DATASET_DATES_VALIDATION",
    "DATASET_DATES_TEST",
    "DATASET_REGION",
    "DATASET_VARIABLES_GLORYS",
    "DATASET_OBSERVATIONS",
    "MODALITIES",
    "Z",
    "X",
    "Y",
]

# fmt: off
#
# ----- Preprocessing
#
# TODO | Physical bounds of the variables, (lower, upper) with None when unbounded
VARIABLES_CLIPPING = {
    "so" : (0, None),
}

# ----- Splits
#
# TODO | Dates of the small subset available on Torch
DATASET_DATES_TRAINING   = ("1993-01-01", "2016-12-31")
DATASET_DATES_VALIDATION = ("2017-01-01", "2019-12-31")
DATASET_DATES_TEST       = ("2020-01-01", "2021-12-31")

# ----- Domain
#
# TODO | Crop of the GLORYS grid, every modality being regridded on it
DATASET_REGION = {
    "x": slice(0, 512),
    "y": slice(0, 256),
    "z": slice(0, 32),
}

# ----- Variables
#
# TODO | GLORYS states x_t, 3D variables only for the first prototype (zos, 2D, to add later)
DATASET_VARIABLES_GLORYS = [
    "thetao",
    "so",
    "uo",
    "vo",
]

# TODO | Surface observations y_t, the variables of each L3 product, each product having its own encoder
DATASET_OBSERVATIONS = {
    "sst" : ["sea_surface_temperature"],
    "sla" : ["sla"],
}

# Modalities | Each with its own encoder and latent
MODALITIES = ["glorys", *DATASET_OBSERVATIONS]

# ----- Dimensions
#
X = DATASET_REGION["x"].stop - DATASET_REGION["x"].start  # Longitudes
Y = DATASET_REGION["y"].stop - DATASET_REGION["y"].start  # Latitudes
Z = DATASET_REGION["z"].stop - DATASET_REGION["z"].start  # Depth levels
