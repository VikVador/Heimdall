r"""Information about our datasets, the GLORYS states and the surface observations."""

__all__ = [
    "VARIABLES_CLIPPING",
    "VARIABLES_STANDARDIZATION",
    "DATASET_DATES_TRAINING",
    "DATASET_DATES_VALIDATION",
    "DATASET_DATES_TEST",
    "DATASET_VARIABLES",
    "DATASET_REGION",
    "DATASET_PADDING",
    "DATASET_SHAPES",
    "MODALITIES",
]

# fmt: off
#
# ----- Preprocessing
#
VARIABLES_CLIPPING = {
    "so"                   : (0, None),
    "Sea_Surface_Salinity" : (0, None),
    "CHL"                  : (0, None),
}

VARIABLES_STANDARDIZATION = {
    "thetao"                  : "daily",
    "so"                      : "daily",
    "uo"                      : "daily",
    "vo"                      : "daily",
    "sea_surface_temperature" : "daily",
    "Sea_Surface_Salinity"    : "daily",
    "eastward_wind"           : "daily",
    "northward_wind"          : "daily",
    "CHL"                     : "daily",
    "default"                 : "daily",
}

# ----- Splits
#
DATASET_DATES_TRAINING   = ("2000-01-01", "2018-12-31")
DATASET_DATES_VALIDATION = ("2019-01-01", "2021-12-31")
DATASET_DATES_TEST       = ("2022-01-01", "2025-12-31")

# ----- Variables
#
DATASET_VARIABLES = {
    "glorys"   : ["thetao", "so", "uo", "vo"],
    "sst"      : ["sea_surface_temperature"],
    "sss_asc"  : ["Sea_Surface_Salinity"],
    "sss_des"  : ["Sea_Surface_Salinity"],
    "wind_asc" : ["eastward_wind", "northward_wind"],
    "wind_des" : ["eastward_wind", "northward_wind"],
    "chl"      : ["CHL"],
}

MODALITIES = list(DATASET_VARIABLES)

# ----- Domain
#
DATASET_REGION = {
    "glorys"   : {"x": slice(0, 4320), "y": slice(25, 2041), "z": slice(0, 48)},
    "sst"      : {"x": slice(0, 3600), "y": slice(0,  1600)},
    "sss_asc"  : {"x": slice(0, 1800), "y": slice(5,   837)},
    "sss_des"  : {"x": slice(0, 1800), "y": slice(5,   837)},
    "wind_asc" : {"x": slice(0, 1440), "y": slice(46,  718)},
    "wind_des" : {"x": slice(0, 1440), "y": slice(46,  718)},
    "chl"      : {"x": slice(0, 8640), "y": slice(32, 4320)},
}

# Longitudes added on each side of the region, taken from the other side of the globe (periodic)
DATASET_PADDING = {
    "glorys"   : 0,
    "sst"      : 8,
    "sss_asc"  : 4,
    "sss_des"  : 4,
    "wind_asc" : 0,
    "wind_des" : 0,
    "chl"      : 0,
}

DATASET_SHAPES = {
    modality: tuple(
        region[axis].stop - region[axis].start + (2 * DATASET_PADDING[modality] if axis == "x" else 0)
        for axis in ("z", "y", "x")
        if axis in region
    )
    for modality, region in DATASET_REGION.items()
}
