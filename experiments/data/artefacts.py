r"""Find the columns of GLORYS whose values are out of their physical bounds on many days (fixed artefacts)."""

import argparse
import numpy as np

from multiprocessing import get_context

from heimdall.config import PATH_DATASETS, PATH_PROJECT
from heimdall.data import (
    DATASET_DATES_TRAINING,
    DATASET_PADDING,
    DATASET_REGION,
    DATASET_VARIABLES,
    VARIABLES_BOUNDS,
)
from heimdall.data.dataset import HeimdallDataset
from heimdall.data.tools import crop, decode, open_zarr

# Only GLORYS, whose artefacts are fixed points of the model
MODALITY = "glorys"

# Written on Torch, then copied with rclone to the folder of the dataset on OSN (see its README)
PATH_OUTPUT = PATH_PROJECT / "datasets"

# Partial counts of each year, combined by --reduce
PATH_PARTS = PATH_PROJECT / "datasets" / "artefacts-parts"

# One training day out of STRIDE is read, a fixed artefact being out of bounds on many days
STRIDE = 5

# A point is an artefact when its values are out of bounds on at least this share of its days with a value, and on at
# least MIN_DAYS days (a single corrupted read never making an artefact)
SHARE_OF_DAYS = 0.01
MIN_DAYS = 3


def count(dates: list[str]) -> np.ndarray:
    r"""Count, for each point, the days with a value and the days with a value out of its physical bounds.

    Arguments:
        dates : Days 'YYYY-MM-DD' to read.

    Returns:
        counts : Days with a value and days out of bounds, for any variable (2, Z, Y, X).
    """

    ds = crop(
        open_zarr(PATH_DATASETS[MODALITY]), DATASET_REGION[MODALITY], DATASET_PADDING[MODALITY]
    )

    counts = None
    for date in dates:
        day = ds.sel(time=date)
        valid, wrong = False, False
        for var in DATASET_VARIABLES[MODALITY]:
            x = decode(day[var])
            low, high = VARIABLES_BOUNDS[var]
            valid = valid | np.isfinite(x)
            wrong = wrong | (x < low) | (x > high)
        counts = np.zeros((2, *valid.shape), dtype=np.int32) if counts is None else counts
        counts += np.stack([valid, wrong])

    return counts


def compute_part(year: int, workers: int) -> None:
    r"""Count the days with a value and out of bounds over one training year, one day out of STRIDE.

    Arguments:
        year    : Year to read.
        workers : Number of processes reading the days.
    """

    dataset = HeimdallDataset(f"{year}-01-01", f"{year}-12-31", MODALITY, standardized=False)
    dates = [
        d for d in dataset.dates if DATASET_DATES_TRAINING[0] <= d <= DATASET_DATES_TRAINING[1]
    ]
    dates = dates[::STRIDE]
    if not dates:
        return

    chunks = [list(c) for c in np.array_split(dates, workers) if len(c) > 0]
    with get_context("spawn").Pool(len(chunks)) as pool:
        counts = sum(pool.map(count, chunks))

    PATH_PARTS.mkdir(parents=True, exist_ok=True)
    np.save(PATH_PARTS / f"{MODALITY}-{year}.npy", counts.astype(np.int16))


def reduce() -> None:
    r"""Combine the yearly counts into the columns with an artefact at any depth, saved as a boolean (Y, X)."""

    parts = sorted(PATH_PARTS.glob(f"{MODALITY}-*.npy"))
    valid, wrong = sum(np.load(path).astype(np.int64) for path in parts)
    columns = ((wrong >= SHARE_OF_DAYS * valid) & (wrong >= MIN_DAYS)).any(axis=0)

    path = PATH_OUTPUT / f"artefacts-{MODALITY}.npy"
    np.save(path, columns)
    print(f"{int(columns.sum())} columns, written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the columns of GLORYS out of bounds on many days."
    )
    parser.add_argument(
        "--year", "-y", type=int, default=None, help="Year whose days are counted."
    )
    parser.add_argument("--workers", "-w", type=int, default=4, help="Processes reading the days.")
    parser.add_argument("--reduce", action="store_true", help="Combine the yearly counts.")
    args = parser.parse_args()

    if args.reduce:
        reduce()
    else:
        compute_part(args.year, args.workers)
