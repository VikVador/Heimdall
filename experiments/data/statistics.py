r"""Compute the statistics of a dataset on its training period: mean, std and std of the change from one day to the next."""

import argparse
import json
import numpy as np

from datetime import date as Date
from datetime import timedelta
from multiprocessing import get_context

from heimdall.config import PATH_PROJECT
from heimdall.data import DATASET_DATES_TRAINING, DATASET_VARIABLES
from heimdall.data.dataset import HeimdallDataset, get_excluded_days
from heimdall.data.tools import moments

# Written on Torch, then copied with rclone to the folder of the dataset on OSN (see its README)
PATH_OUTPUT = PATH_PROJECT / "datasets"

# Partial sums of each year, combined by --reduce
PATH_PARTS = PATH_PROJECT / "datasets" / "statistics-parts"


def accumulate(args: tuple[str, list[str]]) -> np.ndarray:
    r"""Sum the moments of the fields and of their change from one day to the next, over consecutive days.

    Arguments:
        args : Name of the modality and its consecutive days 'YYYY-MM-DD' to read.

    Returns:
        sums : Moments of the fields and of their daily change (2, 3, C, Z), Z = 1 for the observations.
    """

    modality, dates = args
    dataset = HeimdallDataset(
        dates[0], dates[-1], modality, standardized=False, fill_with_nans=True
    )

    sums, previous, yesterday = None, None, None
    for date in dates:
        x = dataset.preprocess(date)[0]
        x = x if x.dim() == 4 else x[:, None]  # (C, Z, Y, X)
        sums = np.zeros((2, 3, *x.shape[:2])) if sums is None else sums

        # The change is only taken between two consecutive days, an excluded day breaking the pair
        consecutive = previous is not None and Date.fromisoformat(date) - yesterday == timedelta(
            days=1
        )

        # One variable at a time, a day of GLORYS in float64 taking 13 GB
        for c in range(len(x)):
            sums[0, :, c] += moments(x[c]).numpy()
            if consecutive:
                sums[1, :, c] += moments(x[c] - previous[c]).numpy()

        previous, yesterday = x, Date.fromisoformat(date)

    return sums


def compute_part(modality: str, year: int, workers: int) -> None:
    r"""Sum the moments of the training days of one year, the days being split between processes.

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.
        year     : Year to read.
        workers  : Number of processes, each reading consecutive days.
    """

    excluded = get_excluded_days(modality)
    dataset = HeimdallDataset(
        f"{year}-01-01", f"{year}-12-31", modality, standardized=False, excluded_days=excluded
    )
    dates = [
        d for d in dataset.dates if DATASET_DATES_TRAINING[0] <= d <= DATASET_DATES_TRAINING[1]
    ]
    if not dates:
        return

    # Each process reads consecutive days, the pairs across two processes being left out
    chunks = [list(c) for c in np.array_split(dates, workers) if len(c) > 0]
    with get_context("spawn").Pool(len(chunks)) as pool:
        sums = sum(pool.map(accumulate, [(modality, chunk) for chunk in chunks]))

    PATH_PARTS.mkdir(parents=True, exist_ok=True)
    np.save(PATH_PARTS / f"{modality}-{year}.npy", sums)


def reduce(modality: str) -> None:
    r"""Combine the yearly sums of a modality into its statistics, written as json.

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.
    """

    sums = sum(np.load(path) for path in sorted(PATH_PARTS.glob(f"{modality}-*.npy")))
    (n, s, s2), (n_change, s_change, s2_change) = sums

    mean = s / n
    std = np.sqrt(s2 / n - mean**2)
    std_change = np.sqrt(s2_change / n_change - (s_change / n_change) ** 2)

    statistics = {
        var: {
            "mean": mean[c].tolist(),
            "std": std[c].tolist(),
            "std_change": std_change[c].tolist(),
        }
        for c, var in enumerate(DATASET_VARIABLES[modality])
    }
    path = PATH_OUTPUT / f"statistics-{modality}.json"
    path.write_text(json.dumps(statistics, indent=1))
    print(f"{modality}: {int(n.sum())} values, written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the statistics of a dataset on its training period."
    )
    parser.add_argument("--modality", "-m", type=str, required=True, help="e.g. glorys, sst, chl.")
    parser.add_argument(
        "--year", "-y", type=int, default=None, help="Year whose sums are computed."
    )
    parser.add_argument("--workers", "-w", type=int, default=4, help="Processes reading the days.")
    parser.add_argument(
        "--reduce", action="store_true", help="Combine the yearly sums into the statistics."
    )
    args = parser.parse_args()

    if args.reduce:
        reduce(args.modality)
    else:
        compute_part(args.modality, args.year, args.workers)
