r"""List the days of an observation product with too few values, to leave them out of its datasets."""

import argparse
import json
import numpy as np

from torch.utils.data import DataLoader

from heimdall.config import PATH_EXCLUDED_DAYS
from heimdall.data.dataset import HeimdallDataset

# A day is left out when it has fewer values than this share of the median day
SHARE_OF_MEDIAN = 0.1


def count_values(modality: str, workers: int) -> tuple[list[str], np.ndarray]:
    r"""Count the values of every day of a modality, after its physical bounds and its ocean mask.

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.
        workers  : Number of processes reading the days.

    Returns:
        dates  : Date strings 'YYYY-MM-DD' of every day of the store.
        counts : Number of points with a value of the first variable, for each day.
    """

    dataset = HeimdallDataset("2000-01-01", "2025-12-31", modality=modality, standardized=False)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        multiprocessing_context="spawn" if workers > 0 else None,
    )
    counts = np.array([mask[0, 0].sum().item() for _, mask, _ in loader])

    return dataset.dates, counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List the days of a product with too few values.")
    parser.add_argument(
        "--modality", "-m", type=str, required=True, help="e.g. sst, sss_asc, chl."
    )
    parser.add_argument("--workers", "-w", type=int, default=8, help="Processes reading the days.")
    args = parser.parse_args()

    dates, counts = count_values(args.modality, args.workers)
    excluded = [
        d for d, c in zip(dates, counts, strict=True) if c < SHARE_OF_MEDIAN * np.median(counts)
    ]

    path = PATH_EXCLUDED_DAYS[args.modality]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(excluded, indent=1))
    print(f"{args.modality}: {len(excluded)} days left out of {len(dates)}, written to {path}")
