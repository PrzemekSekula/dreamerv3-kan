"""Compute human-normalized Atari 100k scores from a summary CSV.

Reads a summary CSV produced by prepare_summary.py (columns: <game>_mean,
<game>_std, …), normalises the per-game mean scores using the human and random
baselines from human_data.py, and writes a new CSV that contains only the
normalised mean columns (one per game).  After saving, the script prints a
table of per-game normalised scores for the last checkpoint row, followed by
the mean human-normalised score (IQM-style mean is intentionally skipped here;
we use the plain mean for a quick sanity check).

Usage
-----
    python notebooks/calculate_normalized_results.py [--input PATH] [--output PATH]

Defaults
--------
    --input  data/atari100k/kan_ac/summary.csv
    --output data/atari100k/kan_ac/summary_normalized.csv
"""

import argparse
import pathlib

import pandas as pd

from human_data import get_human_normalized_score

_REPO_ROOT = pathlib.Path(__file__).parent.parent


def _extract_games(df: pd.DataFrame) -> list[str]:
    """Return sorted list of game names present in the dataframe."""
    games = []
    for col in df.columns:
        if col.endswith("_mean"):
            games.append(col[: -len("_mean")])
    return sorted(games)


def normalise(input_path: pathlib.Path, output_path: pathlib.Path) -> pd.DataFrame:
    """Read *input_path*, normalise means, write *output_path*, return result."""
    df = pd.read_csv(input_path)
    games = _extract_games(df)

    norm_df = pd.DataFrame()
    for game in games:
        raw_col = f"{game}_mean"
        norm_df[f"{game}_norm"] = df[raw_col].apply(
            lambda score, g=game: get_human_normalized_score(g, score)
        )

    norm_df.to_csv(output_path, index=False)
    return norm_df


def display(norm_df: pd.DataFrame) -> None:
    """Print per-game normalised scores for the last row and the overall mean."""
    last = norm_df.iloc[-1]
    games = [col[: -len("_norm")] for col in norm_df.columns]

    print(f"\n{'Game':<25} {'Normalised score':>18}")
    print("-" * 45)
    for game in games:
        score = last[f"{game}_norm"]
        print(f"{game:<25} {score:>18.4f}")

    mean_score = last.mean()
    median_score = last.median()
    print("-" * 45)
    print(f"{'Mean HNS':<25} {mean_score:>18.4f}")
    print(f"{'Median HNS':<25} {median_score:>18.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=_REPO_ROOT / "data/atari100k/kan_ac/summary.csv",
        help="Path to the input summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Path for the output normalised CSV (default: <input_dir>/summary_normalized.csv).",
    )
    args = parser.parse_args()

    input_path: pathlib.Path = args.input
    output_path: pathlib.Path = (
        args.output
        if args.output is not None
        else input_path.parent / "summary_normalized.csv"
    )

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    norm_df = normalise(input_path, output_path)
    display(norm_df)
    print(f"Saved normalised results to {output_path}")


if __name__ == "__main__":
    main()
