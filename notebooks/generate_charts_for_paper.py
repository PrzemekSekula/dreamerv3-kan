"""Generate paper-quality per-game training curve figures.

For each Atari game shared by both conditions, produces a PDF vector figure
comparing DreamerV3 (MLP baseline) and Sparse-KAN training curves as
mean ± one standard deviation bands.  Output is written to paper/figures/
and is suitable for direct inclusion via \\includegraphics in LaTeX.

Step mapping: 40 checkpoints over 4 × 10^5 environment steps → 10 000 steps
per checkpoint row, so the x-axis runs from 10 k to 400 k steps.

Usage
-----
    python notebooks/generate_charts_for_paper.py
    python notebooks/generate_charts_for_paper.py --output path/to/figures
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = pathlib.Path(__file__).parent.parent
_DEFAULT_FIG_DIR = _REPO / "paper" / "figures"

_ORG_CSV = _REPO / "data" / "atari100k" / "original" / "summary.csv"
_KAN_CSV = _REPO / "data" / "atari100k" / "kan_ac" / "summary.csv"

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
_COLOUR_ORG = "#2166ac"   # blue  — DreamerV3 baseline
_COLOUR_KAN = "#d6604d"   # red-orange — Sparse-KAN
_ALPHA_BAND = 0.18

_FIG_W = 3.2   # inches; three figures fit across A4 text width (≈ 16 cm)
_FIG_H = 2.2   # inches

_STEPS_PER_ROW = 10_000   # 4 × 10^5 total / 40 checkpoints

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.titleweight":   "bold",
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.framealpha":  0.75,
    "legend.edgecolor":   "0.8",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "0.88",
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.5,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _game_names(df: pd.DataFrame) -> list[str]:
    return sorted(c[:-5] for c in df.columns if c.endswith("_mean"))


def _steps(n: int) -> np.ndarray:
    return np.arange(1, n + 1) * _STEPS_PER_ROW


def _pretty_name(game: str) -> str:
    replacements = {
        "ms_pacman": "Ms. Pac-Man",
        "kung_fu_master": "Kung Fu Master",
        "up_n_down": "Up N Down",
        "crazy_climber": "Crazy Climber",
        "road_runner": "Road Runner",
        "bank_heist": "Bank Heist",
        "battle_zone": "Battle Zone",
        "demon_attack": "Demon Attack",
        "chopper_command": "Chopper Command",
        "private_eye": "Private Eye",
    }
    if game in replacements:
        return replacements[game]
    return game.replace("_", " ").title()


def _y_limits(means_list: list[np.ndarray]) -> tuple[float, float]:
    """Return y-axis limits clipped to the range of the mean lines."""
    all_vals = np.concatenate([m[~np.isnan(m)] for m in means_list if len(m)])
    if len(all_vals) == 0:
        return (0, 1)
    lo, hi = all_vals.min(), all_vals.max()
    pad = (hi - lo) * 0.12 if hi != lo else (abs(hi) * 0.12 or 1.0)
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# Per-game figure
# ---------------------------------------------------------------------------

def make_figure(
    game: str,
    df_org: pd.DataFrame,
    df_kan: pd.DataFrame,
) -> plt.Figure:
    """Return a Figure for *game* comparing the two conditions."""
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    n = min(len(df_org), len(df_kan))
    x = _steps(n)

    plotted_means: list[np.ndarray] = []

    # ---- DreamerV3 baseline (always present) ----
    org_mean = df_org[f"{game}_mean"].iloc[:n].to_numpy(dtype=float)
    org_std  = df_org[f"{game}_std"].iloc[:n].to_numpy(dtype=float)
    org_std  = np.nan_to_num(org_std, nan=0.0)

    ax.plot(x, org_mean, color=_COLOUR_ORG, label="DreamerV3", zorder=3)
    ax.fill_between(
        x,
        org_mean - org_std,
        org_mean + org_std,
        color=_COLOUR_ORG, alpha=_ALPHA_BAND, zorder=2,
    )
    plotted_means.append(org_mean)

    # ---- Sparse-KAN (may have NaN if data missing) ----
    kan_col = f"{game}_mean"
    if kan_col in df_kan.columns:
        kan_mean = df_kan[kan_col].iloc[:n].to_numpy(dtype=float)
        if not np.all(np.isnan(kan_mean)):
            kan_std = (
                df_kan[f"{game}_std"].iloc[:n].to_numpy(dtype=float)
                if f"{game}_std" in df_kan.columns
                else np.zeros(n)
            )
            kan_std = np.nan_to_num(kan_std, nan=0.0)

            ax.plot(x, kan_mean, color=_COLOUR_KAN, label="Sparse-KAN", zorder=3)
            ax.fill_between(
                x,
                kan_mean - kan_std,
                kan_mean + kan_std,
                color=_COLOUR_KAN, alpha=_ALPHA_BAND, zorder=2,
            )
            plotted_means.append(kan_mean)
        else:
            ax.text(
                0.5, 0.5, "data unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="0.55", style="italic",
            )

    # ---- Axes ----
    ax.set_ylim(*_y_limits(plotted_means))
    ax.set_xlim(x[0], x[-1])

    ax.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}")
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    ax.set_xlabel("Steps (×10⁵)", labelpad=2)
    ax.set_ylabel("Eval Return", labelpad=2)
    ax.set_title(_pretty_name(game))
    ax.tick_params(length=2, pad=2)

    if len(plotted_means) > 1 or kan_col not in df_kan.columns:
        ax.legend(loc="upper left", handlelength=1.4, handletextpad=0.4)

    fig.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=pathlib.Path, default=_DEFAULT_FIG_DIR,
        help="Directory for output PDF figures (default: paper/figures/).",
    )
    args = parser.parse_args()

    out_dir: pathlib.Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    df_org = pd.read_csv(_ORG_CSV)
    df_kan = pd.read_csv(_KAN_CSV)

    games = _game_names(df_org)
    print(f"Generating {len(games)} figures → {out_dir}/")

    for game in games:
        fig = make_figure(game, df_org, df_kan)
        out_path = out_dir / f"{game}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  {game}.pdf")

    print(f"\nDone. {len(games)} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
