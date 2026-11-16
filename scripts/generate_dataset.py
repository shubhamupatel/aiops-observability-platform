"""Generate the committed synthetic dataset, save it to data/synthetic/, draw the README figures, and print a summary."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from aiops.simulator import AnomalyKind, MetricConfig, generate_dataset, save_dataset

SEED = 42
N_SERIES = 20
REPO_ROOT = Path(__file__).resolve().parent.parent	# works wherever the script is run from
DATA_DIR = REPO_ROOT / "data" / "synthetic"
IMG_DIR = REPO_ROOT / "docs" / "img"
COLORS = {	# one color per anomaly type, shared by both figures
    AnomalyKind.SPIKE: "tab:red",
    AnomalyKind.LEVEL_SHIFT: "tab:orange",
    AnomalyKind.SEASONAL_SUPPRESSION: "tab:purple",
    AnomalyKind.DRIFT: "tab:green",
}
HALF_SAMPLE = np.timedelta64(150, "s")	# half of a 5-minute sample, so a shaded window covers its samples fully


def main():
    config = MetricConfig()
    dataset = generate_dataset(config, N_SERIES, SEED)
    save_dataset(dataset, config, SEED, DATA_DIR)
    print(f"Saved {N_SERIES} series to {DATA_DIR}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_overview(dataset[:3], IMG_DIR / "dataset_overview.png")
    _plot_gallery(dataset, IMG_DIR / "anomaly_gallery.png")
    _print_summary(dataset, config)


def _plot_overview(dataset, path):
    fig, axes = plt.subplots(len(dataset), 1, figsize=(14, 2.6 * len(dataset)), sharex=True)
    for ax, series in zip(axes, dataset):
        times = series.components.timestamps
        ax.plot(times, series.components.observed, linewidth=0.5, color="tab:blue")
        for event in series.events:
            if event.kind is AnomalyKind.SPIKE:	# one sample is too thin to shade at this scale, so mark it with a dot
                ax.scatter(times[event.start], series.components.observed[event.start], color=COLORS[event.kind], s=25, zorder=3)
            else:
                _shade(ax, times, event.start, event.end, COLORS[event.kind], 0.25)
        ax.set_ylim(*series.components.bounds)
        ax.set_ylabel(f"series {series.index}\nCPU (%)")
    axes[0].set_title(f"Synthetic CPU series with their labeled anomalies (seed {SEED})")
    axes[0].legend(handles=_legend_handles(), loc="upper right", ncol=4, fontsize=8)
    fig.autofmt_xdate()	# the panels share one time axis, so only the bottom one needs date labels
    _save(fig, path)


def _plot_gallery(dataset, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, kind in zip(axes.flat, COLORS):
        series, event = _first_event_of(dataset, kind)
        pad = max(36, (event.end - event.changed_start) // 2)	# at least 3 hours of context on each side
        low = max(0, event.changed_start - pad)
        high = min(len(series.components.level), event.end + pad)
        times = series.components.timestamps
        ax.plot(times[low:high], series.components.observed[low:high], linewidth=0.8, color="tab:blue", label="observed")
        ax.plot(times[low:high], series.components.expected[low:high], linewidth=1.6, color="black", label="expected")
        for other in series.events:	# neighbors that fall in view get a lighter shade, so no jump goes unexplained
            if other is not event and other.changed_start < high and other.end > low:
                _shade(ax, times, max(other.start, low), min(other.end, high), COLORS[other.kind], 0.12)
        _shade(ax, times, event.start, event.end, COLORS[kind], 0.25, label="labeled")
        if event.changed_start < event.start:	# drift: the ramp begins before its label does
            _shade(ax, times, event.changed_start, event.start, COLORS[kind], 0.08, label="unlabeled ramp")
        ax.set_ylim(*series.components.bounds)
        ax.set_title(f"{kind.replace('_', ' ')} (series {series.index}, magnitude {event.params['magnitude']:.1f})", fontsize=10)
        ax.legend(loc="upper left", fontsize=7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)	# each panel covers different dates, so each keeps its labels
    _save(fig, path)


def _shade(ax, times, start, end, color, alpha, label=None):
    ax.axvspan(times[start] - HALF_SAMPLE, times[end - 1] + HALF_SAMPLE, color=color, alpha=alpha, label=label, linewidth=0)


def _first_event_of(dataset, kind):
    for series in dataset:
        for event in series.events:
            if event.kind is kind:
                return series, event
    raise ValueError(f"no {kind} in the dataset")


def _legend_handles():
    return [Patch(color=color, alpha=0.5, label=kind.replace("_", " ")) for kind, color in COLORS.items()]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


def _print_summary(dataset, config):
    expected_counts = {kind: getattr(config, kind).count for kind in AnomalyKind}
    every_series_matches = all(Counter(event.kind for event in series.events) == expected_counts for series in dataset)
    totals = Counter(event.kind for series in dataset for event in series.events)
    shares = [float(series.labels.mean()) for series in dataset]
    print(f"Events per type: {', '.join(f'{kind} {totals[kind]}' for kind in AnomalyKind)}")
    print(f"Every series has the configured counts: {every_series_matches}")
    print(f"Labeled share per series: mean {np.mean(shares):.1%}, min {min(shares):.1%}, max {max(shares):.1%}")


if __name__ == "__main__":
    main()