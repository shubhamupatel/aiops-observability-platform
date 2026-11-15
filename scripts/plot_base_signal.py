"""Sanity plot of the anomaly-free base signal: the first 3 days and the full series."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aiops.simulator import BaseSignalConfig, generate_base_signal
from aiops.simulator.config import MINUTES_PER_DAY

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"	# repo-root/docs/img, wherever you run it from


def main():
    config = BaseSignalConfig()
    signal = generate_base_signal(config, np.random.default_rng(SEED))
    steps_per_day = MINUTES_PER_DAY // config.interval_minutes
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    views = [
        ("base_signal_3days.png", 3 * steps_per_day, "first 3 days"),
        ("base_signal_14days.png", config.n_steps, "full 14 days"),
    ]
    for filename, end, label in views:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(signal.timestamps[:end], signal.observed[:end], linewidth=0.8, label="observed (what detectors see)")
        ax.plot(signal.timestamps[:end], signal.expected[:end], linewidth=1.8, label="expected (true formula)")
        ax.set_ylim(*config.bounds)	# full 0-100 range, so the headroom is visible
        ax.set_ylabel("CPU utilization (%)")
        ax.set_title(f"Base signal, {label} (seed {SEED})")
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=120)
        plt.close(fig)
        print(f"Saved {OUT_DIR / filename}")


if __name__ == "__main__":
    main()