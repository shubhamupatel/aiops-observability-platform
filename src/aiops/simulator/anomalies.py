"""Anomaly injectors: each takes a series and returns a new series plus a record of what it injected."""

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from aiops.simulator.config import LevelShiftConfig, SpikeConfig
from aiops.simulator.generator import SignalComponents


class AnomalyKind(StrEnum):
    """The anomaly types the simulator can inject. Each injector adds its own member."""

    SPIKE = "spike"
    LEVEL_SHIFT = "level_shift"


@dataclass(frozen=True)
class AnomalyEvent:
    """Ground-truth record of one injected anomaly. Per-point labels are built from these."""

    kind: AnomalyKind
    start: int	# first anomalous sample
    end: int	# one past the last anomalous sample, like a Python slice
    params: dict[str, float]	# what was injected, like {"magnitude": 27.4}

    def __post_init__(self):
        if not 0 <= self.start < self.end:
            raise ValueError(f"need 0 <= start < end, got start={self.start}, end={self.end}")


def inject_spike(
    components: SignalComponents,
    start: int,
    config: SpikeConfig,
    rng: np.random.Generator,
) -> tuple[SignalComponents, AnomalyEvent]:
    """Inject a spike: add a one-sample upward jump at start. Returns a new series."""
    _check_start(start, len(components.level))

    low, high = config.magnitude_range
    magnitude = float(rng.uniform(low, high))
    level = components.level.copy()	# create a writable copy; the original stays read-only and unchanged
    level[start] += magnitude

    event = AnomalyEvent(AnomalyKind.SPIKE, start, start + 1, {"magnitude": magnitude})
    return replace(components, level=level), event

def inject_level_shift(
    components: SignalComponents,
    start: int,
    config: LevelShiftConfig,
    rng: np.random.Generator,
) -> tuple[SignalComponents, AnomalyEvent]:
    """Inject a level shift: raise the level by a fixed amount for a random duration, then drop back. Returns a new series."""
    n_steps = len(components.level)
    _check_start(start, n_steps)
    shortest, longest = (m // components.interval_minutes for m in config.duration_range_minutes)	# minutes to samples
    if start + longest > n_steps:
        raise ValueError(f"start={start} leaves no room for the longest shift ({longest} samples) before the series ends")

    low, high = config.magnitude_range
    magnitude = float(rng.uniform(low, high))
    n_samples = int(rng.integers(shortest, longest, endpoint=True))	# endpoint=True makes longest itself possible
    end = start + n_samples
    level = components.level.copy()
    level[start:end] += magnitude

    event = AnomalyEvent(AnomalyKind.LEVEL_SHIFT, start, end, {"magnitude": magnitude})
    return replace(components, level=level), event

def _check_start(start: int, n_steps: int) -> None:
    if not 0 <= start < n_steps:
        raise ValueError(f"start must be in [0, {n_steps}), got {start}")