"""Anomaly injectors: each takes a series and returns a new series plus a record of what it injected."""

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from aiops.simulator.config import LevelShiftConfig, SeasonalSuppressionConfig, SpikeConfig
from aiops.simulator.generator import SignalComponents


class AnomalyKind(StrEnum):
    """The anomaly types the simulator can inject. Each injector adds its own member."""

    SPIKE = "spike"
    LEVEL_SHIFT = "level_shift"
    SEASONAL_SUPPRESSION = "seasonal_suppression"


@dataclass(frozen=True)
class AnomalyEvent:
    """The ground-truth record of an injected anomaly."""

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
    shortest, longest = _duration_bounds(components, start, config.duration_range_minutes)

    low, high = config.magnitude_range
    magnitude = float(rng.uniform(low, high))	# drawn before the duration; keep this order so seeds reproduce the data
    n_samples = int(rng.integers(shortest, longest, endpoint=True))	# endpoint=True makes longest itself possible
    end = start + n_samples
    level = components.level.copy()
    level[start:end] += magnitude

    event = AnomalyEvent(AnomalyKind.LEVEL_SHIFT, start, end, {"magnitude": magnitude})
    return replace(components, level=level), event


def inject_seasonal_suppression(
    components: SignalComponents,
    start: int,
    config: SeasonalSuppressionConfig,
    rng: np.random.Generator,
) -> tuple[SignalComponents, AnomalyEvent]:
    """Inject a seasonal suppression: flatten the daily cycle to zero for a random duration, then let it return. Returns a new series."""
    shortest, longest = _duration_bounds(components, start, config.duration_range_minutes)

    n_samples = int(rng.integers(shortest, longest, endpoint=True))
    end = start + n_samples
    magnitude = float(np.abs(components.seasonal[start:end]).max())	# largest change to the expected value, read before zeroing
    seasonal = components.seasonal.copy()
    seasonal[start:end] = 0.0

    event = AnomalyEvent(AnomalyKind.SEASONAL_SUPPRESSION, start, end, {"magnitude": magnitude})
    return replace(components, seasonal=seasonal), event


def _check_start(start: int, n_steps: int) -> None:
    if not 0 <= start < n_steps:
        raise ValueError(f"start must be in [0, {n_steps}), got {start}")


def _duration_bounds(
    components: SignalComponents,
    start: int,
    duration_range_minutes: tuple[int, int],
) -> tuple[int, int]:
    """Convert a duration range to samples, after checking that start leaves room for the longest one."""
    n_steps = len(components.level)
    _check_start(start, n_steps)
    shortest, longest = (m // components.interval_minutes for m in duration_range_minutes)	# minutes to samples
    if start + longest > n_steps:
        raise ValueError(f"start={start} leaves no room for the longest duration ({longest} samples) before the series ends")
    return shortest, longest