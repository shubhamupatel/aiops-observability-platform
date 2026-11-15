"""Anomaly injectors: each takes a series and returns a new series plus a record of what it injected."""

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from aiops.simulator.config import SpikeConfig
from aiops.simulator.generator import SignalComponents


class AnomalyKind(StrEnum):
    """The anomaly types the simulator can inject. Each injector adds its own member."""

    SPIKE = "spike"


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
    """Add a one-sample upward jump at start. Returns a new series; the input is left untouched."""
    n_steps = len(components.level)
    if not 0 <= start < n_steps:
        raise ValueError(f"start must be in [0, {n_steps}), got {start}")

    low, high = config.magnitude_range
    magnitude = float(rng.uniform(low, high))
    level = components.level.copy()	# create a writable copy; the original stays read-only and unchanged
    level[start] += magnitude

    event = AnomalyEvent(AnomalyKind.SPIKE, start, start + 1, {"magnitude": magnitude})
    return replace(components, level=level), event