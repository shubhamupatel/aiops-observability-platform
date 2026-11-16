"""Scheduler: places each anomaly type's configured count into a series without overlaps, then injects them."""

from collections.abc import Callable

import numpy as np

from aiops.simulator.anomalies import (
    AnomalyEvent,
    AnomalyKind,
    inject_drift,
    inject_level_shift,
    inject_seasonal_suppression,
    inject_spike,
)
from aiops.simulator.config import (
    DriftConfig,
    LevelShiftConfig,
    MetricConfig,
    ScheduleConfig,
    SeasonalSuppressionConfig,
    SpikeConfig,
)
from aiops.simulator.generator import SignalComponents

AnomalyConfig = SpikeConfig | LevelShiftConfig | SeasonalSuppressionConfig | DriftConfig
Injector = Callable[..., tuple[SignalComponents, AnomalyEvent]]

INJECTORS: dict[AnomalyKind, Injector] = {	# placed in this order: longest types first, so the short ones fill the gaps
    AnomalyKind.SEASONAL_SUPPRESSION: inject_seasonal_suppression,
    AnomalyKind.DRIFT: inject_drift,
    AnomalyKind.LEVEL_SHIFT: inject_level_shift,
    AnomalyKind.SPIKE: inject_spike,
}


class PlacementError(RuntimeError):
    """Raised when an anomaly can't be placed within the attempt limit."""


def inject_anomalies(
    components: SignalComponents,
    config: MetricConfig,
    rng: np.random.Generator,
) -> tuple[SignalComponents, list[AnomalyEvent]]:
    """Inject every anomaly the config asks for at random, non-overlapping starts. Returns a new series and its events in time order."""
    events: list[AnomalyEvent] = []
    for kind, inject in INJECTORS.items():
        kind_config = getattr(config, kind)	# config fields are named after the AnomalyKind values
        for _ in range(kind_config.count):
            components, event = _place_one(components, inject, kind_config, config.schedule, events, rng)
            events.append(event)
    return components, sorted(events, key=lambda event: event.changed_start)


def _place_one(
    components: SignalComponents,
    inject: Injector,
    kind_config: AnomalyConfig,
    schedule: ScheduleConfig,
    placed: list[AnomalyEvent],
    rng: np.random.Generator,
) -> tuple[SignalComponents, AnomalyEvent]:
    """Try random starts until the injected anomaly keeps its gap from every placed one."""
    interval = components.interval_minutes
    earliest = schedule.warmup_minutes // interval
    latest = len(components.level) - _longest_samples(kind_config, interval)	# the injectors need room for the longest duration
    gap = schedule.min_gap_minutes // interval

    for _ in range(schedule.max_attempts):
        start = int(rng.integers(earliest, latest, endpoint=True))
        candidate, event = inject(components, start, kind_config, rng)
        if not any(_too_close(event, other, gap) for other in placed):
            return candidate, event	# rejected candidates are simply dropped; the injectors never changed the input
    raise PlacementError(f"couldn't place a {event.kind} in {schedule.max_attempts} attempts; try fewer or shorter anomalies")


def _longest_samples(kind_config: AnomalyConfig, interval: int) -> int:
    durations = getattr(kind_config, "duration_range_minutes", None)
    return 1 if durations is None else durations[1] // interval	# a spike has no duration range; it lasts one sample


def _too_close(a: AnomalyEvent, b: AnomalyEvent, gap: int) -> bool:
    """True if the changed stretches of a and b overlap or have fewer than gap normal samples between them."""
    return a.changed_start < b.end + gap and b.changed_start < a.end + gap