"""Synthetic CPU metrics simulator with labeled anomalies."""

from aiops.simulator.anomalies import (
    AnomalyEvent,
    AnomalyKind,
    inject_drift,
    inject_level_shift,
    inject_seasonal_suppression,
    inject_spike,
)
from aiops.simulator.config import (
    BaseSignalConfig,
    DriftConfig,
    LevelShiftConfig,
    MetricConfig,
    ScheduleConfig,
    SeasonalSuppressionConfig,
    SpikeConfig,
)
from aiops.simulator.generator import SignalComponents, generate_base_signal
from aiops.simulator.scheduler import PlacementError, inject_anomalies

__all__ = [
    "AnomalyEvent",
    "AnomalyKind",
    "BaseSignalConfig",
    "DriftConfig",
    "LevelShiftConfig",
    "MetricConfig",
    "PlacementError",
    "ScheduleConfig",
    "SeasonalSuppressionConfig",
    "SignalComponents",
    "SpikeConfig",
    "generate_base_signal",
    "inject_anomalies",
    "inject_drift",
    "inject_level_shift",
    "inject_seasonal_suppression",
    "inject_spike",
]