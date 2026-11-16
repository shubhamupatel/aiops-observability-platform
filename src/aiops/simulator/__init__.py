"""Synthetic CPU metrics simulator with labeled anomalies."""

from aiops.simulator.anomalies import (
    AnomalyEvent,
    AnomalyKind,
    inject_level_shift,
    inject_seasonal_suppression,
    inject_spike,
)
from aiops.simulator.config import (
    BaseSignalConfig,
    LevelShiftConfig,
    MetricConfig,
    SeasonalSuppressionConfig,
    SpikeConfig,
)
from aiops.simulator.generator import SignalComponents, generate_base_signal

__all__ = [
    "AnomalyEvent",
    "AnomalyKind",
    "BaseSignalConfig",
    "LevelShiftConfig",
    "MetricConfig",
    "SeasonalSuppressionConfig",
    "SignalComponents",
    "SpikeConfig",
    "generate_base_signal",
    "inject_level_shift",
    "inject_seasonal_suppression",
    "inject_spike",
]