"""Synthetic CPU metrics simulator with labeled anomalies (Phase 1)."""

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
    SeasonalSuppressionConfig,
    SpikeConfig,
)
from aiops.simulator.generator import SignalComponents, generate_base_signal

__all__ = [
    "AnomalyEvent",
    "AnomalyKind",
    "BaseSignalConfig",
    "DriftConfig",
    "LevelShiftConfig",
    "MetricConfig",
    "SeasonalSuppressionConfig",
    "SignalComponents",
    "SpikeConfig",
    "generate_base_signal",
    "inject_drift",
    "inject_level_shift",
    "inject_seasonal_suppression",
    "inject_spike",
]