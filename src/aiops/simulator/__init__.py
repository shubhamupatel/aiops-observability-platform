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
from aiops.simulator.dataset import SimulatedSeries, generate_dataset
from aiops.simulator.generator import SignalComponents, generate_base_signal
from aiops.simulator.labels import labels_from_events
from aiops.simulator.scheduler import PlacementError, inject_anomalies
from aiops.simulator.storage import config_from_dict, config_to_dict, save_dataset

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
    "SimulatedSeries",
    "SpikeConfig",
    "config_from_dict",
    "config_to_dict",
    "generate_base_signal",
    "generate_dataset",
    "inject_anomalies",
    "inject_drift",
    "inject_level_shift",
    "inject_seasonal_suppression",
    "inject_spike",
    "labels_from_events",
    "save_dataset",
]