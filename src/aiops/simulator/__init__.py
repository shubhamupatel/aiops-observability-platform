"""Synthetic CPU metrics simulator with labeled anomalies."""

from aiops.simulator.anomalies import AnomalyEvent, AnomalyKind, inject_spike
from aiops.simulator.config import BaseSignalConfig, MetricConfig, SpikeConfig
from aiops.simulator.generator import SignalComponents, generate_base_signal

__all__ = [
    "AnomalyEvent",
    "AnomalyKind",
    "BaseSignalConfig",
    "MetricConfig",
    "SignalComponents",
    "SpikeConfig",
    "generate_base_signal",
    "inject_spike",
]