"""Dataset assembly: generates N independent series, each with its own anomalies and ground truth."""

from dataclasses import dataclass

import numpy as np

from aiops.simulator.anomalies import AnomalyEvent
from aiops.simulator.config import MetricConfig
from aiops.simulator.generator import SignalComponents, generate_base_signal
from aiops.simulator.labels import labels_from_events
from aiops.simulator.scheduler import inject_anomalies


@dataclass(frozen=True, eq=False)
class SimulatedSeries:
    """One generated series: its position in the dataset, its data, and its ground-truth events."""

    index: int	# position in the dataset, 0 to n_series - 1
    components: SignalComponents
    events: list[AnomalyEvent]	# in time order

    @property
    def labels(self) -> np.ndarray:
        """True for every sample inside an event's labeled window, computed fresh from the events."""
        return labels_from_events(self.events, len(self.components.level))


def generate_dataset(config: MetricConfig, n_series: int, seed: int) -> list[SimulatedSeries]:
    """Generate n_series independent series from one seed. Series i comes out the same however many are generated."""
    if n_series < 1:
        raise ValueError(f"n_series must be at least 1, got {n_series}")
    child_seeds = np.random.SeedSequence(seed).spawn(n_series)	# one independent stream per series
    return [_generate_one(config, index, child) for index, child in enumerate(child_seeds)]


def _generate_one(config: MetricConfig, index: int, seed_sequence: np.random.SeedSequence) -> SimulatedSeries:
    rng = np.random.default_rng(seed_sequence)
    base = generate_base_signal(config.base, rng)
    components, events = inject_anomalies(base, config, rng)
    return SimulatedSeries(index=index, components=components, events=events)