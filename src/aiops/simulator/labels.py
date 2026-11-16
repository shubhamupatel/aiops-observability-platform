"""Per-point labels: turns ground-truth events into one True/False value per sample."""

import numpy as np

from aiops.simulator.anomalies import AnomalyEvent


def labels_from_events(events: list[AnomalyEvent], n_steps: int) -> np.ndarray:
    """Mark every sample inside an event's labeled window as True. Raises if events overlap or run past the series."""
    labels = np.zeros(n_steps, dtype=bool)	# every sample starts out normal
    for event in events:
        if event.end > n_steps:
            raise ValueError(f"{event.kind} ends at {event.end}, past the end of the series ({n_steps} samples)")
        if labels[event.start:event.end].any():	# some sample here was already claimed by another event
            raise ValueError(f"{event.kind} starting at {event.start} overlaps another event's labeled window")
        labels[event.start:event.end] = True
    return labels