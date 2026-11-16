"""Scoring a detector's flags against ground truth: which samples count, which events were caught, and the metrics both give."""

from dataclasses import dataclass, fields

import numpy as np

from aiops.data import LabeledSeries

MINUTES_PER_DAY = 24 * 60


@dataclass(frozen=True)
class ScoringConfig:
    """Rules for which samples count toward a score. Every detector is scored under the same rules."""

    warmup_minutes: int = 1440	# the first day isn't scored, so every detector gets one normal cycle to learn from, like NAB's probationary period
    grace_minutes: int = 60	# flags this soon after an event ends count neither way; half the 2-hour minimum gap, so it never reaches the next event

    def __post_init__(self):
        if self.warmup_minutes < 0 or self.grace_minutes < 0:
            raise ValueError("warmup_minutes and grace_minutes must be non-negative")


@dataclass(frozen=True)
class EventResult:
    """Whether one ground-truth event was caught, and how long after its labeled start the first flag came."""

    kind: str	# like "spike"
    detected: bool
    delay_minutes: float | None	# None when the event was missed


@dataclass(frozen=True)
class Score:
    """Counts from comparing flags with ground truth, and the metrics they give. Pool series by adding counts, not by averaging metrics."""

    true_positives: int	# flagged samples inside a labeled window
    false_positives: int	# flagged normal samples
    false_negatives: int	# labeled samples left unflagged
    true_negatives: int	# normal samples left unflagged
    false_alarms: int	# runs of consecutive flags that touch no labeled window
    scored_minutes: float	# how much time the counts cover
    events: tuple[EventResult, ...]	# one per ground-truth event

    @property
    def precision(self) -> float:
        """Of the flagged samples, the share that were anomalous."""
        return _ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Of the anomalous samples, the share that were flagged. Long anomalies dominate it."""
        return _ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        """Harmonic mean of point precision and point recall."""
        return _harmonic_mean(self.precision, self.recall)

    @property
    def false_positive_rate(self) -> float:
        """Of the normal samples, the share that were flagged."""
        return _ratio(self.false_positives, self.false_positives + self.true_negatives)

    @property
    def event_recall(self) -> float:
        """Of the events, the share with at least one flag inside their labeled window. Each event counts once, whatever its length."""
        return _ratio(sum(event.detected for event in self.events), len(self.events))

    @property
    def composite_f1(self) -> float:
        """Harmonic mean of point precision and event recall: every wrong flag costs, and every event counts once. Thresholds are tuned on this."""
        return _harmonic_mean(self.precision, self.event_recall)

    @property
    def false_alarms_per_day(self) -> float:
        """False alarms per day of scored time: what an on-call engineer would feel."""
        return _ratio(self.false_alarms, self.scored_minutes / MINUTES_PER_DAY)

    @property
    def event_recall_by_kind(self) -> dict[str, tuple[int, int]]:
        """Events detected and events in total for each kind, like {"spike": (28, 30)}. Counts, not shares, since some kinds are rare."""
        kinds = sorted({event.kind for event in self.events})
        return {
            kind: (sum(e.detected for e in self.events if e.kind == kind), sum(e.kind == kind for e in self.events))
            for kind in kinds
        }

    @property
    def median_delay_minutes(self) -> float | None:
        """Median time from a detected event's labeled start to its first flag. None when nothing was detected."""
        delays = [event.delay_minutes for event in self.events if event.detected]
        return float(np.median(delays)) if delays else None


def scored_mask(labeled: LabeledSeries, config: ScoringConfig = ScoringConfig()) -> np.ndarray:
    """True for every sample that counts: not in the warm-up, a drift's unlabeled ramp, or the grace after an event's end."""
    interval = _interval_minutes(labeled.series.timestamps)
    scored = np.ones(len(labeled.series.values), dtype=bool)
    scored[:int(config.warmup_minutes / interval)] = False
    grace = int(config.grace_minutes / interval)
    for event in labeled.events:
        scored[event.changed_start:event.start] = False	# drift's ramp before it crosses the label threshold; empty for the other kinds
        scored[event.end:event.end + grace] = False	# the drop back to normal and the samples right after it
    return scored | labeled.labels	# never exclude a labeled sample


def score_series(labeled: LabeledSeries, flags: np.ndarray, config: ScoringConfig = ScoringConfig()) -> Score:
    """Score one series: compare a detector's per-sample flags with its ground truth, counting only scored samples."""
    if flags.dtype != bool or flags.shape != labeled.series.values.shape:
        raise ValueError(f"{labeled.series.name}: flags must be a True/False array with one entry per sample")
    interval = _interval_minutes(labeled.series.timestamps)
    labels = labeled.labels
    scored = scored_mask(labeled, config)
    hits = flags & scored	# flags on unscored samples count neither for nor against the detector
    return Score(
        true_positives=int((hits & labels).sum()),
        false_positives=int((hits & ~labels).sum()),
        false_negatives=int((~hits & labels).sum()),
        true_negatives=int((~hits & ~labels & scored).sum()),
        false_alarms=_count_false_alarms(flags, labels, scored),
        scored_minutes=float(scored.sum() * interval),
        events=tuple(_event_result(event.kind, hits[event.start:event.end], interval) for event in labeled.events),
    )


def pool(scores: list[Score]) -> Score:
    """Pool scores from many series by adding their counts, so every sample and every event weighs the same."""
    if not scores:
        raise ValueError("need at least one score to pool")
    totals = {item.name: sum(getattr(score, item.name) for score in scores) for item in fields(Score) if item.name != "events"}
    return Score(**totals, events=tuple(event for score in scores for event in score.events))


def split_tuning_and_test(dataset: list[LabeledSeries]) -> tuple[list[LabeledSeries], list[LabeledSeries]]:
    """Split the dataset in two: the first half of the series for choosing thresholds, the second half for reporting results."""
    if len(dataset) < 2:
        raise ValueError(f"need at least 2 series to split, got {len(dataset)}")
    half = len(dataset) // 2	# 20 series: series_00 to 09 for tuning, 10 to 19 for the results
    return dataset[:half], dataset[half:]


def _interval_minutes(timestamps: np.ndarray) -> float:
    gaps = np.diff(timestamps)
    if len(gaps) == 0 or (gaps != gaps[0]).any():
        raise ValueError("scoring needs at least two evenly spaced samples")
    return float(gaps[0] / np.timedelta64(1, "m"))	# like 5.0


def _event_result(kind: str, window_hits: np.ndarray, interval: float) -> EventResult:
    found = np.flatnonzero(window_hits)	# positions of flags inside the event's window, counted from its start
    if len(found) == 0:
        return EventResult(kind=kind, detected=False, delay_minutes=None)
    return EventResult(kind=kind, detected=True, delay_minutes=float(found[0] * interval))


def _count_false_alarms(flags: np.ndarray, labels: np.ndarray, scored: np.ndarray) -> int:
    """Count runs of consecutive flags that touch no labeled sample but at least one scored one. Runs wholly in unscored time are ignored."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))	# where each run of flags starts and stops
    starts, ends = edges[::2], edges[1::2]
    return sum(1 for start, end in zip(starts, ends) if not labels[start:end].any() and scored[start:end].any())


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0	# 0/0 counts as 0, like scikit-learn, so flagging nothing never looks good


def _harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b else 0.0