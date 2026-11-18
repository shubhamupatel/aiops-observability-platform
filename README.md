# aiops-observability-platform

Detects anomalies in server metrics and explains each one in plain language with an LLM. Built on a synthetic metrics simulator with ground-truth labels, so detection is measured with precision and recall.

**Status:** In progress. Phase 1 of 7 (synthetic metrics simulator) is complete. Phase 2 (baseline detector) is in progress: the data loader, the scoring rules, and the detector are built and tested; threshold tuning and the held-out evaluation are next, so there are no detection results yet. The LLM explanation layer is planned, not yet built.

![Three synthetic CPU series with their labeled anomalies](docs/img/dataset_overview.png)

## What's built: the simulator

The simulator generates realistic CPU-utilization series with injected anomalies whose exact location and size are known, so detectors can be scored against ground truth.

- **Normal signal:** 14 days at 5-minute intervals (4,032 samples), matching AWS CloudWatch and the Numenta Anomaly Benchmark (NAB). A 35% baseline, a daily cycle peaking at 2 p.m., and AR(1) noise, because real CPU readings are correlated over time.
- **Four anomaly types:** spikes, level shifts, seasonal suppressions (the daily cycle disappears for 12 to 24 hours), and drifts (a slow climb that drops back, like a memory leak ended by a restart).
- **Placement:** a scheduler places 3 spikes, 2 level shifts, 1 suppression, and 1 drift in each series without overlaps, after a one-day warm-up and with at least 2 hours between anomalies.
- **Ground truth kept separate:** labels come from the noise-free signal and are saved apart from the observed data, so a detector can't read the answer key.
- **Reproducible:** one seed produces the whole dataset, and each series has its own independent random stream.

![One example of each anomaly type](docs/img/anomaly_gallery.png)

## The dataset

`data/synthetic/` holds 20 series with 140 labeled anomalies, covering about 11% of samples.

| File | Contents |
|---|---|
| `series/series_00.csv` to `series_19.csv` | `timestamp,value`: only what a monitoring system would observe |
| `events.json` | every anomaly's type, window, timestamps, and parameters |
| `metadata.json` | the config, seed, series count, and NumPy version that produced the data |

A test regenerates the dataset from `metadata.json` and fails if the committed files don't match the code byte for byte.

## In progress: the baseline detector

Phase 2 builds a first detector and a fair way to grade it. The scoring rules were fixed before the detector existed, so they couldn't be shaped around its results.

- **Loader:** reads the committed files back the way a detector would see them, timestamps and values only, with each series' ground-truth events held in a separate object. It rejects malformed files, missing or non-finite values, out-of-order timestamps, and events that don't line up with their series.
- **Metrics:** point-wise precision, recall, F1, and false-positive rate, plus event-wise recall, false alarms per day, and detection delay. Thresholds are tuned on composite F1, the harmonic mean of point-wise precision and event-wise recall, so every wrong flag costs something and every anomaly counts once, whatever its length.
- **Scoring rules:** the first day isn't scored, a drift's ramp below its label threshold counts neither way, and flags in the hour after an anomaly ends are neither credited nor penalized. Series 00 to 09 are for tuning; series 10 to 19 are scored once for the reported results.
- **Seasonal z-score detector:** compares each reading with the median of the same time of day over up to the last 7 days, and scores how many typical deviations away it is, in either direction. It uses only past data, so it could run live, and it keeps its expected value and deviation at every sample so an alert can be explained later. Only its threshold will be tuned.
- **Next:** choosing the threshold, one run on the held-out series alongside a static "alert above N% CPU" rule for comparison, and the results.

## Quickstart

Requires Python 3.13.

```
git clone https://github.com/shubhamupatel/aiops-observability-platform.git
cd aiops-observability-platform
python -m venv .venv
.venv\Scripts\activate	# on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt	# the exact versions the project was tested with
pip install -e . --no-deps	# the project itself, without re-resolving versions
pytest	# any warning counts as a failure
python scripts/generate_dataset.py	# regenerates data/synthetic and the figures
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | Synthetic CPU metrics simulator with labeled anomalies | Done |
| 2 | Baseline detector, scored with precision, recall, F1, and false-positive rate | In progress |
| 3 | A second detector from a different family, compared on the same data | Planned |
| 4 | Gemini explanation layer with structured outputs and tool calls | Planned |
| 5 | Prompt design and anti-hallucination evaluation | Planned |
| 6 | Docker and GitHub Actions CI/CD | Planned |
| 7 | Write-up with results | Planned |

Also planned: validation on real AWS CPU data from NAB.

## Known limitations

### Simulator

- No downward spikes or level shifts; the daytime part of a suppression is the only downward deviation.
- Suppression windows include near-normal samples around 8 a.m. and 8 p.m., where the daily cycle crosses its average.
- A single daily cycle with no weekly pattern, and every series shares one server profile.
- CPU only, and anomalies never overlap.
- The sudden drop at the end of a level shift or drift is labeled normal. Scoring treats the hour after each anomaly as neutral, so flagging the drop costs nothing, but flags that continue past that hour count as false positives.
- Noise correlation and spike length are defined per sample, so changing the sampling interval requires retuning them.

### Detection and scoring

- The detector can't separate small sustained shifts, around 5 to 8 points, from noise.
- It gives no score for the first day plus 6 hours, so an anomaly in the 6 hours after the warm-up counts as missed.
- Until 3 earlier days exist, one anomalous day can pull the expected value, so days 2 and 3 are less reliable.
- Detection and scoring need evenly spaced samples; real data with gaps isn't handled yet.
- A flag one sample after a spike doesn't count as catching it.

## Layout

```
src/aiops/simulator/	config, generator, anomalies, scheduler, labels, dataset, storage
src/aiops/data/	loader: reads saved series back, with their ground truth kept separate
src/aiops/evaluation/	scoring: point-wise and event-wise metrics, and the tuning/test split
src/aiops/detection/	seasonal: the seasonal z-score detector
tests/	211 tests
scripts/	dataset generation and plotting
data/synthetic/	the committed dataset
docs/img/	figures
```

## License

MIT