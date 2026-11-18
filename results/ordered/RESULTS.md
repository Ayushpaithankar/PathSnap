# PathSnap ordered-label experiments

The deployed learned model is **LightGBM ranker**. Decoder CV selected it among models trained on a grouped objective. The best overall training-fold combination was **Nearest road + Viterbi** (76.18% versus 75.87% for the deployed model). On the student test split, the deployed model decodes to 88.25%; nearest-road scores plus the same decoder reach 88.38%. A small holdout gap should be interpreted with the CV ranking.

| Method | CV point top-1 | CV grouped loss | CV decoded top-1 | Test point top-1 | Test grouped loss | Test decoded top-1 | Test route F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearest road | 49.23% | 3.898 | 76.18% | 52.95% | 1.046 | 88.38% | 91.87% |
| Distance + heading | 68.84% | 3.696 | 75.47% | 79.26% | 0.777 | 87.42% | 91.48% |
| Conditional logit | 76.71% | 3.479 | 75.34% | 85.43% | 0.570 | 87.29% | 91.01% |
| Grouped MLP | 77.28% | 3.398 | 75.50% | 86.21% | 0.468 | 87.44% | 91.66% |
| ExtraTrees | 77.05% | 3.415 | 75.73% | 86.29% | 0.496 | 87.75% | 91.05% |
| **LightGBM ranker** | 78.00% | 3.374 | 75.87% | 88.31% | 0.428 | 88.25% | 91.63% |
| HistGradientBoosting | 77.77% | 3.397 | 75.98% | 87.48% | 0.468 | 88.60% | 91.27% |

CV point metrics average across two repeats of eight trip folds. Decoded CV pools observations from the first eight folds. The deployed model’s mean paired point gain over distance + heading is +9.15 percentage points (paired SD 4.28 pp). Training and test metrics use the ordered labels, so the earlier geometric-label percentages are on a different target.
Rescored on the same ordered labels, the frozen legacy pipeline reaches 80.03% before decoding and 83.19% after decoding. The comparison measures the complete pipeline changes, including the expanded candidate set.

## Labels and coverage

Ordered alignment labels every observation. The full 80-trip training split contains 208,417 observations. Of those, 185,264 have the correct arc in the candidate set and enter the model fit. The remaining 23,153 stay in validation and coverage denominators. The ten-trip student test split has 19,160 labels and 81 candidate misses.

The largest train candidate-miss counts come from `00000054` (16,888 misses out of 19,313 observations) and `00000033` (3,868 misses out of 5,844 observations). These include extended portions far from the supplied reference. See the per-trip alignment audit.

## Calibration, tuning, and decoding

Optuna ran 30 TPE trials (21 completed, 9 pruned). Within completed trials, 18 stopped tree growth before their cap in at least one fold. The training-trip exponential residual mean is 3.18 m. The selected decoder uses beta 15.92 m, a 60 m/s speed cap, a 90 s long-gap threshold, and emission weight 0.60. Large-gap transitions must pass directed routing and the strict speed cap. The evaluation keeps the student test split out of tuning, model selection, and probability calibration.

## Student test by trip

| Trip | Labeled points | Candidate misses | Nearest road | Distance + heading | Model top-1 | Decoded top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00000000 | 2,503 | 5 | 53.46% | 92.29% | 95.84% | 95.81% |
| 00000007 | 1,354 | 0 | 46.31% | 90.18% | 94.76% | 85.60% |
| 00000009 | 1,049 | 30 | 51.38% | 76.93% | 88.18% | 89.42% |
| 00000040 | 2,995 | 0 | 53.46% | 85.91% | 94.36% | 93.52% |
| 00000052 | 262 | 0 | 77.48% | 91.22% | 97.71% | 99.24% |
| 00000056 | 3,958 | 22 | 68.60% | 83.35% | 89.11% | 91.99% |
| 00000065 | 768 | 1 | 68.88% | 79.56% | 90.89% | 90.23% |
| 00000076 | 1,486 | 12 | 41.25% | 75.84% | 91.52% | 95.69% |
| 00000077 | 3,911 | 11 | 38.07% | 59.40% | 73.77% | 71.59% |
| 00000091 | 874 | 0 | 56.18% | 77.46% | 87.19% | 91.08% |

## Legacy error audit on trip 00000077

The original classifier made 956 labeled errors. 709 were exact reverse arcs, 6 were stationary or slow, and 51 lacked the legacy target candidate. The original decoder had 480 reverse-arc errors. Intersection, parallel-road and adjacent-unlabeled tags are overlapping diagnostic proxies. The error-level CSV gives each flag.
With ordered labels, trip `00000077` reaches 73.77% before decoding and 71.59% after. The largest test-trip loss from decoding is 9.16 percentage points on `00000007`.
