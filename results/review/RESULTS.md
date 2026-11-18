# PathSnap review validation

**Release default: LightGBM untuned.** Both the release default and the tuned ranker are available in the dashboard when they differ.

The reference-distance rule excluded 00000033, 00000054 from training (25,157 observations). The reviewed split contains 26 development trips and 52 separate eight-fold assessment trips. The final supervised model fit used 180,365 trusted points with a reference candidate.

| Method | Trusted CV point top-1 | Trusted CV decoded | Covered CV log loss | Test point top-1 | Test decoded | Test route F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearest road | 54.66% | 87.55% | 0.928 | 52.95% | 88.51% | 93.13% |
| Distance + heading | 77.70% | 86.19% | 0.727 | 79.26% | 87.02% | 93.67% |
| LightGBM untuned | 92.04% | 87.94% | 0.231 | 91.42% | 88.53% | 92.16% |
| LightGBM tuned without context | 88.34% | 86.96% | 0.319 | 88.58% | 88.06% | 92.81% |
| LightGBM tuned | 91.87% | 87.09% | 0.223 | 91.74% | 87.87% | 93.65% |
| HistGradientBoosting | 91.84% | 87.63% | 0.234 | 90.44% | 88.06% | 91.94% |

Assessment coverage: 120,433 trusted observations out of 122,042, with 473 trusted candidate misses. Covered log loss excludes misses; accuracy still counts them as errors. The all-point loss with the 1e-12 floor remains in the CSV for comparison.

## Paired uncertainty

| Split | Tuned ranker minus | Decoded difference | 95% trip interval | Wins / losses / ties |
| --- | --- | ---: | ---: | ---: |
| trusted_cv | Nearest road | -0.45 pp | [-0.73, -0.15] pp | 14 / 36 / 2 |
| trusted_cv | LightGBM untuned | -0.85 pp | [-1.71, -0.22] pp | 22 / 29 / 1 |
| trusted_cv | LightGBM tuned without context | +0.13 pp | [-0.13, +0.47] pp | 25 / 22 / 5 |
| trusted_cv | HistGradientBoosting | -0.54 pp | [-1.14, -0.05] pp | 17 / 29 / 6 |
| test | Nearest road | -0.64 pp | [-1.05, -0.02] pp | 2 / 8 / 0 |
| test | LightGBM untuned | -0.66 pp | [-1.34, -0.06] pp | 1 / 6 / 3 |
| test | LightGBM tuned without context | -0.19 pp | [-0.89, +0.68] pp | 6 / 3 / 1 |
| test | HistGradientBoosting | -0.19 pp | [-0.44, +0.11] pp | 4 / 5 / 1 |

The test route F1 difference (tuned ranker minus nearest) is +0.51 pp, with a paired trip interval of [+0.07, +1.06] pp. Point accuracy uses aligned proxy labels. Route overlap uses the original ordered arc list.

## What tuning and context contributed

- Compared with **LightGBM untuned**, the tuned ranker changes test point accuracy by +0.32 pp and decoded accuracy by -0.66 pp.
- Compared with **LightGBM tuned without context**, the tuned ranker changes test point accuracy by +3.16 pp and decoded accuracy by -0.19 pp.
- Compared with **HistGradientBoosting**, the tuned ranker changes test point accuracy by +1.30 pp and decoded accuracy by -0.19 pp.

### Emission ablation under the tuned ranker decoder

| Emissions | Test decoded top-1 | Route F1 |
| --- | ---: | ---: |
| No emissions | 84.54% | 93.99% |
| Random | 80.54% | 93.44% |
| Nearest road | 88.04% | 93.54% |
| LightGBM tuned | 87.87% | 93.65% |

## Decoder search

| Method | Emission weight | Beta multiplier | Gap threshold (s) | Speed cap (m/s) |
| --- | ---: | ---: | ---: | ---: |
| Nearest road | 0.5 | 4.0 | 15.0 | 28.0 |
| Distance + heading | 0.25 | 10.0 | 15.0 | 28.0 |
| LightGBM untuned | 2.0 | 1.0 | 15.0 | 28.0 |
| LightGBM tuned without context | 0.5 | 2.0 | 60.0 | 28.0 |
| LightGBM tuned | 0.25 | 4.0 | 15.0 | 28.0 |
| HistGradientBoosting | 1.0 | 1.0 | 15.0 | 28.0 |

The local profiles below hold the other coordinates fixed. An edge or flat profile does not establish an interior optimum.

| Method | Coordinate | Selected | Best probed value(s) | Selected accuracy | Best accuracy |
| --- | --- | ---: | --- | ---: | ---: |
| LightGBM tuned | emission_weight | 0.25 | 0.25 | 76.23% | 76.23% |
| LightGBM tuned | beta_multiplier | 4 | 4 | 76.23% | 76.23% |
| LightGBM tuned | max_gap_s | 15 | 15 | 76.23% | 76.23% |
| LightGBM tuned | max_speed_mps | 28 | 16, 28 | 76.23% | 76.23% |
| Nearest road | emission_weight | 0.5 | 0.5 | 75.82% | 75.82% |
| Nearest road | beta_multiplier | 4 | 4 | 75.82% | 75.82% |
| Nearest road | max_gap_s | 15 | 15 | 75.82% | 75.82% |
| Nearest road | max_speed_mps | 28 | 16, 28 | 75.82% | 75.82% |

## Tuned ranker decoded error audit

| Exclusive category | Count | Share |
| --- | ---: | ---: |
| other reference arc | 958 | 41.2% |
| reverse arc | 612 | 26.3% |
| adjacent off-route | 537 | 23.1% |
| other off-route | 136 | 5.9% |
| candidate miss | 81 | 3.5% |

Trip 00000077 has 1246 decoded errors. Of 442 reverse errors, 202 have local GPS displacement along the prediction. Point top-1 is 83.64%; decoded top-1 is 68.14%.

Trip 00000007 has 191 decoded errors. Of 0 reverse errors, 0 have local GPS displacement along the prediction. Point top-1 is 97.19%; decoded top-1 is 85.89%.

The longest uninterrupted loss run on trip 7 spans indices 1166 through 1203 (38 observations). Direction agreement is a diagnostic, since heading noise and reference errors can both contribute.

## Release-default decoded error audit (LightGBM untuned)

| Exclusive category | Count | Share |
| --- | ---: | ---: |
| other reference arc | 860 | 39.1% |
| reverse arc | 663 | 30.2% |
| adjacent off-route | 478 | 21.8% |
| other off-route | 115 | 5.2% |
| candidate miss | 81 | 3.7% |
