# Ordered Route Pipeline

The API automatically uses `results/ordered/ordered_model.joblib` when present. Restart Uvicorn after switching artifacts. Set `PATHSNAP_MODEL=results/best_classifier.joblib` for the legacy comparison. The packaged application and exported model can run with `requirements.txt`. Training uses the additional `requirements-training.txt`, including LightGBM and Optuna. LightGBM training on macOS needs the Homebrew `libomp` runtime.

## Run the experiments

From the project root, after installing the training requirements:

```sh
python -m pathsnap.audit_errors
python -m pathsnap.prepare_grouped "inputs/Student folder.zip" work/grouped
python -m pathsnap.alignment_audit "inputs/Student folder.zip" work/grouped results/ordered/alignment_audit.csv
python -m pathsnap.grouped_training work/grouped work/fit --mode tune --trials 30
python -m pathsnap.grouped_training work/grouped work/fit --params work/fit/tuned_hgb.json
python -m pathsnap.select_grouped "inputs/Student folder.zip" work/grouped work/fit results/ordered --tune-decoder
python -m pathsnap.select_grouped "inputs/Student folder.zip" work/grouped work/fit results/ordered
python -m pathsnap.legacy_comparison "inputs/Student folder.zip" work/grouped results/ordered/legacy_on_ordered_labels.csv
python -m pathsnap.report_ordered
python -m pathsnap.maps "inputs/Student folder.zip" student_test 00000052 results/ordered/demo_map.html --model results/ordered/ordered_model.joblib --reference
```

## Targets and features

Beam Viterbi aligns the GPS trajectory to ordered reference arc **occurrences**. Its states are directed arc occurrences and their projected cumulative travel. It scores lateral distance and mismatch between forward route distance and GPS displacement, with penalties for regressive projected travel and unrealistic speed. The beam retains states and successors to avoid dropping observations in sparse sections. This produces one label per observation, including geometrically ambiguous observations. Labels remain reference-derived estimates. The per-trip alignment audit exposes where reference geometry is far from GPS.

The full training split is processed without an observation cap. Choice estimators train on every observation whose correct arc is available in the generated candidate set. Candidate misses stay in label files, coverage summaries, validation, and test denominators. They contribute zero top-1 credit and a 1e-12 probability floor for grouped log loss. No reference arc is injected into candidates. Candidate generation allows up to eight arcs within 100 m and falls back to the nearest arc.

New features are five-point smoothed direction, stationarity, excess distance over the nearest candidate, heading rank, reverse twins, local and trip-wide arc support, node degrees, previous candidate connectivity and signed lateral offset. The supported inference flow loads the complete trip before computing these context features. Reference files are opened only for labeling and evaluation.

## Grouped experiments

The comparison includes nearest road, distance plus one-step heading, conditional logit, a 16-unit grouped MLP, ExtraTrees, LightGBM LambdaRank and HistGradientBoosting. The conditional logit and MLP minimize observation-level softmax loss. LambdaRank receives whole observation groups. The two tree classifier challengers are converted to logits and evaluated under the same grouped scoring. Score slopes are fitted on out-of-fold trip predictions and followed by an observation softmax. Each reported CV calibration slope excludes its evaluation fold.

Two repeats of eight trip-grouped folds compare point top-1 and grouped log loss. The search runs 30 Optuna TPE trials for HistGradientBoosting over L2, minimum leaf samples, maximum depth and iteration count. Each fit stops after eight iterations without sufficient improvement on whole-trip grouped validation loss. Trials can also prune after the first validation fold. The tuning trace records the trees grown and the selected validation iteration. Paired fold deltas are descriptive because repeated folds are correlated.

Decoder selection uses one full eight-fold set of out-of-fold emissions. The exponential residual scale is the mean absolute aligned route/GPS displacement mismatch on each fold's training trips. A grid compares scale multipliers, long-gap thresholds, speed caps and emission weights using conditional-logit emissions. The decoder penalizes immediate reversals, holds plausible arcs for stationary steps and bridges large GPS gaps only when a directed path satisfies the speed bound. Point accuracy after decoding selects among the three models trained with a grouped objective, with a simplicity tie-break for paired differences within 0.2 percentage points and one descriptive standard error. Rule baselines and candidate-wise classifiers compete in the best-overall benchmark. The final decoder and fitted model use the training split. Student-test observations are excluded from the tuning and selection code to preserve the test comparison.

See [the measured results](results/ordered/RESULTS.md). The original PDF and results at the root of `results/` describe the legacy target. Compare methods within the same ordered-label table.
