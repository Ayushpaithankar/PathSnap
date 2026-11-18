# Model training and selection evidence

## What to open in a review

Open `notebooks/PathSnap_Colab.ipynb`. Its first section now contains **saved outputs from an actual local execution**: five learned models each fit in two repetitions of eight whole-trip folds, 18 Optuna TPE trials, and a refit/export of the trained default. The separate `notebooks/PathSnap_Training_Proof.ipynb` is the focused executed record with the same outputs. Both identify the runtime as local. To prove execution in your own Colab session, upload the project archive, rerun those cells in Colab, and save the notebook. Later saved-result charts are historical and are labeled accordingly. The final full decoder reproduction is optional and expensive on a hosted CPU.

`notebooks/PathSnap_CSV_Training_Colab.ipynb` is a shorter demonstration: it fits two LightGBM configurations and HistGradientBoosting on precomputed CSV features, then compares held-out top-1 scores with both road rules. It does not reproduce five-model CV or HMM route metrics.

## Source and files

- Raw data: `inputs/Student folder.zip`, 80 training and 10 student-test trips. The separate evaluator partition is never used for training or model choice.
- Ordered reference alignment and 28 context features: `pathsnap/ordered_labels.py`, `pathsnap/prepare_grouped.py`, `pathsnap/grouped_features.py`.
- Five learned model families and two rule baselines: `pathsnap/grouped_training.py` and `pathsnap/grouped_models.py`.
- Five-model experiment outputs: `results/ordered/cv_summary.csv`, `results/ordered/cv_folds.csv`, `results/ordered/student_test_comparison.csv`. The five trained methods are Conditional logit, Grouped MLP, ExtraTrees, LightGBM LambdaRank and HistGradientBoosting.
- Bayesian search: `pathsnap/review_experiment.py` and `results/review/ranker_optuna_trials.csv`; the chosen configuration is `results/review/ranker_parameters.json`.
- Decoder search and final selection: `results/review/decoder_sweeps.csv`, `cv_per_trip.csv`, `test_summary.csv`, `trip_bootstrap.csv`, and `selection.json`.
- Exported artifacts: `results/review/deployed.joblib` and `results/review/tuned_ranker.joblib`.
- Executed notebook output files: `results/notebook_proof/five_model/` and `results/notebook_proof/ranker/`. The notebook records Python, dataset SHA256, commands, exit codes, 80 fold scores, 18 trials and the non-null fitted model assertion.

## Why these decisions were made

The ordered five-model comparison found LightGBM LambdaRank had the best learned candidate-level grouped CV top-1 (77.995%) and the best learned student-test top-1 (88.314%). A later context-feature review improved tuned LightGBM student-test point top-1 to 91.738%. The latter is a separate protocol; do not compare those percentages as if only the model changed.

The corrected release deploys **untuned LightGBM + Viterbi**, the trained method with the highest trusted decoded accuracy in the existing 52-trip assessment CV (87.94%). Nearest road + Viterbi scored 87.55%, a +0.40 percentage-point difference whose paired 95% trip-bootstrap interval includes zero. We choose the learned scorer to satisfy the trained-classifier deployment requirement and report the uncertainty. `tuned_ranker.joblib` and `nearest_rule.joblib` remain available as alternatives.

Optuna tuning improved the LightGBM standalone student-test top-1 from 91.42% to 91.74% and reduced covered log loss from 0.257 to 0.242, but it **hurt decoded CV accuracy**: tuned minus untuned was -0.85 percentage points, with a paired 95% trip-bootstrap interval of [-1.71, -0.22]. This is why the tuned parameters were not deployed. The earlier rule-default selection only compared tuned LightGBM with nearest road and omitted this stronger learned challenger.

Student-test results in the later review were 88.53% decoded point accuracy and 92.16% route F1 for untuned LightGBM, versus 88.51% and 93.13% for nearest road, and 87.87% and 93.65% for the tuned ranker. These metrics measure different outcomes. The selection correction occurred after evaluator outcomes had already been viewed in this project; it was based only on the saved, earlier training CV records. Do not present it as blind evaluator holdout selection.

## Honest limits

The ordered alignment creates proxy point labels from supplied route sequences; it does not pair route rows directly to GPS rows. Reference routes can be far from GPS points. The trusted subset and exclusions are recorded in `results/review/protocol.json`, while all-point accuracy retains candidate misses. The short CSV notebook is a reproducibility aid and cannot stand in for the full route experiment. A notebook without executed fresh-training cells is a code recipe plus saved evidence, not proof of that session's execution.
