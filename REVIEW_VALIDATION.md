# Review validation

The review release now deploys **LightGBM untuned**, the trained method with the highest trusted decoded accuracy in the held-out training CV. The API loads `results/review/deployed.joblib` after restart. The dashboard also offers tuned LightGBM and nearest road as comparisons. The Optuna-tuned ranker improved standalone point accuracy and covered log loss but reduced decoded CV accuracy versus untuned LightGBM; `results/review/selection.json` records the paired interval. The selection correction was made after evaluator data had been supplied, so it must not be described as a pristine pre-evaluator decision. It uses the pre-existing training CV table and does not use evaluator scores or labels for fitting or tuning.

## Reproduce

Install the training requirements. On macOS, LightGBM training requires the Homebrew `libomp` runtime. The exported API artifacts run with the base requirements.

```sh
python -m pathsnap.prepare_grouped "inputs/Student folder.zip" work/review-data --context
python -m pathsnap.review_experiment "inputs/Student folder.zip" work/review-data results/review --phase ranker
python -m pathsnap.review_experiment "inputs/Student folder.zip" work/review-data results/review --phase evaluate
python -m pathsnap.review_profiles "inputs/Student folder.zip" work/review-data results/review
python -m pathsnap.audit_review_release "inputs/Student folder.zip" work/review-data results/review
python -m pathsnap.review_report
```

## Protocol

- Keep ordered route labels. Exclude a training trip when more than half its points project more than 100 m from their assigned reference arc. Its counts remain in the protocol. Within each usable trip, score the trusted subset at 75 m or less and report results on all points separately. Fit only on trusted points with a reference candidate. The trusted log loss uses covered points, while accuracy counts candidate misses as errors.
- The 26 development trips are disjoint from the 52 trips assessed in eight-fold CV. Use the development pool for Optuna trials, calibration and method-specific decoder sweeps. Refit each CV model with the assessment fold held out. Student test is outside all choices. Exact trip IDs are saved in the protocol and decoder parameter files.
- Compare both rules, default LightGBM, tuned LightGBM, an ablation of the 15/30-second motion and adjacent-candidate features, and an HGB challenger. The ranker tuner selects from 18 TPE trials using trusted, covered grouped log loss. Longer motion windows stop at clock breaks and large gaps. Topology support connects candidate heads and tails in both temporal directions and weights them by distance and heading.
- Sweep one decoder coordinate at a time for each method. Revisit emission weight and transition scale once after the first pass. The follow-up profile changes one value from the final settings and probes below the original speed and gap boundaries on development trips. This is a local sensitivity check.
- Choose the highest trusted decoded CV accuracy among trained methods for deployment. Report paired trip-bootstrap intervals against nearest road and tuned versus untuned LightGBM rather than treating sub-point differences as decisive. Keep nearest road as the explicit baseline. Preserve raw ordered route F1 as a separate check against GPS-to-route alignment bias. Student-test uncertainty is sampled by whole trip, not by individual GPS points.

See [measured results](results/review/RESULTS.md) and the evidence CSVs in `results/review/`. The older ordered benchmark remains in `results/ordered/`.
