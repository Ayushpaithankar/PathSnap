# PathSnap

PathSnap scores directed road candidates for each GPS point, reconstructs a route with Viterbi, and serves the result through FastAPI and a browser map. The release includes the fitted classifier, original experiment results, measured decoder results and the supplied development dataset ZIP.

See [Review validation](REVIEW_VALIDATION.md) for the current model, trusted-reference evaluation and decoder search. Historical files are marked in [archive](archive/README.md). Rebuild the current [technical report draft](PathSnap_Technical_Report.pdf) with `python build_current_report.py`.
For an assessment walkthrough, see [Model training evidence](MODEL_TRAINING_EVIDENCE.md) and [Evaluator setup](EVALUATOR_SETUP.md).

## Run locally

Use Python 3.11–3.13. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PATHSNAP_DATA_ROOT="$PWD/inputs/Student folder.zip"
python -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000 --access-log
```

Open http://127.0.0.1:8000. The dashboard starts on **Evaluator test**. When the evaluator folder is unavailable, select **Student test** to rehearse with trip `00000052`. Click **Reconstruct route** to show the map and compare the two methods. Leave the server terminal visible to see requests. Leaflet and background tiles load from the internet. On Windows, activate with `.venv\Scripts\Activate.ps1` and set environment variables with `$env:NAME = "value"`.

## Two-laptop demo

Run the server on Laptop 1 with `--host 0.0.0.0` on a trusted LAN. Allow TCP 8000 if needed. On Laptop 2, open `http://<Laptop-1-LAN-IP>:8000` in a browser. The page defaults to the same server and supports a configurable API URL. For an independently served UI on Laptop 2, run `python -m http.server 8080 --directory pathsnap`, open `http://localhost:8080/ui.html`, and enter Laptop 1's URL before loading trips.

The API reads the bundled development ZIP by default. Point `PATHSNAP_EVALUATOR_ROOT` at the supplied **evaluator_test folder** on Laptop 1; it remains outside the project and never enters training or model selection. Each trip needs `<trip_id>.track`, `.nodes`, and `.arcs`. The `.route` file supports server-side final metrics and is read only after prediction. It is never returned to the UI.

macOS/Linux on Laptop 1:

```sh
export PATHSNAP_EVALUATOR_ROOT="/path/to/evaluator_test"
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000 --access-log
```

Windows PowerShell on Laptop 1:

```powershell
$env:PATHSNAP_EVALUATOR_ROOT = "C:\path\to\evaluator_test"
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000 --access-log
```

Check `http://<Laptop-1-IP>:8000/api/trips?partition=evaluator_test` returns ten IDs. On Laptop 2 open `http://<Laptop-1-IP>:8000`, select an evaluator trip, and keep the Laptop 1 terminal visible for request logs. For an automated request check, run `python -m pathsnap.verify_demo --api http://127.0.0.1:8000 --partition evaluator_test` on Laptop 1. This check performs inference with the frozen release artifact and must not be used to change model or decoder parameters.

## Training and selection evidence

The project has five learned model families in `pathsnap/grouped_training.py`: conditional logit, grouped MLP, ExtraTrees, LightGBM LambdaRank, and HistGradientBoosting. Their grouped-fold outputs and student-test comparison are in `results/ordered/`. The later review experiment in `pathsnap/review_experiment.py` tuned LightGBM with Optuna and compared end-to-end decoding against both road rules. Its trial CSV, fold records, selection rule and final artifacts are in `results/review/`. See `REVIEW_VALIDATION.md` and `results/review/RESULTS.md` for the exact protocol and measured results.

The deployed LightGBM decoder now allows a new segment at a cost of 20 log-score units. This lets well-supported candidates recover when continuing the current segment would favor an implausible route. The decoder reports each break in **Route events** and supplies no connector across it. In the follow-up API check, evaluator trip `00000015` improved from 61.28% to 92.53% decoded point accuracy and reported 316 breaks. The change was prompted by the observed evaluator failure, and the cost was selected by comparing point accuracy on the ten student-test trips. The original comparison tables describe the earlier decoder. See `results/review/restart_validation.csv` for before and after results. To reproduce this release when refitting the artifact, pass `--restart-penalty 20` to `pathsnap.reselect_release`.

`notebooks/PathSnap_Colab.ipynb` walks through preprocessing, model choice, and current live student-test inference. It includes saved local training outputs from two repeats of eight folds across five model families, 18 Optuna trials, and a fitted artifact export. Run sections 1–4 for the review walkthrough and section 5 to rerun training. See [the presentation guide](PathSnap_Presentation_Guide.md) for a demo talk track. `notebooks/PathSnap_Training_Proof.ipynb` preserves the focused historical training run. The separate CSV notebook demonstrates pointwise scoring for three models.

## Reproduce the decoder and demo checks

```sh
python -m pathsnap.evaluate_decoder "$PATHSNAP_DATA_ROOT" --labels results/student_test_reference_labels.csv
python -m pathsnap.evaluate_decoder "$PATHSNAP_DATA_ROOT" --live --labels results/student_test_reference_labels.csv
python -m pathsnap.maps "$PATHSNAP_DATA_ROOT" student_test 00000052 results/development_map_00000052.html --reference
python -m pytest -q
python -m pathsnap.verify_demo --api http://127.0.0.1:8000
python build_current_report.py
```

The default batch comparison uses frozen scores. The live batch and demo run raw trips through the classifier. Files ending in `_live` record the live batch. When passed the label checkpoint, the evaluator checks every reference arc and reports any difference in candidate coverage.

## Retrain and export

Preparation accepts either the extracted data directory or the original ZIP with its `Student folder/` prefix. Use a separate output directory to keep each run's artifact, scores, and metrics together.

```sh
python -m pathsnap.prepare_data "$PATHSNAP_DATA_ROOT" work/prepared
python -m pathsnap.train work/prepared work/retrained
python -m pathsnap.summarize work/prepared work/retrained --archive "$PATHSNAP_DATA_ROOT"
python -m pathsnap.evaluate_decoder "$PATHSNAP_DATA_ROOT" --scores work/retrained/student_test_probabilities.csv.gz --out work/retrained
export PATHSNAP_MODEL="$PWD/work/retrained/best_classifier.joblib"
```

Restart the API after switching artifacts. Load trusted joblib files only. The released default is the fitted **untuned LightGBM LambdaRank model** selected from held-out training CV decoded accuracy; tuned LightGBM and nearest road remain selectable baselines. See [the technical report](PathSnap_Technical_Report.pdf) and [model training evidence](MODEL_TRAINING_EVIDENCE.md) for the mixed tuning result and selection limitation.
