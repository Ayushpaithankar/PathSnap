# PathSnap

PathSnap matches noisy GPS readings to directed road segments and reconstructs a route. A fitted **LightGBM LambdaRank** model scores candidate arcs, **Viterbi** selects a sequence, and **FastAPI** serves the route and metrics to a browser map.

**Start here:** [Colab notebook](notebooks/PathSnap_Colab.ipynb) · [Technical report](PathSnap_Technical_Report.pdf) · [Presentation guide](PathSnap_Presentation_Guide.md) · [Two-laptop setup](EVALUATOR_SETUP.md)

[View the dashboard screenshot](results/review/evaluator_ui_loaded.png) · [View the development map screenshot](results/review/demo_map.png)

## Results

| Dataset | Decoded point accuracy |
| --- | ---: |
| Student test, 10 trips | **93.17%** |
| Evaluator follow-up, 10 trips | **95.03%** |
| Evaluator trip `00000015` | **92.53%**, up from 61.28% |

Point accuracy is the share of GPS observations assigned to the reference-aligned directed arc. Ordered route F1 compares the predicted and reference arc sequences. Trip `00000015` reached **57.63% route F1** and reported **316 route breaks**. The evaluator failure prompted the decoder update, and student-test results informed the restart penalty, so the updated figures are follow-up results. See the [per-trip results](results/review/restart_validation.csv) and [validation protocol](REVIEW_VALIDATION.md).

## Run locally

Clone the repository and enter its root. Use Python 3.11–3.13 and put the supplied student dataset at `inputs/Student folder.zip`. The ZIP contains 80 training and 10 student-test trips. The optional evaluator folder is supplied separately.

**macOS or Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000 --access-log
```

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000 --access-log
```

Open [the local dashboard](http://127.0.0.1:8000) and select **Student test → 00000052 → Reconstruct route**. The map’s background tiles require internet access.

### Two-laptop demo

1. On the server laptop, set `PATHSNAP_EVALUATOR_ROOT` before starting Uvicorn. Use `export PATHSNAP_EVALUATOR_ROOT="/path/to/evaluator_test"` on macOS/Linux or `$env:PATHSNAP_EVALUATOR_ROOT = "C:\path\to\evaluator_test"` in PowerShell.
2. Start Uvicorn with `--host 0.0.0.0` and keep the access logs visible. Connect both computers to the same network.
3. On the UI laptop, open `http://<SERVER-LAN-IP>:8000` using the **server’s** IP. Select **Evaluator test → 00000015 → Reconstruct route**. See the [full demo setup](EVALUATOR_SETUP.md).

The API lists trips at `GET /api/trips`, lists scoring methods at `GET /api/models`, and reconstructs routes at `POST /api/predict`. It reads references after prediction to compute metrics. See the [API implementation](pathsnap/api.py) or open [interactive API docs](http://127.0.0.1:8000/docs) while the server is running.

## How the pipeline works

1. **Generate candidates.** Snap each GPS point to up to eight nearby directed arcs. Build features for distance, heading, motion, graph connectivity, and trajectory support.
2. **Prepare training labels.** Align GPS points to the ordered reference route with a beam search. During inference, the candidate generator uses GPS and the road graph.
3. **Rank candidates.** LightGBM scores the arcs within each GPS observation. The pipeline normalizes the scores within each candidate group for Viterbi.
4. **Decode the route.** Viterbi combines candidate scores with directed path feasibility and movement costs. The deployed decoder can restart a segment at a fixed cost of 20 and records a break where it does.

### Important source files

| Area | Source files |
| --- | --- |
| Candidate generation and features | [Candidate scoring](pathsnap/candidates.py), [grouped features](pathsnap/grouped_features.py) |
| Label alignment and preparation | [Ordered labels](pathsnap/ordered_labels.py), [dataset preparation](pathsnap/prepare_grouped.py) |
| Models and grouped cross-validation | [Model implementations](pathsnap/grouped_models.py), [five-model training](pathsnap/grouped_training.py) |
| Bayesian optimization and release export | [Optuna search and assessment](pathsnap/review_experiment.py#L108), [refit and export](pathsnap/reselect_release.py) |
| Graph, decoder, and metrics | [Graph and Viterbi](pathsnap/core.py), [reconstruction and scoring](pathsnap/pipeline.py) |
| API, browser UI, and map | [FastAPI server](pathsnap/api.py), [dashboard HTML](pathsnap/ui.html), [Folium export](pathsnap/maps.py) |
| Verification and report | [Live API checks](pathsnap/verify_demo.py), [automated tests](tests/), [report generator](build_current_report.py) |

Browse [all source files](pathsnap/) and [archived experiments](archive/README.md).

## Training and model selection

The initial experiment compared conditional logit, grouped MLP, ExtraTrees, HistGradientBoosting, and LightGBM using two repeats of eight trip-grouped folds. Nearest road and distance plus heading were rule baselines. A later Optuna TPE search ran 18 LightGBM trials. Tuning improved standalone scoring, but the untuned LightGBM configuration had the highest trusted decoded cross-validation accuracy among trained candidates, **87.94%**. Nearest road plus Viterbi scored **87.55%**, and the paired interval included zero.

| What to open | Files |
| --- | --- |
| Main walkthrough and training output | [Colab notebook](notebooks/PathSnap_Colab.ipynb), [executed training record](notebooks/PathSnap_Training_Proof.ipynb), [model training evidence](MODEL_TRAINING_EVIDENCE.md) |
| Hosted Google Colab run | [Original notebook with saved output](PathSnap_Colab.ipynb), [five-model comparison](results/colab_hosted/five_model_cv.csv), [18 Optuna trials](results/colab_hosted/optuna_trials_displayed.csv), [student-test summary](results/colab_hosted/student_test_summary.csv) |
| Compact CSV demonstration | [CSV training notebook](notebooks/PathSnap_CSV_Training_Colab.ipynb) |
| Five-model experiment | [CV summary](results/ordered/cv_summary.csv), [fold scores](results/ordered/cv_folds.csv), [student-test comparison](results/ordered/student_test_comparison.csv) |
| Bayesian search | [Optuna trials](results/review/ranker_optuna_trials.csv), [best parameters](results/review/ranker_parameters.json), [search plot](results/review/ranker_optuna_history.png) |
| Release decision | [Selection record](results/review/selection.json), [decoder settings from assessment](results/review/decoder_parameters.json), [decoder update results](results/review/restart_validation.csv) |
| Fitted artifacts | [Deployed LightGBM](results/review/deployed.joblib), [tuned LightGBM](results/review/tuned_ranker.joblib), [nearest-road rule](results/review/nearest_rule.joblib) |

The primary notebook includes locally executed output. In Colab, upload the project ZIP containing the student data and run Sections 1–4 for the walkthrough or Section 5 for training and export.

The hosted run used an earlier project archive. Its saved model selection and metrics describe that experiment. The [primary notebook](notebooks/PathSnap_Colab.ipynb) documents the current release.

### Reproduce the training stages

Run from the project root in the activated virtual environment. Use [requirements-training.txt](requirements-training.txt) for training and [requirements.txt](requirements.txt) for inference.

```bash
python -m pip install -r requirements-training.txt
python -m pathsnap.prepare_grouped "inputs/Student folder.zip" work/features
python -m pathsnap.grouped_training work/features work/family-cv --mode compare --repeats 2
python -m pathsnap.prepare_grouped "inputs/Student folder.zip" work/context --context
python -m pathsnap.review_experiment "inputs/Student folder.zip" work/context work/review --phase ranker --trials 18
python -m pathsnap.review_experiment "inputs/Student folder.zip" work/context work/review --phase evaluate
python -m pathsnap.reselect_release work/context work/review --restart-penalty 20
```

The full assessment takes longer than the notebook walkthrough. The refit writes an artifact to `work/review/deployed.joblib`. To serve it, set `PATHSNAP_MODEL` to that file and restart the API.

### Verify the demo

Run the second command while the API is up and the evaluator folder is configured.

```bash
python -m pytest -q
python -m pathsnap.verify_demo --api http://127.0.0.1:8000 --partition evaluator_test --all-methods
```
