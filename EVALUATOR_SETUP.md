# Frozen evaluator demonstration

The release artifact and decoder parameters use training-trip CV evidence. The trained-model default was corrected after the evaluator data had been supplied; the correction did not fit or tune on evaluator rows, but must not be described as a blind pre-evaluator choice. The ten supplied evaluator trips are **only** for final API inference and server-side metrics. Do not use their `.route` files for model fitting, feature selection, tuning, or hardcoded outputs. Keep the evaluator folder on Laptop 1, outside the repository.

## Laptop 1

From the `pathsnap` project directory, install `requirements.txt` in a Python 3.11–3.13 environment. Then start the server with access logs:

macOS/Linux:

```sh
export PATHSNAP_EVALUATOR_ROOT="/absolute/path/to/evaluator_test"
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000 --access-log
```

Windows PowerShell:

```powershell
$env:PATHSNAP_EVALUATOR_ROOT = "C:\absolute\path\to\evaluator_test"
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000 --access-log
```

The folder contains ten eight-digit trip subfolders with `.track`, `.nodes`, `.arcs`, and optional `.route` files. The API lists the ten IDs at `/api/trips?partition=evaluator_test`. Prediction uses only track/nodes/arcs and the frozen artifact. It reads `.route` after reconstruction to calculate aggregate metrics, then sends only metrics and predicted geometry to the UI.

## Laptop 2

Connect both laptops to the same network. Open `http://<Laptop-1-LAN-IP>:8000` in a browser. The UI starts on **Evaluator test** and loads the ten choices from the API. Select a trip and press **Reconstruct route**. Its map displays GPS, candidate roads, the independent arc-score route, and the Viterbi route; no reference geometry is sent to the browser. Keep the Laptop 1 terminal visible to show incoming `GET` and `POST` logs. If the browser is served separately on Laptop 2, use the UI's API settings to enter Laptop 1's URL.

For a frozen automated check on Laptop 1:

```sh
python -m pathsnap.verify_demo --api http://127.0.0.1:8000 --partition evaluator_test --all-methods
```

This verifies both exposed methods on all ten IDs, observation ordering, geometry response shape, no reference arc IDs in the response, and a missing-trip 404. It must not feed back into model selection. The report still needs actual photographs or screenshots of the physical two-laptop setup and API console during the event.
