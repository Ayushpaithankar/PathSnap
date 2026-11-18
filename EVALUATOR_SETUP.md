# Frozen evaluator demonstration

The release scorer was selected using training-trip CV evidence after evaluator results had been seen. An observed failure on evaluator trip 00000015 later motivated the segment restart option. The restart penalty was selected using student-test trips. Evaluator references are used for server-side metrics after prediction. Keep the evaluator folder on Laptop 1, outside the repository.

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

This verifies all exposed methods on ten IDs, observation ordering, geometry response shape, reference arc isolation, and a missing-trip 404. Capture the two-laptop setup and Laptop 1 API console during the event.
