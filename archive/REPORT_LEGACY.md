# PathSnap technical results

## Candidate generation and model selection

The dataset has 80 training trips and 10 student test trips. Candidate generation retrieves nearby directed arcs with a spatial index and projects onto their endpoint segments. It keeps up to five within 75 m, with a nearest-arc fallback. Features use snap distance, directed position and heading, arc length, recent motion and candidate rank. Training samples up to 250 observations per trip and uses four group folds holding out entire trips.

The supplied model comparison tested Logistic Regression, KNN, Decision Tree, Random Forest and HistGradientBoosting. Their mean cross-validation macro F1 scores were 0.8793, 0.8921, 0.8938, 0.9004 and 0.9014 respectively. An eight-trial Bayesian search varied the learning rate and leaf count for the winning HistGradientBoosting classifier. The exported model used a 0.07247 learning rate and 29 maximum leaves with CV macro F1 of 0.9019. See `results/model_comparison.csv`, `fold_metrics.csv` and `bayesian_trials.csv` in the results directory. A separate fresh local run completed the same workflow and kept the untuned configuration by CV. The demo and batch evaluation use the supplied artifact and its paired scores.

## Directed graph and decoder

Each arc retains its source row ID and direction. The graph stores outgoing arc IDs, including parallel and reverse arcs. Each observation state contains a candidate arc and its snapped position. The emission is the log of its clipped, within-observation normalized positive-class score. The transition compares GPS displacement with network travel from the previous snapped position to the current one. Network travel includes partial endpoint arcs and a bounded directed Dijkstra connector. Forward movement on a single arc uses the difference in snapped positions. Transition cost is `abs(network_m - gps_m) / 100`. Travel above `45 × elapsed_seconds + 100` metres is infeasible. Viterbi maximizes cumulative emission minus transition cost and backtracks the winning state at each observation.

The decoder starts a new segment when candidate sets cannot connect, a timestamp fails to advance, or the gap exceeds 30 seconds. Empty candidate sets return null. Connector IDs stay separate from the chosen observation arc. The baseline takes the highest scoring candidate independently.

## Student test results

The development label heuristic projects onto reference arcs, with a direction penalty when recent motion supports it. Labels are excluded beyond 45 m or when the best two costs are within 1 m. Both methods score against the same 16,854 covered observations out of 19,160 (87.96%). The candidate miss rate on covered observations was 1.32%. Labels are geometric proxies and can be wrong at intersections or overlapping roads.

| Metric | Classifier only | Viterbi |
| --- | ---: | ---: |
| Point accuracy on covered observations | 85.68% | 86.29% |
| Direct-neighbor continuity | 90.06% | 94.70% |
| Ordered route overlap F1 | 55.56% | 66.19% |

Continuity counts successive non-null arc pairs that are identical or share a directed endpoint. It excludes intermediate connectors and measures local adjacency, independent of reference accuracy. Ordered route overlap is twice the longest common subsequence length divided by the combined predicted and reference traversal lengths. It collapses consecutive redundant arc IDs and includes decoder connectors. Aggregate ratios pool counts across trips. Trip `00000052` improved from 94.09% to 98.31% in point accuracy. Trip `00000007` declined from 89.65% to 84.55%, and `00000065` declined from 88.21% to 86.37%. Connectivity alone does not resolve which of several plausible roads was used.

## Maps and deployment evidence

`results/development_map_00000052.html` overlays GPS, candidate roads, decoded arcs, dashed connectors and an optional development reference layer. The live UI omits the reference layer and requests each reconstruction from FastAPI. All ten development trips were exercised over HTTP and the browser rendered a live result for trip `00000052`. The frozen and live runs differ by one point in each method on trip `00000077`. The published table uses the frozen score export for both methods.

The server has access logging enabled for the Laptop 1 terminal. The browser on Laptop 2 connects to the server's LAN address. The actual two-device rehearsal, venue screenshots and reserved trip evaluation require those devices and evaluator files.

## Work provenance

The supplied Member 1 classifier and training evidence are included with their matching score export. Codex assisted with implementation, integration, verification and this report draft. The repository preserves the measured batch and live results independently.
