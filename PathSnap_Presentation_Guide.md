# PathSnap presentation guide

## 1. Open with the problem · 30 seconds

“We start with a sequence of noisy GPS points and a directed road graph. A point can be close to two parallel roads, a junction, or both directions of the same road. Our goal is to assign a directed road segment to each point and reconstruct a plausible route. We use LightGBM to score candidate arcs and Viterbi to make the route decisions across the whole trip.”

Show the map first if time is tight. Blue points are GPS readings. The road layer shows candidate roads. Use **Arc scores**, **Viterbi**, and **Both** to compare the independent choices with the reconstructed route. The GPS point slider lets you inspect where a specific reading falls.

## 2. Show how we prepared the data · 1–2 minutes

Open notebook sections **1. Inspect the files and the fitted release** and **2. Turn GPS points into training examples**.

1. **Read each trip.** The `.track` file contains longitude, latitude, and time. The `.nodes` and `.arcs` files define the directed road network. The `.route` file is an ordered list of traversed arcs. Preserve trip IDs like `00000052` and the original zero-based directed arc IDs. The raw student bundle supplies 80 training trips and 10 student-test trips. The separate evaluator set has 10 trips.
2. **Generate candidates.** Project each point onto nearby directed arcs. Search within 100 m, keep up to eight arcs, and use a nearest fallback when needed. Keep alternate directions as distinct candidates. An observation whose reference arc is absent is a candidate miss.
3. **Create the target in route order.** Use beam search to align the trip to the reference sequence, balancing distance from the GPS point with plausible forward movement along the route. The resulting arc for each GPS point is a proxy label. A reference arc can cover several GPS points, so the two files have different row meanings.
4. **Build features from GPS and the road graph.** Explain snap distance, whether the candidate points in the direction of travel, and whether nearby readings also support that arc. The full 28-feature release set also covers stationary points, signed offset, reverse twins, endpoint degrees, connectivity to adjacent candidates, and 15- and 30-second motion context. The model can use later observations because the demo processes a complete recorded trip.
5. **Filter for final fitting quality.** The initial comparison used all 80 training trips. In the later review, a trusted alignment was within 75 m of the reference. Two trips had more than half their points over 100 m from the reference and were excluded. The final fit used 78 trips and 180,365 trusted observations with a reference candidate. All 19,160 student-test observations still count in point accuracy, including candidate misses.

**What to point at in the notebook:** The sample table prints several directed arcs for one GPS observation with different distances, headings, and scores. Use the reference column to show how a supervised label was assigned.

## 3. Explain the experiments and model choice · 1–2 minutes

Open notebook section **3. Explain how LightGBM was selected**. The initial family comparison used eight folds repeated twice, grouped by whole trip. Keeping all points from a trip together prevents neighboring readings from the same trip appearing in both training and validation.

| Initial grouped CV method | Observation top-1 |
|---|---:|
| Nearest road rule | 49.23% |
| Distance plus heading rule | 68.84% |
| Conditional logit | 76.71% |
| Grouped MLP | 77.28% |
| ExtraTrees | 77.05% |
| HistGradientBoosting | 77.77% |
| LightGBM LambdaRank | **78.00%** |

“LambdaRank matches our decision. Each GPS observation is a group of competing arcs, and the model learns to rank the correct arc higher within that group. It led our initial comparison on top-1 and grouped log loss. HGB was close, so the family comparison alone shows a small margin.”

The later feature and decoder experiment used a different protocol. After the two exclusions, 26 training trips were used for development and 52 for grouped assessment. Optuna TPE evaluated 18 LightGBM settings using grouped validation log loss. An export applies calibrated within-observation scores before Viterbi.

| Later review configuration | Trusted decoded CV | Student top-1 before decoding | Student covered log loss |
|---|---:|---:|---:|
| Nearest road + Viterbi | 87.55% | 52.95% | 0.940 |
| HistGradientBoosting + Viterbi | 86.65% | 90.54% | 0.268 |
| LightGBM tuned + Viterbi | 87.09% | **91.74%** | **0.242** |
| LightGBM untuned + Viterbi | **87.94%** | 91.42% | 0.257 |

“We chose the fitted untuned LightGBM configuration because it led the learned methods on the held-out decoded CV metric. Tuning helped standalone scores and log loss, but reduced decoded CV accuracy by about 0.85 percentage points. The decoded gain over nearest road was about 0.40 points, and its paired confidence interval included zero.”

During questions about overfitting, point to whole-trip CV and the separate student test. The MLP loss curve in the technical report illustrates where validation loss flattened. The close CV scores and subsequent decoder adjustment limit how broadly we can generalize the measured gains.

## 4. Explain Viterbi and the fix · 1–2 minutes

“Think of every GPS point as a column of candidate roads. Each candidate gets an emission score from LightGBM. To move between columns, Viterbi checks whether the directed road graph allows travel between the snapped positions. It penalizes a large mismatch between road travel and the GPS step, considers elapsed time and reversals, and keeps the best cumulative path. At the end it backtracks to recover the chosen arcs.”

“Trip 00000015 was a failure case. LightGBM alone correctly picked the arc on 91.09% of points, but the old decoder reached 61.28%. It could stay on a weakly supported path because an unbroken route was still technically feasible. We added a second choice at each step: start a new segment from the best score so far, pay a fixed cost, and leave the gap visible. We tested candidate costs on the student-test trips and used 20 log-score units. This is one general decoder rule applied to all trips.”

| Evaluator trip 00000015 | Earlier decoder | Updated decoder |
|---|---:|---:|
| Decoded point accuracy | 61.28% | **92.53%** |
| Ordered route overlap (F1) | 35.65% | **57.63%** |

The updated result matched 6,342 of 6,854 readings and reported **316 breaks**. This is a useful limitation to show live. A higher point score can coexist with a fragmented route. Direct continuity counts consecutive chosen arcs with touching endpoints or the same arc. It measures a different property from decoder break events or agreement with the reference sequence.

The evaluator error motivated the change. The penalty comparison used the student-test set, and the evaluator rerun was a follow-up on a previously viewed case. Student-test decoded accuracy rose from 88.53% to 93.17% across ten trips after the update. Because the student-test results informed the penalty choice, present 93.17% as a development comparison. Three other evaluator trips lost up to 1.09 percentage points.

## 5. Show how the model ran on test and how to give the demo · 1–2 minutes

1. Before the review, start FastAPI on Laptop 1 with the evaluator folder configured and access logs enabled. Use `--host 0.0.0.0` for the two-laptop demo. Connect both laptops to the same network and open `http://<Laptop-1-IP>:8000` on Laptop 2.
2. Start on **Student test → 00000052** if you want a rehearsal. Then choose **Evaluator test → 00000015 → LightGBM untuned** and click **Reconstruct route**.
3. Explain the request. Laptop 2 sends a trip ID and selected method. Laptop 1 loads the fitted joblib artifact, regenerates the same candidate features from the selected trip’s GPS and graph, scores them, and runs Viterbi. The test trip is processed by the frozen model and decoder. The server opens the reference afterward to compute metrics and returns predicted geometry and aggregate scores.
4. Toggle **Arc scores** and **Viterbi**. Point to the **92.5%** point accuracy, **57.6%** ordered route overlap, and **316** route events. Open **View route events** and show a segment break. On Laptop 1 show the incoming `POST /api/predict` access log.
5. Open notebook section **4. Reproduce current student inference** and show the live test table. Then open **5. Rerun training** to show actual grouped fold and Optuna logs. Explain that the saved outputs were executed locally. To produce a Colab session record, run section 5 in Colab and save its outputs.

### Metrics when you get a follow-up question

- **Top-1 or point accuracy** counts correct directed arc choices divided by the labeled GPS readings. A missed reference candidate counts as an error. Use the before and after columns to distinguish the scorer from the full pipeline.
- **Grouped log loss** measures how much normalized probability the model assigns to the correct candidate. Lower is better. Covered log loss is calculated for observations where the reference arc is available among the candidates.
- **Candidate macro F1** used in the historical report averages F1 for positive and negative candidate rows. It serves as a supplementary comparison. The team selected the release on decoded CV point accuracy.
- **Ordered route F1** uses the longest common subsequence of predicted and reference arcs. It equals twice the number of matched arcs divided by the sum of the two route lengths.
- **Precision and recall** answer different questions. For the ordered arc sequence, precision is matched arcs divided by predicted route length, and recall is matched arcs divided by reference route length. Their harmonic mean is the route F1 shown in the UI. Point accuracy answers the separate per-GPS question.

### Technical details to have ready

- **Feature selection:** Compare the tuned ranker with and without the nine additional context features. Trusted decoded CV was 87.09% with context and 86.96% without it. Student-test standalone top-1 was 91.74% with context and 88.58% without. Present these as the measured ablation and distinguish the small decoded CV difference from the larger standalone test difference.
- **Untuned LightGBM configuration:** 90 trees, 31 maximum leaves per tree, a 0.08 learning rate, and at least 40 candidate rows per leaf. The later tuned model used 110 trees, 20 leaves and a depth cap of four. The ranking score is scaled and passed through a softmax over each observation’s candidates to obtain emissions.
- **Data quality check:** The supplied development data had no invalid GPS coordinates or timestamps in the inventory. It contained 135 gaps over 30 seconds across the 90 trips. Directed reachability and the speed checks determine whether the decoder can bridge a gap.
- **Where the code lives:** `ordered_labels.py` aligns reference routes, `grouped_features.py` builds features, `grouped_training.py` runs family CV, `review_experiment.py` runs tuning and assessment, `reselect_release.py` refits and exports, `core.py` decodes, and `api.py` and `ui.html` serve the demo.
