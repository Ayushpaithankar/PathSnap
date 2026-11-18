"""Independent development tuning and trip-blocked map matching evaluation."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .core import RoadGraph, decode, load_reference, load_trip
from .grouped_features import CONTEXT_FEATURES, GROUP_FEATURES, RULES, baseline_scores
from .grouped_models import NumericTreeEnsemble, fit_model, fit_temperature, model_scores, softmax
from .grouped_training import folds, read_split
from .pipeline import lcs_length, sequence
from .select_grouped import group_rows


RANKER, UNTUNED, NO_CONTEXT, HGB = ('LightGBM tuned', 'LightGBM untuned',
                                    'LightGBM tuned without context', 'HistGradientBoosting')
METHODS = RULES + (UNTUNED, NO_CONTEXT, RANKER, HGB)
DEFAULT = dict(n_estimators=90, num_leaves=31, max_depth=-1, learning_rate=.08,
               min_child_samples=40, reg_lambda=0.)


def read_quality(prepared, partition, trips):
    quality = {}
    for trip in trips:
        with np.load(prepared / partition / (trip + '.npz')) as blob:
            quality[trip] = blob['alignment_distance']
    return quality


def setup(prepared, out):
    train = read_split(prepared / 'train')
    quality = read_quality(prepared, 'train', train)
    bad = sorted(t for t, q in quality.items() if np.mean(q > 100.) > .5)
    usable = {t: d for t, d in train.items() if t not in bad}
    dev = sorted(folds(usable, 718, 3)[0])
    cv = sorted(set(usable) - set(dev))
    (out / 'protocol.json').write_text(json.dumps(dict(excluded_trips=bad, development_trips=dev,
        assessment_cv_trips=cv, trusted_radius_m=75, broken_trip_rule='more than half above 100 m',
        train_observations=sum(len(d.labels) for d in train.values()),
        excluded_observations=sum(len(train[t].labels) for t in bad)), indent=2) + '\n')
    return train, quality, usable, dev, cv


def arrays(trips, quality, ids, columns):
    xs, ys, counts = [], [], []
    for t in ids:
        d = trips[t]
        target = d.target
        valid = (quality[t] <= 75.) & (np.bincount(d.obs[target], minlength=len(d.labels)) > 0)
        use = valid[d.obs]
        xs.append(d.x[use, :columns])
        ys.append(target[use])
        counts.append(d.counts[valid])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(counts)


def train_model(name, data, threads, params):
    x, y, counts = data
    if name == HGB:
        return fit_model(HGB, x, y, counts, params={'max_iter': 48}, threads=threads)
    from lightgbm import LGBMRanker
    config = DEFAULT | (params if name in (RANKER, NO_CONTEXT) else {})
    return LGBMRanker(**config, objective='lambdarank', max_bin=63, n_jobs=threads,
                      verbosity=-1, random_state=42).fit(x, y.astype(int), group=counts)


def score(name, model, trip):
    if name in RULES:
        return baseline_scores(name, trip.x)
    n = len(GROUP_FEATURES) if name == NO_CONTEXT else len(CONTEXT_FEATURES)
    return model_scores(model, trip.x[:, :n])


def point_stats(values, trip, quality, slope):
    counts, target = trip.counts, trip.target
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    p = softmax(values * slope, counts)
    winners = values == np.repeat(np.maximum.reduceat(values, starts), counts)
    first = np.minimum.reduceat(np.where(winners, np.arange(len(values)), len(values)), starts)
    correct = target[first]
    prob = np.add.reduceat(p * target, starts)
    covered = np.add.reduceat(target.astype(int), starts) > 0
    trusted = quality <= 75.
    valid = trusted & covered
    return dict(point_correct=int(correct.sum()), trusted_point_correct=int(correct[trusted].sum()),
                observations=len(correct), trusted_observations=int(trusted.sum()),
                misses=int((~covered).sum()), trusted_misses=int((trusted & ~covered).sum()),
                covered_trusted=int(valid.sum()), covered_log_loss_sum=float(-np.log(np.maximum(prob[valid], 1e-12)).sum()),
                all_log_loss_sum=float(-np.log(np.maximum(prob, 1e-12)).sum()))


def calibrate(predictions, trips, quality, ids):
    scores, targets, counts = [], [], []
    for t in ids:
        d = trips[t]
        valid = quality[t] <= 75.
        rows = valid[d.obs]
        scores.append(predictions[t][rows])
        targets.append(d.target[rows])
        counts.append(d.counts[valid])
    return fit_temperature(np.concatenate(scores), np.concatenate(targets), np.concatenate(counts))


def tune_ranker(usable, quality, dev, out, trials, threads):
    import optuna
    subset = {t: usable[t] for t in dev}
    internal = folds(subset, 721, 3)
    caches = [(arrays(subset, quality, sorted(set(dev)-set(ids)), len(CONTEXT_FEATURES)), ids)
              for ids in internal[:2]]
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial):
        params = dict(n_estimators=trial.suggest_int('n_estimators', 40, 150, step=10),
            num_leaves=trial.suggest_int('num_leaves', 12, 56, step=4),
            max_depth=trial.suggest_int('max_depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', .025, .15, log=True),
            min_child_samples=trial.suggest_int('min_child_samples', 25, 120),
            reg_lambda=trial.suggest_float('reg_lambda', .01, 8., log=True))
        rows = []
        for dataset, ids in caches:
            model = train_model(RANKER, dataset, threads, params)
            rows += [point_stats(score(RANKER, model, subset[t]), subset[t], quality[t], 1.) for t in ids]
        loss = sum(r['covered_log_loss_sum'] for r in rows) / sum(r['covered_trusted'] for r in rows)
        return loss

    study.enqueue_trial(DEFAULT | {'max_depth': 10, 'num_leaves': 32, 'reg_lambda': .01})
    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(out / 'ranker_optuna_trials.csv', index=False)
    (out / 'ranker_parameters.json').write_text(json.dumps(study.best_params, indent=2) + '\n')


def features_for(name):
    return GROUP_FEATURES if name == NO_CONTEXT else CONTEXT_FEATURES


def fit_and_score(usable, quality, train_ids, valid, params, threads):
    matrix = arrays(usable, quality, train_ids, len(CONTEXT_FEATURES))
    models = {}
    predictions = {}
    for name in METHODS:
        if name not in RULES:
            n = len(features_for(name))
            models[name] = train_model(name, (matrix[0][:, :n], *matrix[1:]), threads, params)
        predictions[name] = {t: score(name, models.get(name), d) for t, d in valid.items()}
    return models, predictions


def residual_beta(usable, quality, train_ids):
    errors = [usable[t].residuals[1:][(quality[t][1:] <= 75.) & (quality[t][:-1] <= 75.)]
              for t in train_ids]
    return max(1., float(np.concatenate(errors).mean()))


GRID = dict(emission_weight=[0., .25, .5, .75, 1., 1.5, 2.],
            beta_multiplier=[.5, 1., 2., 4., 6., 10.],
            max_gap_s=[15., 30., 60., 90., 120.],
            max_speed_mps=[28., 40., 52., 64., 80.])


def decoder_args(config, beta):
    return dict(emission_weight=config['emission_weight'],
                beta_m=beta * config['beta_multiplier'], max_gap_s=config['max_gap_s'],
                max_speed_mps=config['max_speed_mps'], reversal_penalty=2.,
                stationary_speed=1., bridge_gaps=True)


def tune_decoder(graphs, trips, quality, scores, slope, beta, name):
    batches = [(t, graphs[t], group_rows(trips[t], softmax(s*slope, trips[t].counts)))
               for t, s in scores.items()]
    records = []
    chosen = dict(emission_weight=.75, beta_multiplier=4., max_gap_s=60., max_speed_mps=52.)
    cached = {}

    def measure(config):
        key = tuple(config[k] for k in GRID)
        if key not in cached:
            correct = total = 0
            for t, (track, graph, _), groups in batches:
                ids, _, _ = decode(track, graph, groups, **decoder_args(config, beta))
                keep = quality[t] <= 75.
                correct += int(((np.asarray(ids) == trips[t].labels) & keep).sum())
                total += int(keep.sum())
            cached[key] = correct / total
        return cached[key]

    # Sweep one coordinate at a time, then revisit emission and beta for their interaction.
    for step, key in enumerate([*GRID, 'emission_weight', 'beta_multiplier'], 1):
        candidates = []
        for value in GRID[key]:
            config = chosen | {key: value}
            acc = measure(config)
            records.append(dict(method=name, sweep=step, coordinate=key, top1=acc, **config))
            candidates.append((acc, value))
        chosen[key] = max(candidates, key=lambda v: (v[0], -abs(v[1]-chosen[key])))[1]
    return chosen, records


def bundle(data, partition, trip_id):
    track, nodes, arcs = load_trip(data, partition, trip_id)
    return track, RoadGraph(nodes, arcs), load_reference(data, partition, trip_id)


def measure_trip(data, q, values, slope, loaded, decoder):
    track, graph, reference = loaded
    rows = group_rows(data, softmax(values * slope, data.counts))
    ids, links, _ = decode(track, graph, rows, **decoder)
    truth = data.labels
    correct = np.asarray(ids) == truth
    confidence = q <= 75.
    order = sequence(ids, links)
    result = point_stats(values, data, q, slope)
    result.update(decoded_correct=int(correct.sum()), trusted_decoded_correct=int(correct[confidence].sum()),
                  route_matches=lcs_length(order, reference), sequence_total=len(order)+len(reference))
    return result, ids


def paired_interval(frame, method, baseline, outcome, count, seed=2027):
    pivot = frame.pivot(index='trip_id', columns='method', values=outcome)
    denom = frame.drop_duplicates('trip_id').set_index('trip_id').loc[pivot.index, count].to_numpy()
    difference = (pivot[method] - pivot[baseline]).to_numpy()
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(denom), (10000, len(denom)))
    gains = difference[draw].sum(axis=1) / np.maximum(denom[draw].sum(axis=1), 1)
    return dict(delta=float(difference.sum()/denom.sum()), low=float(np.quantile(gains, .025)),
                high=float(np.quantile(gains, .975)), wins=int((difference > 0).sum()),
                losses=int((difference < 0).sum()), ties=int((difference == 0).sum()))


def summarize(frame):
    rows = []
    for name, g in frame.groupby('method', sort=False):
        s = g.sum(numeric_only=True)
        rows.append(dict(method=name, trips=len(g), observations=int(s.observations),
                         trusted_observations=int(s.trusted_observations), candidate_misses=int(s.misses),
                         trusted_misses=int(s.trusted_misses), trusted_point_top1=s.trusted_point_correct/s.trusted_observations,
                         point_top1=s.point_correct/s.observations, decoded_top1=s.decoded_correct/s.observations,
                         trusted_decoded_top1=s.trusted_decoded_correct/s.trusted_observations,
                         trusted_covered_log_loss=s.covered_log_loss_sum/s.covered_trusted,
                         all_log_loss_with_miss_floor=s.all_log_loss_sum/s.observations,
                         route_f1=2*s.route_matches/s.sequence_total))
    return pd.DataFrame(rows)


def export(name, model, slopes, configs, beta, output, trips):
    saved = NumericTreeEnsemble(model) if name in (RANKER, UNTUNED, NO_CONTEXT) else model
    if model is not None and saved is not model:
        for trip in trips.values():
            np.testing.assert_allclose(score(name, saved, trip), score(name, model, trip),
                                       atol=1e-10, rtol=1e-10)
    artifact = dict(version='ordered-v2', name=name, model=saved, features=features_for(name),
                    calibration_slope=slopes[name], decoder=decoder_args(configs[name], beta),
                    candidate_settings={'radius': 100., 'limit': 8, 'context': True})
    joblib.dump(artifact, output, compress=3)


def audit_predictions(trip, loaded, values, ids):
    track, graph, reference = loaded
    on_route = set(reference)
    starts = np.r_[0, np.cumsum(trip.counts)]
    records = []
    for i, (truth, predicted) in enumerate(zip(trip.labels, ids)):
        if truth == predicted:
            continue
        r = slice(starts[i], starts[i+1])
        raw = int(trip.arc[starts[i] + np.argmax(values[r])])
        missed = not np.any(trip.arc[r] == truth)
        a, b = graph.arcs[truth]
        reverse, adjacent, gps_prefers = False, False, False
        if predicted is not None:
            c, d = graph.arcs[predicted]
            reverse = (a, b) == (d, c) and a != b
            adjacent = bool({a, b} & {c, d}) and predicted not in on_route
            p, q = track[max(0, i-2)], track[min(len(track)-1, i+2)]
            scale = np.cos(np.radians(track[i]['lat']))
            motion = np.array([(q['lon']-p['lon'])*scale, q['lat']-p['lat']])
            arc_vector = np.subtract(graph.nodes[d], graph.nodes[c]) * [scale, 1.]
            gps_prefers = reverse and np.linalg.norm(motion)*111195 >= 4. and motion @ arc_vector > 0.
        category = ('candidate miss' if missed else 'reverse arc' if reverse else 'other reference arc'
                    if predicted in on_route else 'adjacent off-route' if adjacent else
                    'unmatched' if predicted is None else 'other off-route')
        records.append(dict(trip_id=trip.trip_id, observation_index=i, reference_arc_id=truth,
                            decoded_arc_id=predicted, ranker_arc_id=raw, category=category,
                            reverse_predicted_agrees_with_gps=gps_prefers, ranker_was_correct=raw==truth))
    return records


def evaluate(args, original, quality, usable, dev, cv):
    out = args.out
    params = json.loads((out / 'ranker_parameters.json').read_text())
    internal = folds({t: usable[t] for t in dev}, 993, 4)
    calibrate_ids, tune_ids = internal[:2]
    development_pred = {name: {} for name in METHODS}
    for ids in (calibrate_ids, tune_ids):
        fit_ids = sorted(set(dev) - set(ids))
        _, prediction = fit_and_score(usable, quality, fit_ids, {t: usable[t] for t in ids},
                                      params, args.threads)
        for name in METHODS:
            development_pred[name].update(prediction[name])
    slopes = {name: calibrate(pred, usable, quality, calibrate_ids)
              for name, pred in development_pred.items()}
    beta = residual_beta(usable, quality, sorted(set(dev)-set(tune_ids)))
    graphs = {t: bundle(args.data, 'train', t) for t in tune_ids}
    tuning, configs = [], {}
    for name, pred in development_pred.items():
        config, trace = tune_decoder(graphs, usable, quality, {t: pred[t] for t in tune_ids},
                                     slopes[name], beta, name)
        tuning.extend(trace)
        configs[name] = config
        print('decoder tuned', name, config, flush=True)
    del graphs
    pd.DataFrame(tuning).to_csv(out / 'decoder_sweeps.csv', index=False)
    (out / 'decoder_parameters.json').write_text(json.dumps(dict(configs=configs, slopes=slopes,
        calibration_trips=calibrate_ids, decoder_development_trips=tune_ids), indent=2) + '\n')

    cv_records = []
    for fold, ids in enumerate(folds({t: usable[t] for t in cv}, 429), 1):
        train_ids = sorted(set(usable) - set(ids))
        _, preds = fit_and_score(usable, quality, train_ids, {t: usable[t] for t in ids}, params, args.threads)
        fold_beta = residual_beta(usable, quality, train_ids)
        for t in ids:
            loaded = bundle(args.data, 'train', t)
            for name, scores in preds.items():
                m, _ = measure_trip(usable[t], quality[t], scores[t], slopes[name], loaded,
                                    decoder_args(configs[name], fold_beta))
                cv_records.append(dict(method=name, fold=fold, trip_id=t, **m))
        print('assessment CV fold', fold, flush=True)
    cv_frame = pd.DataFrame(cv_records)
    cv_frame.to_csv(out / 'cv_per_trip.csv', index=False)
    summarize(cv_frame).to_csv(out / 'cv_summary.csv', index=False)
    ci = paired_interval(cv_frame, RANKER, RULES[0], 'trusted_decoded_correct', 'trusted_observations')
    # The assignment requires a trained candidate scorer in the deployed pipeline.
    # Select by decoded CV accuracy among learned methods; retain rules as baselines.
    learned = (UNTUNED, NO_CONTEXT, RANKER, HGB)
    cv_summary = summarize(cv_frame).set_index('method')
    selected = max(learned, key=lambda name: cv_summary.loc[name, 'trusted_decoded_top1'])
    assert not set(dev) & set(cv)
    # Holdout is loaded after selection. Neither hyperparameters nor the release choice use it.
    test = read_split(args.prepared / 'student_test')
    test_quality = read_quality(args.prepared, 'student_test', test)
    models, preds = fit_and_score(usable, quality, sorted(usable), test, params, args.threads)
    final_beta = residual_beta(usable, quality, usable)
    rng = np.random.default_rng(61)
    test_records, ablation, audit = [], [], []
    for t, trip in test.items():
        loaded = bundle(args.data, 'student_test', t)
        for name, predictions in preds.items():
            result, ids = measure_trip(trip, test_quality[t], predictions[t], slopes[name],
                                       loaded, decoder_args(configs[name], final_beta))
            test_records.append(dict(method=name, trip_id=t, **result))
            if name == RANKER:
                audit.extend(audit_predictions(trip, loaded, predictions[t], ids))
        config = decoder_args(configs[RANKER], final_beta)
        for method, values, slope, cfg in [
            ('No emissions', preds[RANKER][t], slopes[RANKER], config | {'emission_weight': 0.}),
            ('Random', rng.normal(size=len(trip.arc)), 1., config),
            (RULES[0], preds[RULES[0]][t], slopes[RULES[0]], config),
            (RANKER, preds[RANKER][t], slopes[RANKER], config),
        ]:
            result, _ = measure_trip(trip, test_quality[t], values, slope, loaded, cfg)
            ablation.append(dict(method=method, trip_id=t, **result))
        print('student test', t, flush=True)
    test_frame = pd.DataFrame(test_records)
    test_frame.to_csv(out / 'test_per_trip.csv', index=False)
    summarize(test_frame).to_csv(out / 'test_summary.csv', index=False)
    summarize(pd.DataFrame(ablation)).to_csv(out / 'emission_ablation.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'ranker_error_audit.csv', index=False)
    intervals = []
    for split, frame in [('trusted_cv', cv_frame), ('test', test_frame)]:
        keys = ('trusted_decoded_correct', 'trusted_observations') if split == 'trusted_cv' else ('decoded_correct', 'observations')
        for baseline in (RULES[0], UNTUNED, NO_CONTEXT, HGB):
            intervals.append(dict(split=split, method=RANKER, baseline=baseline,
                                  **paired_interval(frame, RANKER, baseline, *keys)))
    pd.DataFrame(intervals).to_csv(out / 'trip_bootstrap.csv', index=False)
    export(selected, models.get(selected), slopes, configs, final_beta, out / 'deployed.joblib', test)
    export(RULES[0], None, slopes, configs, final_beta, out / 'nearest_rule.joblib', test)
    export(RANKER, models[RANKER], slopes, configs, final_beta, out / 'tuned_ranker.joblib', test)
    (out / 'selection.json').write_text(json.dumps(dict(selected=selected,
        selection_policy='highest trusted decoded assessment CV accuracy among trained models; rules remain baselines',
        trusted_cv_ranker_vs_nearest=ci,
        trusted_cv_selected_vs_nearest=paired_interval(cv_frame, selected, RULES[0],
            'trusted_decoded_correct', 'trusted_observations'),
        trusted_cv_tuned_vs_untuned=paired_interval(cv_frame, RANKER, UNTUNED,
            'trusted_decoded_correct', 'trusted_observations'),
        transition_beta_m=final_beta, model_fit_trips=len(usable),
        trusted_fit_observations=len(arrays(usable, quality, sorted(usable), len(CONTEXT_FEATURES))[2])),
        indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--phase', choices=['ranker', 'evaluate'], default='evaluate')
    parser.add_argument('--trials', type=int, default=18)
    parser.add_argument('--threads', type=int, default=4)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train, quality, usable, dev, cv = setup(a.prepared, a.out)
    with threadpool_limits(limits=a.threads):
        if a.phase == 'ranker':
            tune_ranker(usable, quality, dev, a.out, a.trials, a.threads)
        else:
            evaluate(a, train, quality, usable, dev, cv)


if __name__ == '__main__':
    main()
