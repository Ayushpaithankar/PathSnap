"""Fit decoder parameters on training folds, select by decoded accuracy, and export."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .core import RoadGraph, decode, load_reference, load_trip
from .grouped_features import GROUP_FEATURES
from .grouped_models import NumericTreeEnsemble, fit_model, model_scores, observation_scores, softmax
from .grouped_training import GROUPED_MODELS, MODELS, RULES, calibrate, fitting_arrays, pooled, read_split, rule_scores, safe_name
from .pipeline import lcs_length, sequence
from .prepare_grouped import SETTINGS


CONFIGS = [
    dict(scale=1., max_gap_s=30., max_speed_mps=45., emission_weight=1.),
    dict(scale=2., max_gap_s=15., max_speed_mps=38., emission_weight=.7),
    dict(scale=4., max_gap_s=60., max_speed_mps=55., emission_weight=1.),
    dict(scale=3., max_gap_s=25., max_speed_mps=45., emission_weight=1.5),
    dict(scale=6., max_gap_s=45., max_speed_mps=35., emission_weight=2.),
    dict(scale=5., max_gap_s=90., max_speed_mps=60., emission_weight=.6),
]


def fitted_scale(trips, excluded=()):
    # One residual per consecutive observation from the ordered route alignment.
    errors = np.concatenate([d.residuals[1:] for t, d in trips.items() if t not in excluded])
    return max(1., float(errors.mean()))


def decoder_parameters(config, fitted_beta):
    return dict(beta_m=fitted_beta * config['scale'], max_gap_s=config['max_gap_s'],
                max_speed_mps=config['max_speed_mps'], emission_weight=config['emission_weight'],
                reversal_penalty=2., stationary_speed=1., bridge_gaps=True)


def group_rows(data, probabilities):
    return {i: [{'arc_id': int(data.arc[k]), 'snapped_lon': data.snaps[k, 0],
                 'snapped_lat': data.snaps[k, 1], 'probability': float(probabilities[k])}
                for k in range(first, last)]
            for i, (first, last) in enumerate(zip(np.r_[0, np.cumsum(data.counts)[:-1]],
                                                  np.cumsum(data.counts)))}


def route_result(data, track, graph, groups, cfg, reference=None):
    pred, links, warnings = decode(track, graph, groups, **cfg)
    chosen = np.array([a if a is not None else -1 for a in pred])
    score = int((chosen == data.labels).sum())
    row = dict(correct=score, observations=len(chosen), decoded_top1=score/len(chosen),
               events=len(warnings), bridged_gaps=sum('bridged' in w for w in warnings),
               unmatched=int((chosen < 0).sum()))
    if reference is not None:
        ordered = sequence(pred, links)
        row['sequence_matches'] = lcs_length(ordered, reference)
        row['sequence_length_sum'] = len(ordered) + len(reference)
        row['sequence_f1'] = 2 * row['sequence_matches'] / row['sequence_length_sum']
    return row


def load_oof(folder, name):
    with np.load(folder / f'oof_{safe_name(name)}_1.npz') as file:
        predictions = dict(file)
    rows = pd.read_csv(folder / f'folds_{safe_name(name)}.csv').query('repeat == 1')
    by_trip = {t: (row.fold, row.slope, row.trips.split()) for row in rows.itertuples()
               for t in row.trips.split()}
    return predictions, by_trip


def tune_decoder(data, trips, fit_dir, out, source):
    initial, folds = load_oof(fit_dir, source)
    scales = {fold: fitted_scale(trips, ids) for fold, _, ids in folds.values()}
    records = []
    for trip_id, trip in trips.items():
        fold, slope, _ = folds[trip_id]
        track, nodes, arcs = load_trip(data, 'train', trip_id)
        graph = RoadGraph(nodes, arcs)
        groups = group_rows(trip, softmax(initial[trip_id] * slope, trip.counts))
        for k, config in enumerate(CONFIGS):
            row = route_result(trip, track, graph, groups, decoder_parameters(config, scales[fold]))
            records.append(dict(config=k, fold=fold, trip_id=trip_id, **row))
        print('decoder tune', trip_id, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'decoder_tuning_per_trip.csv', index=False)
    totals = frame.groupby('config')[['correct', 'observations', 'events', 'bridged_gaps']].sum()
    totals['accuracy'] = totals.correct / totals.observations
    for key in CONFIGS[0]:
        totals[key] = [c[key] for c in CONFIGS]
    totals.to_csv(out / 'decoder_tuning.csv')
    best = int(totals.accuracy.idxmax())
    (out / 'decoder_search.json').write_text(json.dumps(dict(source=source, config=best,
        best=CONFIGS[best], configs=CONFIGS), indent=2) + '\n')


def select(data, trips, student, fit_dir, out, threads, reuse_decoder_cv=False):
    names = RULES + MODELS
    config = json.loads((out / 'decoder_search.json').read_text())['best']
    oof = {name: load_oof(fit_dir, name) for name in names}
    fits = pd.concat([pd.read_csv(fit_dir / f'folds_{safe_name(name)}.csv') for name in names])
    calibration = []
    for name, first in oof.items():
        for repeat in sorted(fits.repeat.unique()):
            if repeat == 1:
                scores = first[0]
            else:
                with np.load(fit_dir / f'oof_{safe_name(name)}_{repeat}.npz') as file:
                    scores = dict(file)
            for row in fits[(fits.model == name) & (fits.repeat == repeat)].itertuples():
                _, loss = pooled(trips, {t: scores[t] for t in row.trips.split()})
                calibration.append(dict(model=name, repeat=repeat, fold=row.fold,
                    uncalibrated_grouped_log_loss=loss, calibrated_grouped_log_loss=row.grouped_log_loss,
                    calibration_slope=row.slope))
    pd.DataFrame(calibration).to_csv(out / 'calibration_cv.csv', index=False)
    fits.to_csv(out / 'cv_folds.csv', index=False)
    for filename in ('tuned_hgb.json', 'optuna_trials.csv'):
        (out / filename).write_bytes((fit_dir / filename).read_bytes())
    base = fits[fits.model == RULES[1]].set_index(['repeat', 'fold']).top1
    comparisons = []
    for name, rows in fits.groupby('model', sort=False):
        diffs = rows.set_index(['repeat', 'fold']).top1 - base
        comparisons.append(dict(model=name, top1=rows.top1.mean(),
                                grouped_log_loss=rows.grouped_log_loss.mean(),
                                paired_top1_delta_vs_rule=diffs.mean(),
                                paired_top1_delta_sd=diffs.std()))
    pd.DataFrame(comparisons).to_csv(out / 'cv_summary.csv', index=False)
    if reuse_decoder_cv:
        frame = pd.read_csv(out / 'decoder_cv_trips.csv', dtype={'trip_id': str})
        if set(zip(frame.model, frame.trip_id)) != {(name, t) for name in names for t in trips}:
            raise ValueError('Cached decoder comparison does not cover the current methods and trips')
    else:
        decoder_rows = []
        _, common = oof[RULES[1]]
        scales = {fold: fitted_scale(trips, ids) for fold, _, ids in common.values()}
        for key, trip in trips.items():
            track, nodes, arcs = load_trip(data, 'train', key)
            graph = RoadGraph(nodes, arcs)
            for name, (predictions, folds) in oof.items():
                fold, slope, _ = folds[key]
                groups = group_rows(trip, softmax(predictions[key] * slope, trip.counts))
                row = route_result(trip, track, graph, groups, decoder_parameters(config, scales[fold]))
                decoder_rows.append(dict(model=name, fold=fold, trip_id=key, **row))
            print('decoder comparison', key, flush=True)
        frame = pd.DataFrame(decoder_rows)
        frame.to_csv(out / 'decoder_cv_trips.csv', index=False)
    paired = frame.groupby(['model', 'fold'])[['correct', 'observations']].sum()
    paired['accuracy'] = paired.correct / paired.observations
    pivot = paired.accuracy.unstack('model')
    totals = frame.groupby('model')[['correct', 'observations']].sum()
    totals['decoded_accuracy'] = totals.correct / totals.observations
    totals['paired_delta_vs_heading'] = (pivot.sub(pivot[RULES[1]], axis=0)).mean()
    totals['paired_delta_sd'] = (pivot.sub(pivot[RULES[1]], axis=0)).std()
    totals.to_csv(out / 'decoder_cv_summary.csv')
    overall_best = str(totals.decoded_accuracy.idxmax())
    best_learned = str(totals.loc[list(MODELS), 'decoded_accuracy'].idxmax())
    best = str(totals.loc[list(GROUPED_MODELS), 'decoded_accuracy'].idxmax())
    # Prefer the simpler choice if the paired gain is both small and inside one descriptive SE.
    winner = best
    for name in GROUPED_MODELS:
        delta = pivot[best] - pivot[name]
        pooled_delta = totals.loc[best, 'decoded_accuracy'] - totals.loc[name, 'decoded_accuracy']
        if name == best or pooled_delta <= min(.002, delta.std()/np.sqrt(len(delta))):
            winner = name
            break
    params = json.loads((fit_dir / 'tuned_hgb.json').read_text()) if winner == MODELS[-1] else None
    model = fit_model(winner, *fitting_arrays(trips, sorted(trips)), params=params,
                      threads=threads) if winner not in RULES else None
    slope = calibrate(trips, oof[winner][0])
    beta = fitted_scale(trips)
    cfg = decoder_parameters(config, beta)
    deployed = NumericTreeEnsemble(model) if winner == 'LightGBM ranker' else model
    if deployed is not model:
        for trip in student.values():
            np.testing.assert_allclose(model_scores(deployed, trip.x), model_scores(model, trip.x),
                                       atol=1e-10, rtol=1e-10)
    artifact = dict(version='ordered-v2', name=winner, model=deployed, features=GROUP_FEATURES,
                    calibration_slope=slope, decoder=cfg, candidate_settings=SETTINGS)
    joblib.dump(artifact, out / 'ordered_model.joblib', compress=3)
    # Refit every challenger on all training trips after selection; student test is untouched until here.
    preds = {name: {} for name in names}
    for name in names:
        challenger = model if name == winner else (
            fit_model(name, *fitting_arrays(trips, sorted(trips)),
                      params=json.loads((fit_dir / 'tuned_hgb.json').read_text()) if name == MODELS[-1] else None,
                      threads=threads) if name not in RULES else None)
        preds[name] = {t: rule_scores(name, d) if name in RULES else model_scores(challenger, d.x)
                       for t, d in student.items()}
    slopes = {name: calibrate(trips, oof[name][0]) for name in names}
    rows = []
    for t, d in student.items():
        track, nodes, arcs = load_trip(data, 'student_test', t)
        graph = RoadGraph(nodes, arcs)
        ref = load_reference(data, 'student_test', t)
        for name, scores in preds.items():
            score = scores[t]
            top1, loss = observation_scores(score, d.target, d.counts, slopes[name])
            groups = group_rows(d, softmax(score * slopes[name], d.counts))
            route = route_result(d, track, graph, groups, cfg, ref)
            rows.append(dict(model=name, trip_id=t, classifier_top1=top1, grouped_log_loss=loss,
                             candidate_misses=len(d.labels)-int(d.target.sum()), **route))
        print('student test', t, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / 'student_test_per_trip.csv', index=False)
    summary = []
    for name, group in results.groupby('model', sort=False):
        n = int(group.observations.sum())
        top1, loss = pooled(student, preds[name], slopes[name])
        summary.append(dict(model=name, selected=name==winner, observations=n, classifier_top1=top1,
                            grouped_log_loss=loss, decoded_top1=group.correct.sum()/n,
                            sequence_f1=2*group.sequence_matches.sum()/group.sequence_length_sum.sum()))
    pd.DataFrame(summary).to_csv(out / 'student_test_comparison.csv', index=False)
    total_train = sum(len(d.labels) for d in trips.values())
    train_covered = sum(int(d.target.sum()) for d in trips.values())
    (out / 'configuration.json').write_text(json.dumps(dict(model=winner,
        best_grouped_cv_model=best, best_learned_cv_model=best_learned,
        best_overall_cv_method=overall_best,
        training_observations=total_train, training_observations_with_candidate=train_covered,
        training_candidate_misses=total_train-train_covered, fitted_exponential_beta_m=beta,
        calibration_slope=slope, decoder=cfg, student_test_observations=sum(len(d.labels) for d in student.values()),
        student_test_candidate_misses=sum(len(d.labels)-int(d.target.sum()) for d in student.values())),
        indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('fit_dir', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--tune-decoder', action='store_true')
    parser.add_argument('--source', default='Conditional logit')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--reuse-decoder-cv', action='store_true',
                        help='Reuse a decoder replay with the same models, data and decoder parameters')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trips = read_split(args.prepared / 'train')
    with threadpool_limits(limits=args.threads):
        if args.tune_decoder:
            tune_decoder(args.data, trips, args.fit_dir, args.out, args.source)
        else:
            select(args.data, trips, read_split(args.prepared / 'student_test'),
                   args.fit_dir, args.out, args.threads, args.reuse_decoder_cv)


if __name__ == '__main__':
    main()
