"""Repeated trip-grouped comparison and grouped-objective TPE tuning."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .grouped_features import RULES, baseline_scores
from .grouped_models import fit_model, fit_temperature, model_scores, observation_scores


MODELS = ('Conditional logit', 'Grouped MLP', 'ExtraTrees', 'LightGBM ranker', 'HistGradientBoosting')
GROUPED_MODELS = ('Conditional logit', 'Grouped MLP', 'LightGBM ranker')


@dataclass
class TripData:
    trip_id: str
    x: np.ndarray
    obs: np.ndarray
    arc: np.ndarray
    labels: np.ndarray
    residuals: np.ndarray
    snaps: np.ndarray
    timestamps: np.ndarray

    @property
    def counts(self):
        return np.bincount(self.obs, minlength=len(self.labels))

    @property
    def target(self):
        return self.arc == self.labels[self.obs]


def read_split(folder):
    result = {}
    for file in sorted(folder.glob('*.npz')):
        with np.load(file) as d:
            result[file.stem] = TripData(file.stem, *(d[k] for k in
                ('features', 'obs', 'arc', 'labels', 'residuals', 'snaps', 'timestamps')))
    return result


def folds(trips, seed, n_folds=8):
    """Shuffle ties and balance held-out folds by total observation count."""
    rng = np.random.default_rng(seed)
    order = sorted(rng.permutation(sorted(trips)), key=lambda t: -len(trips[t].labels))
    # Randomize trip membership as well as tie order, then greedily balance.
    order = sorted(order, key=lambda t: -(len(trips[t].labels) * rng.uniform(.75, 1.25)))
    groups = [[] for _ in range(n_folds)]
    totals = np.zeros(n_folds, dtype=int)
    for trip in order:
        k = int(totals.argmin())
        groups[k].append(trip)
        totals[k] += len(trips[trip].labels)
    return groups


def fitting_arrays(trips, ids):
    rows, targets, sizes = [], [], []
    for key in ids:
        d = trips[key]
        target = d.target
        eligible = np.bincount(d.obs[target], minlength=len(d.labels)) > 0
        keep = eligible[d.obs]
        rows.append(d.x[keep])
        targets.append(target[keep])
        sizes.append(d.counts[eligible])
    return np.concatenate(rows), np.concatenate(targets), np.concatenate(sizes)


def rule_scores(name, data):
    return baseline_scores(name, data.x)


def pooled(trips, predictions, slope=1.):
    count = sum(len(trips[t].labels) for t in predictions)
    values = [(len(trips[t].labels), *observation_scores(score, trips[t].target, trips[t].counts, slope))
              for t, score in predictions.items()]
    return tuple(sum(v[0] * v[k] for v in values) / count for k in (1, 2))


def calibrate(trips, predictions):
    return fit_temperature(np.concatenate(list(predictions.values())),
                           np.concatenate([trips[t].target for t in predictions]),
                           np.concatenate([trips[t].counts for t in predictions]))


def safe_name(name):
    return name.lower().replace(' ', '_').replace('+', 'plus')


def compare(trips, out, names, params, repeats, threads):
    for name in names:
        report = []
        for repeat in range(repeats):
            test_folds = folds(trips, 42 + repeat)
            predictions = {}
            for k, holdout in enumerate(test_folds):
                train_ids = sorted(set(trips) - set(holdout))
                if name not in RULES:
                    model = fit_model(name, *fitting_arrays(trips, train_ids),
                                      params=params if name == MODELS[-1] else None, threads=threads)
                scores = {t: rule_scores(name, trips[t]) if name in RULES else
                          model_scores(model, trips[t].x) for t in holdout}
                predictions.update(scores)
                top1, nll = pooled(trips, scores)
                print(name, repeat+1, k+1, f'top1={top1:.4f}, nll={nll:.4f}', flush=True)
            for k, holdout in enumerate(test_folds):
                slope = calibrate(trips, {t: s for t, s in predictions.items() if t not in holdout})
                top1, loss = pooled(trips, {t: predictions[t] for t in holdout}, slope)
                report.append(dict(model=name, repeat=repeat+1, fold=k+1, top1=top1, grouped_log_loss=loss,
                                   slope=slope, trips=' '.join(sorted(holdout))))
            np.savez_compressed(out / f'oof_{safe_name(name)}_{repeat+1}.npz', **predictions)
        pd.DataFrame(report).to_csv(out / f'folds_{safe_name(name)}.csv', index=False)


def tune(trips, out, trials, threads):
    import optuna
    selected = folds(trips, 42)[:2]
    cached = [(fitting_arrays(trips, sorted(set(trips)-set(ids))),
               tuple(np.concatenate([getattr(trips[t], key) for t in ids])
                     for key in ('x', 'target', 'counts'))) for ids in selected]
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))

    def objective(trial):
        params = dict(l2_regularization=trial.suggest_float('l2_regularization', 1e-3, 8., log=True),
                      min_samples_leaf=trial.suggest_int('min_samples_leaf', 18, 130),
                      max_depth=trial.suggest_int('max_depth', 4, 10),
                      max_iter=trial.suggest_int('max_iter', 24, 96, step=8))
        losses = []
        best_stages = []
        actual_stages = []
        for k, (train, validation) in enumerate(cached):
            model = fit_model(MODELS[-1], *train, params=params, threads=threads, validation=validation)
            stage = int(np.argmax(model.validation_score_[1:])) + 1
            loss = -float(model.validation_score_[stage])
            actual_stages.append(model.n_iter_)
            best_stages.append(stage)
            losses.append(loss)
            trial.report(float(np.mean(losses)), step=k)
            if k == 0 and trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('validation_selected_iterations', best_stages)
        trial.set_user_attr('iterations_grown', actual_stages)
        return float(np.mean(losses))

    study.optimize(objective, n_trials=trials)
    params = study.best_params
    params['max_iter'] = int(np.mean(study.best_trial.user_attrs['validation_selected_iterations']))
    (out / 'tuned_hgb.json').write_text(json.dumps(params, indent=2) + '\n')
    study.trials_dataframe().to_csv(out / 'optuna_trials.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prepared', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--mode', choices=['compare', 'tune'], default='compare')
    parser.add_argument('--names', nargs='+', default=RULES + MODELS)
    parser.add_argument('--params', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trips = read_split(args.prepared / 'train')
    if len(trips) != 80:
        parser.error('Expected all 80 training trips')
    if set(args.names) - set(MODELS + RULES):
        parser.error('Unknown model name')
    with threadpool_limits(limits=args.threads):
        if args.mode == 'tune':
            tune(trips, args.out, args.trials, args.threads)
        else:
            params = json.loads(args.params.read_text()) if args.params else None
            compare(trips, args.out, args.names, params, args.repeats, args.threads)


if __name__ == '__main__':
    main()
