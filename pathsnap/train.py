"""Compare five classifiers and tune the winner by Bayesian optimisation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from pathsnap.candidates import FEATURES, HANDOFF

MODELS = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'HistGradientBoosting']
SEED = 42


def model_for(name, params):
    if name == 'Logistic Regression':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, random_state=SEED, **params))
    if name == 'KNN':
        return make_pipeline(StandardScaler(), KNeighborsClassifier(algorithm='kd_tree', weights='distance', **params))
    if name == 'Decision Tree':
        return DecisionTreeClassifier(random_state=SEED, **params)
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=80, n_jobs=4, random_state=SEED, **params)
    return HistGradientBoostingClassifier(max_iter=100, random_state=SEED, **params)


BASE = dict(zip(MODELS, [{'C': 1.}, {'n_neighbors': 15}, {'max_depth': 10, 'min_samples_leaf': 10},
    {'max_depth': 14, 'min_samples_leaf': 5}, {'learning_rate': .1, 'max_leaf_nodes': 31}]))


def search_params(name, x):
    a, b = x
    if name == 'Logistic Regression':
        return {'C': float(10 ** (-2 + 4 * a))}
    if name == 'KNN':
        return {'n_neighbors': int(3 + 2 * round(19 * a))}
    if name in ['Decision Tree', 'Random Forest']:
        return {'max_depth': int(round(4 + 20 * a)), 'min_samples_leaf': int(round(2 + 38 * b))}
    return {'learning_rate': float(.025 * 10 ** a), 'max_leaf_nodes': int(round(12 + 52 * b))}


def read_partition(root, name, cap=None):
    rng = np.random.default_rng(SEED)
    tables = []
    for file in sorted((root / name).glob('*-candidates.pkl')):
        frame = pd.read_pickle(file)
        if cap:
            points = frame.observation_index.unique()
            if len(points) > cap:
                frame = frame[frame.observation_index.isin(rng.choice(points, cap, replace=False))]
        tables.append(frame)
    if not tables:
        raise ValueError(f'No candidate tables found for {name}')
    return pd.concat(tables, ignore_index=True)


def metrics(frame, probability):
    scored = frame.loc[:, ['trip_id', 'observation_index', 'target']].assign(probability=probability)
    pred = probability >= .5
    target = scored.target
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='macro', zero_division=0)
    top = scored.loc[scored.groupby(['trip_id', 'observation_index'], sort=False).probability.idxmax()]
    return dict(macro_f1=f1, macro_precision=precision, macro_recall=recall,
                     accuracy=float(np.mean(target == pred)), top1=float(top.target.mean()),
                     log_loss=log_loss(target, probability, labels=[0, 1]))


def evaluate(name, params, train, validation):
    model = model_for(name, params)
    start = perf_counter()
    model.fit(train[FEATURES], train.target)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    p = model.predict_proba(validation[FEATURES])[:, 1]
    predict_seconds = perf_counter() - start
    return model, p, dict(metrics(validation, p), fit_seconds=fit_seconds, predict_seconds=predict_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--cap', type=int, default=250, help='max observations sampled per training trip, 0 uses all')
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--trials', type=int, default=8)
    args = parser.parse_args()
    if args.trials < 1 or args.cap < 0:
        parser.error('Use at least one trial and a nonnegative observation cap')
    args.out.mkdir(parents=True, exist_ok=True)
    raw_train = read_partition(args.data, 'train', args.cap)
    student = read_partition(args.data, 'student_test')
    train = raw_train[raw_train.reference_arc_id >= 0].reset_index(drop=True)
    validation = student[student.reference_arc_id >= 0].reset_index(drop=True)
    folds = list(GroupKFold(args.folds).split(train, groups=train.trip_id))
    fold_log, details, summary = [], [], []

    def cv(name, params, phase):
        scores = []
        for fold, (tr, va) in enumerate(folds, 1):
            _, _, result = evaluate(name, params, train.iloc[tr], train.iloc[va].reset_index(drop=True))
            fold_log.append(dict(model=name, phase=phase, params=json.dumps(params), fold=fold, **result))
            scores.append(result['macro_f1'])
        return float(np.mean(scores)), float(np.std(scores))

    def record(name, params, phase, mean, std):
        model, probability, result = evaluate(name, params, train, validation)
        summary.append(dict(model=name, phase=phase, params=json.dumps(params), cv_macro_f1=mean,
                            cv_macro_f1_std=std, **{f'student_{k}': v for k, v in result.items()}))
        for trip_id, frame in validation.assign(probability=probability).groupby('trip_id'):
            frame = frame.reset_index(drop=True)
            details.append(dict(model=name, phase=phase, trip_id=trip_id, **metrics(frame, frame.probability.to_numpy())))
        matrix = confusion_matrix(validation.target, probability >= .5, labels=[0, 1])
        pd.DataFrame(matrix, index=['actual_0', 'actual_1'], columns=['pred_0', 'pred_1']).to_csv(
            args.out / f"confusion_{name.lower().replace(' ', '_')}_{phase}.csv")
        print(name, phase, f'CV macro F1={mean:.4f}, student top-1={result["top1"]:.4f}', flush=True)
        return model

    models = {}
    for name in MODELS:
        mean, std = cv(name, BASE[name], 'baseline')
        models[name] = record(name, BASE[name], 'baseline', mean, std)
    winner = max(summary, key=lambda r: r['cv_macro_f1'])
    name = winner['model']
    rng = np.random.default_rng(SEED)
    pool = rng.random((1200, 2))
    tried, scores, trials = [], [], []
    used = set()
    for i in range(args.trials):
        if i < 3:
            order = range(len(pool))
        else:
            gp = GaussianProcessRegressor(kernel=Matern(length_scale=.3, nu=2.5), alpha=.00001,
                                          normalize_y=True, optimizer=None)
            gp.fit(np.asarray(tried), scores)
            mu, sigma = gp.predict(pool, return_std=True)
            gain = mu - max(scores) - .001
            z = gain / np.maximum(sigma, 1e-12)
            improvement = gain * ndtr(z) + sigma * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
            order = np.argsort(-improvement)
        for j in order:
            params = search_params(name, pool[j])
            key = json.dumps(params)
            if key not in used:
                used.add(key)
                break
        else:
            break  # Discrete parameter space exhausted.
        mean, std = cv(name, params, f'trial_{i + 1}')
        tried.append(pool[j]); scores.append(mean)
        trials.append(dict(trial=i + 1, params=key, cv_macro_f1=mean, cv_macro_f1_std=std))
        print(f'Trial {i + 1}: {key} CV macro F1={mean:.4f}', flush=True)
    best = max(trials, key=lambda r: r['cv_macro_f1'])
    params = json.loads(best['params'])
    tuned = record(name, params, 'tuned', best['cv_macro_f1'], best['cv_macro_f1_std'])
    use_tuned = best['cv_macro_f1'] >= winner['cv_macro_f1']
    selected = tuned if use_tuned else models[name]
    if not use_tuned:
        params = BASE[name]
    configuration = dict(name=name, params=params, tuned=use_tuned, features=FEATURES,
                         candidate_settings=dict(radius=75., limit=5), seed=SEED, cap=args.cap, folds=args.folds)
    joblib.dump(dict(configuration, model=selected), args.out / 'best_classifier.joblib', compress=3)
    (args.out / 'model_config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    student['probability'] = selected.predict_proba(student[FEATURES])[:, 1]
    student[HANDOFF].to_csv(args.out / 'student_test_probabilities.csv.gz', index=False)
    pd.DataFrame(summary).to_csv(args.out / 'model_comparison.csv', index=False)
    pd.DataFrame(details).to_csv(args.out / 'per_trip_metrics.csv', index=False)
    pd.DataFrame(trials).to_csv(args.out / 'bayesian_trials.csv', index=False)
    pd.DataFrame(fold_log).to_csv(args.out / 'fold_metrics.csv', index=False)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
