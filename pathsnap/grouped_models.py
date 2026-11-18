"""Observation-level objectives and candidate model inference."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


def softmax(scores, counts):
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    values = np.exp(scores - np.repeat(np.maximum.reduceat(scores, starts), counts))
    return values / np.repeat(np.add.reduceat(values, starts), counts)


def fit_temperature(scores, target, counts):
    if not np.any(target):
        return 1.
    def objective(log_slope):
        probability = softmax(scores * np.exp(log_slope), counts)
        return -np.log(np.maximum(probability[target], 1e-12)).mean()
    choice = minimize_scalar(objective, bounds=(-4., 4.), method='bounded')
    return float(np.exp(choice.x))


def observation_scores(scores, target, counts, slope=1.):
    probability = softmax(scores * slope, counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    winners = scores == np.repeat(np.maximum.reduceat(scores, starts), counts)
    first = np.minimum.reduceat(np.where(winners, np.arange(len(scores)), len(scores)), starts)
    truth_probability = np.add.reduceat(probability * target, starts)
    return float(target[first].mean()), float(-np.log(np.maximum(truth_probability, 1e-12)).mean())


class ConditionalChoice:
    """Linear conditional logit or one-hidden-layer grouped softmax network."""
    def __init__(self, hidden=0):
        self.hidden = hidden

    def fit(self, x, target, counts, validation=None):
        self.mean_ = x.mean(axis=0)
        self.scale_ = np.maximum(x.std(axis=0), 1e-6)
        x = np.clip((x - self.mean_) / self.scale_, -12., 12.)
        if validation is not None:
            valid_x, valid_target, valid_counts = validation
            valid_x = np.clip((valid_x-self.mean_) / self.scale_, -12., 12.)
        d, h = x.shape[1], self.hidden
        if not h:
            def loss(w):
                p = softmax(x @ w, counts)
                return (-np.log(np.maximum(p[target], 1e-12)).mean() + 1e-4 * (w @ w),
                        x.T @ (p - target) / len(counts) + 2e-4 * w)
            self.weight_ = minimize(loss, np.zeros(d), jac=True, method='L-BFGS-B',
                                    options={'maxiter': 45}).x
            return self
        rng = np.random.default_rng(42)
        starts = np.r_[0, np.cumsum(counts)]
        w = rng.normal(0, .2, (d, h)).astype(np.float32)
        v = rng.normal(0, .2, h).astype(np.float32)
        bias = np.zeros(h, dtype=np.float32)
        pms = [w, v, bias]
        first = [np.zeros_like(p) for p in pms]
        second = [np.zeros_like(p) for p in pms]
        self.loss_history_ = []
        step = 0
        for epoch in range(8):
            for group in np.array_split(rng.permutation(len(counts)), max(1, len(counts) // 768)):
                lengths = counts[group]
                offsets = np.r_[0, np.cumsum(lengths)[:-1]]
                rows = np.repeat(starts[group] - offsets, lengths) + np.arange(lengths.sum())
                batch = x[rows]
                act = np.tanh(batch @ w + bias)
                pred = softmax(act @ v, lengths)
                residual = (pred - target[rows]).astype(np.float32) / len(group)
                hidden_gradient = residual[:, None] * v * (1. - act * act)
                gradients = [batch.T @ hidden_gradient + 1e-4*w, act.T @ residual + 1e-4*v,
                             hidden_gradient.sum(axis=0)]
                step += 1
                for param, a, b, gradient in zip(pms, first, second, gradients):
                    a[:] = .9*a + .1*gradient
                    b[:] = .999*b + .001*gradient*gradient
                    param -= .012 * (a / (1-.9**step)) / (np.sqrt(b / (1-.999**step)) + 1e-7)
            train_prob = softmax(np.tanh(x @ w + bias) @ v, counts)
            record = {'epoch': epoch+1, 'train_grouped_log_loss':
                      float(-np.log(np.maximum(train_prob[target], 1e-12)).mean())}
            if validation is not None:
                valid_prob = softmax(np.tanh(valid_x @ w + bias) @ v, valid_counts)
                record['valid_grouped_log_loss'] = float(
                    -np.log(np.maximum(valid_prob[valid_target], 1e-12)).mean())
            self.loss_history_.append(record)
        self.weight_, self.output_, self.bias_ = pms
        return self

    def score(self, x):
        x = np.clip((x-self.mean_) / self.scale_, -12., 12.)
        if self.hidden:
            return np.tanh(x @ self.weight_ + self.bias_) @ self.output_
        return x @ self.weight_


class NumericTreeEnsemble:
    """Portable predictor for a fitted numeric LightGBM ranking ensemble."""
    def __init__(self, model):
        self.trees = []
        for tree in model.booster_.dump_model()['tree_info']:
            features, thresholds, left, right, outputs = [], [], [], [], []

            def flatten(node):
                index = len(features)
                features.append(node.get('split_feature', -1))
                thresholds.append(node.get('threshold', np.nan))
                left.append(-1)
                right.append(-1)
                outputs.append(node.get('leaf_value', 0.))
                if 'split_feature' in node:
                    if node['decision_type'] != '<=':
                        raise ValueError('Expected numeric LightGBM splits')
                    left[index] = flatten(node['left_child'])
                    right[index] = flatten(node['right_child'])
                return index

            flatten(tree['tree_structure'])
            self.trees.append(tuple(np.asarray(a) for a in (features, thresholds, left, right, outputs)))

    def score(self, x):
        result = np.zeros(len(x))
        for feature, threshold, left, right, output in self.trees:
            nodes = np.zeros(len(x), dtype=int)
            rows = np.flatnonzero(feature[nodes] >= 0)
            while len(rows):
                n = nodes[rows]
                go_left = x[rows, feature[n]].astype(float) <= threshold[n]
                nodes[rows] = np.where(go_left, left[n], right[n])
                rows = rows[feature[nodes[rows]] >= 0]
            result += output[nodes]
        return result


def fit_model(name, x, target, counts, params=None, threads=4, validation=None):
    params = params or {}
    if name in ('Conditional logit', 'Grouped MLP'):
        return ConditionalChoice(16 if name == 'Grouped MLP' else 0).fit(x, target, counts, validation)
    if name == 'ExtraTrees':
        return ExtraTreesClassifier(n_estimators=24, max_depth=15, min_samples_leaf=14,
                                    n_jobs=threads, random_state=42).fit(x, target)
    if name == 'LightGBM ranker':
        from lightgbm import LGBMRanker
        return LGBMRanker(objective='lambdarank', n_estimators=90, num_leaves=31, learning_rate=.08,
                          max_bin=63, min_child_samples=40, n_jobs=threads, verbosity=-1,
                          random_state=42).fit(x, target.astype(int), group=counts)
    config = dict(max_iter=65, l2_regularization=.3, min_samples_leaf=50, max_depth=7,
                  early_stopping=False, max_bins=63, random_state=42)
    config.update(params)
    if validation is not None:
        valid_x, valid_y, valid_counts = validation

        def grouped_scorer(estimator, evaluation_x, y):
            # sklearn also requests a diagnostic score on a sampled training set.
            # Only the external whole-trip validation score drives early stopping.
            if len(y) != len(valid_y) or not np.array_equal(y, valid_y):
                return 0.
            p = np.clip(estimator.predict_proba(evaluation_x)[:, 1], 1e-6, 1-1e-6)
            return -observation_scores(np.log(p / (1-p)), valid_y, valid_counts)[1]

        config.update(early_stopping=True, scoring=grouped_scorer, n_iter_no_change=8, tol=2e-4)
        return HistGradientBoostingClassifier(**config).fit(x, target, X_val=valid_x, y_val=valid_y)
    return HistGradientBoostingClassifier(**config).fit(x, target)


def model_scores(model, x):
    if isinstance(model, (ConditionalChoice, NumericTreeEnsemble)):
        return model.score(x)
    if hasattr(model, 'decision_function'):
        return model.decision_function(x)
    if hasattr(model, 'predict_proba'):
        p = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1-1e-6)
        return np.log(p / (1.-p))
    return model.booster_.predict(x, raw_score=True)
