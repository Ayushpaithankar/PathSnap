"""Probe the final decoder settings locally and extend boundary coordinates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .grouped_features import CONTEXT_FEATURES, RULES
from .grouped_models import softmax
from .grouped_training import read_split
from .review_experiment import RANKER, GRID, arrays, bundle, decoder_args, group_rows, residual_beta, score, train_model
from .core import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    directory = args.out
    protocol = json.loads((directory / 'protocol.json').read_text())
    saved = json.loads((directory / 'decoder_parameters.json').read_text())
    params = json.loads((directory / 'ranker_parameters.json').read_text())
    dev = protocol['development_trips']
    trips = {t: d for t, d in read_split(args.prepared / 'train').items() if t in dev}
    quality = {}
    for t in dev:
        with np.load(args.prepared / 'train' / (t + '.npz')) as file:
            quality[t] = file['alignment_distance']
    tune_ids = saved['decoder_development_trips']
    fit_ids = sorted(set(dev) - set(tune_ids))
    beta = residual_beta(trips, quality, fit_ids)
    with threadpool_limits(limits=2):
        model = train_model(RANKER, arrays(trips, quality, fit_ids, len(CONTEXT_FEATURES)), 2, params)
    graphs = {t: bundle(args.data, 'train', t) for t in tune_ids}
    records = []
    for name in (RANKER, RULES[0]):
        active = saved['configs'][name]
        candidates = {t: group_rows(trips[t], softmax(score(name, model, trips[t]) *
                      saved['slopes'][name], trips[t].counts)) for t in tune_ids}
        for key, values in (GRID | {'max_gap_s': [5., 10., *GRID['max_gap_s']],
                                    'max_speed_mps': [16., 22., *GRID['max_speed_mps']]}).items():
            for val in values:
                config = decoder_args(active | {key: val}, beta)
                n = correct = 0
                for t, (track, graph, _) in graphs.items():
                    ids, _, _ = decode(track, graph, candidates[t], **config)
                    keep = quality[t] <= 75.
                    correct += int(((np.asarray(ids) == trips[t].labels) & keep).sum())
                    n += int(keep.sum())
                records.append(dict(method=name, coordinate=key, value=val,
                                    selected=val == active[key], correct=correct,
                                    observations=n, accuracy=correct/n))
    pd.DataFrame(records).to_csv(directory / 'decoder_final_profiles.csv', index=False)


if __name__ == '__main__':
    main()
