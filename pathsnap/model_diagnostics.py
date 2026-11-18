"""Candidate-row F1 and timing on the ordered five-model protocol."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .grouped_models import fit_model, model_scores
from .grouped_training import MODELS, fitting_arrays, read_split, safe_name
from .core import RoadGraph, continuity, load_trip


def counts_for(trips, scores, data_root=None):
    tp = fp = fn = tn = correct = observations = 0
    connected = comparable = 0
    for trip_id, d in trips.items():
        s = np.asarray(scores[trip_id])
        assert len(s) == len(d.arc)
        truth = d.target
        begins = np.r_[0, np.cumsum(d.counts)[:-1]]
        if not len(begins) or np.any(d.counts == 0):
            raise ValueError(f'Unexpected zero-candidate observation in {trip_id}')
        winners = begins + np.array([int(np.argmax(s[a:a+n])) for a,n in zip(begins,d.counts)])
        hit = int(truth[winners].sum())
        positives = int(truth.sum())
        rows = len(truth)
        obs = len(d.labels)
        tp += hit; fp += obs-hit; fn += positives-hit
        tn += rows - hit - (obs-hit) - (positives-hit)
        correct += hit; observations += obs
        if data_root is not None:
            _, nodes, arcs = load_trip(data_root, 'student_test', trip_id)
            stats = continuity(d.arc[winners].tolist(), RoadGraph(nodes, arcs))
            connected += stats['connected']; comparable += stats['comparable']
    positive_f1 = 2*tp/(2*tp+fp+fn)
    negative_f1 = 2*tn/(2*tn+fp+fn)
    result = dict(candidate_macro_f1=(positive_f1+negative_f1)/2,
                  candidate_positive_f1=positive_f1, top1=correct/observations,
                  observations=observations, candidate_rows=tp+fp+fn+tn)
    if data_root is not None:
        result['classifier_continuity'] = connected/comparable if comparable else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prepared', type=Path)
    parser.add_argument('oof_dir', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--data', type=Path, default=Path('inputs/Student folder.zip'))
    args = parser.parse_args()
    train = read_split(args.prepared / 'train')
    test = read_split(args.prepared / 'student_test')
    assert len(train) == 80 and len(test) == 10
    rows = []
    x, y, groups = fitting_arrays(train, sorted(train))
    for name in MODELS:
        cv_scores = []
        for repeat in (1, 2):
            source = args.oof_dir / f'oof_{safe_name(name)}_{repeat}.npz'
            with np.load(source) as archive:
                cv_scores.append(counts_for(train, {t: archive[t] for t in train}))
        start = time.perf_counter()
        with threadpool_limits(limits=args.threads):
            model = fit_model(name, x, y, groups, threads=args.threads)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = {t: model_scores(model, d.x) for t, d in test.items()}
        inference_seconds = time.perf_counter() - start
        test_stats = counts_for(test, predictions, args.data)
        rows.append(dict(method=name, cv_macro_f1=float(np.mean([v['candidate_macro_f1'] for v in cv_scores])),
                         test_macro_f1=test_stats['candidate_macro_f1'],
                         test_top1=test_stats['top1'], fit_seconds=fit_seconds,
                         classifier_continuity=test_stats['classifier_continuity'],
                         inference_seconds=inference_seconds,
                         inference_ms_per_observation=1000*inference_seconds/test_stats['observations'],
                         f1_definition='macro F1 over candidate rows, one positive prediction per observation'))
        print(name, rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
