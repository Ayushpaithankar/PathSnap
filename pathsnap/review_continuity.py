"""Measure live classifier and Viterbi direct continuity on student-test trips."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .candidates import predict_trip
from .core import list_trip_ids, load_trip
from .pipeline import model_trip, reconstruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    root = Path('results/review')
    methods = {'LightGBM untuned': 'deployed.joblib',
               'LightGBM tuned': 'tuned_ranker.joblib', 'Nearest road': 'nearest_rule.joblib'}
    rows = []
    for name, filename in methods.items():
        artifact = joblib.load(root / filename)
        before = after = comparable = 0
        for trip_id in list_trip_ids(args.data, 'student_test'):
            track, nodes, arcs = load_trip(args.data, 'student_test', trip_id)
            trip = model_trip(trip_id, track, nodes, arcs)
            with threadpool_limits(limits=1):
                _, table = predict_trip(trip, artifact)
            result = reconstruct(trip_id, track, nodes, arcs, table.to_dict('records'), artifact['decoder'])
            a, b = result['baseline']['continuity'], result['decoded']['continuity']
            assert a['comparable'] == b['comparable']
            before += a['connected']; after += b['connected']; comparable += a['comparable']
        rows.append(dict(method=name, pairs=comparable, classifier_continuity=before/comparable,
                         decoded_continuity=after/comparable))
        print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
