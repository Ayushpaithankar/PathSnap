"""Audit the actual release default against ordered labels, including the rule if selected."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .candidates import predict_trip
from .core import decode, validate_candidates
from .grouped_training import read_split
from .pipeline import model_trip
from .review_experiment import audit_predictions, bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    model = joblib.load(args.out / 'deployed.joblib')
    cases = []
    for t, d in read_split(args.prepared / 'student_test').items():
        loaded = bundle(args.data, 'student_test', t)
        track, graph, _ = loaded
        trip = model_trip(t, track, graph.nodes, graph.arcs)
        _, table = predict_trip(trip, model)
        assert table.arc_id.tolist() == d.arc.tolist()
        groups = validate_candidates(track, graph.arcs, table.to_dict('records'))
        ids, _, _ = decode(track, graph, groups, **model['decoder'])
        cases += audit_predictions(d, loaded, table.probability.to_numpy(), ids)
    pd.DataFrame(cases).rename(columns={'ranker_arc_id': 'scoring_arc_id',
        'ranker_was_correct': 'scoring_was_correct'}).to_csv(args.out / 'deployed_error_audit.csv', index=False)


if __name__ == '__main__':
    main()
