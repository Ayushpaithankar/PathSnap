"""Rescore the frozen legacy pipeline against the current ordered labels."""
import argparse
from pathlib import Path

import pandas as pd

from .core import load_reference, load_trip
from .grouped_training import read_split
from .pipeline import lcs_length, reconstruct, sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    test = read_split(args.prepared / 'student_test')
    scores = pd.read_csv('results/student_test_probabilities.csv.gz', dtype={'trip_id': str},
                         float_precision='round_trip')
    report = []
    for trip_id, frame in scores.groupby('trip_id'):
        data = test[trip_id]
        track, nodes, arcs = load_trip(args.data, 'student_test', trip_id)
        reference = load_reference(args.data, 'student_test', trip_id)
        rows = frame.to_dict('records')
        response = reconstruct(trip_id, track, nodes, arcs, rows)
        choices = set(zip(frame.observation_index, frame.arc_id))
        record = dict(trip_id=trip_id, observations=len(data.labels),
                      candidate_misses=sum((i, arc) not in choices for i, arc in enumerate(data.labels)))
        for name in ('baseline', 'decoded'):
            prediction = response[name]
            ids = prediction['arc_ids']
            ordered = sequence(ids, prediction['connecting_arc_ids'])
            record[name + '_correct'] = sum(a == b for a, b in zip(ids, data.labels))
            record[name + '_ordered_matches'] = lcs_length(ordered, reference)
            record[name + '_sequence_length_sum'] = len(ordered) + len(reference)
        report.append(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
