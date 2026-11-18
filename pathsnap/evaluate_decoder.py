"""Evaluate both methods using live inference or frozen scores."""
import argparse
import json
from pathlib import Path

import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

from .candidates import predict_trip
from .core import list_trip_ids, load_reference, load_trip
from .pipeline import metrics, model_trip, reconstruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--scores', type=Path, default=Path('results/student_test_probabilities.csv.gz'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--model', type=Path, default=Path('results/best_classifier.joblib'))
    parser.add_argument('--labels', type=Path, help='Frozen reference labels for verification')
    args = parser.parse_args()
    ids = list_trip_ids(args.data_root, 'student_test')
    if len(ids) != 10:
        parser.error('Expected ten student test trips')
    artifact = joblib.load(args.model) if args.live else None
    if not args.live:
        scores = pd.read_csv(args.scores, dtype={'trip_id': str}, float_precision='round_trip')
        if set(ids) != set(scores.trip_id):
            parser.error('Expected matching dataset and score exports')
        tables = dict(tuple(scores.groupby('trip_id')))
    labels = pd.read_csv(args.labels, dtype={'trip_id': str}) if args.labels else None
    if labels is not None and set(ids) != set(labels.trip_id):
        parser.error('Expected reference labels for every student test trip')
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for trip_id in ids:
        track, nodes, arcs = load_trip(args.data_root, 'student_test', trip_id)
        trip = model_trip(trip_id, track, nodes, arcs)
        if args.live:
            with threadpool_limits(limits=1):
                _, table = predict_trip(trip, artifact)
        else:
            table = tables[trip_id]
        rows = table.to_dict('records')
        result = reconstruct(trip_id, track, nodes, arcs, rows)
        reference = load_reference(args.data_root, 'student_test', trip_id)
        if not reference:
            raise ValueError(f'Missing reference for {trip_id}')
        expected = labels[labels.trip_id == trip_id] if labels is not None else None
        measurements = metrics(result, trip, rows, reference, expected)
        row = {k: v for k, v in measurements.items() if k not in ('baseline', 'decoded')}
        row.update(trip_id=trip_id, breaks=len(result['warnings']),
                   unmatched=result['decoded']['arc_ids'].count(None))
        for name in ('baseline', 'decoded'):
            row.update({f'{name}_{k}': v for k, v in measurements[name].items()})
        records.append(row)
        print(trip_id, row['baseline_point_accuracy'], row['decoded_point_accuracy'], flush=True)
    frame = pd.DataFrame(records)
    suffix = '_live' if args.live else ''
    frame.to_csv(args.out / f'decoder_per_trip{suffix}.csv', index=False)
    totals = frame.sum(numeric_only=True)
    summary = {'trips': len(frame), 'observations': int(totals.observations),
               'labeled': int(totals.labeled), 'label_coverage': totals.labeled / totals.observations,
               'candidate_miss_rate': totals.candidate_misses / totals.labeled}
    if labels is not None:
        summary['candidate_coverage_differences'] = int(totals.candidate_coverage_differences)
    for name in ('baseline', 'decoded'):
        summary[name] = {
            'point_accuracy': totals[f'{name}_correct'] / totals.labeled,
            'direct_continuity': totals[f'{name}_connected'] / totals[f'{name}_comparable'],
            'sequence_f1': 2 * totals[f'{name}_sequence_matches'] / totals[f'{name}_sequence_denominator'],
        }
    report = json.dumps(summary, indent=2)
    (args.out / f'decoder_summary{suffix}.json').write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
