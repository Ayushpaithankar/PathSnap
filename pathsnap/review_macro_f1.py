"""Derive exact candidate-row macro F1 from saved point-correct counts and candidate coverage."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def macro_f1(tp, obs, positive_rows, total_rows):
    fp = obs - tp
    fn = positive_rows - tp
    tn = total_rows - tp - fp - fn
    if min(tp, fp, fn, tn) < 0:
        raise ValueError('Invalid confusion counts')
    return (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fp+fn))/2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prepared', type=Path)
    parser.add_argument('review', type=Path)
    args = parser.parse_args()
    output = []
    for split, partition, filename, trusted in [
        ('cv_trusted', 'train', 'cv_per_trip.csv', True),
        ('student_test', 'student_test', 'test_per_trip.csv', False)]:
        frame = pd.read_csv(args.review / filename, dtype={'trip_id': str})
        trip_stats = {}
        for trip_id in frame.trip_id.unique():
            with np.load(args.prepared / partition / f'{trip_id}.npz') as blob:
                obs, arc, labels = blob['obs'], blob['arc'], blob['labels']
                selected = blob['alignment_distance'] <= 75. if trusted else np.ones(len(labels), dtype=bool)
                trip_stats[trip_id] = (int(selected.sum()), int(selected[obs].sum()),
                    int(((arc == labels[obs]) & selected[obs]).sum()))
        for method, rows in frame.groupby('method'):
            obs = sum(trip_stats[t][0] for t in rows.trip_id)
            total = sum(trip_stats[t][1] for t in rows.trip_id)
            positive = sum(trip_stats[t][2] for t in rows.trip_id)
            tp = int(rows['trusted_point_correct' if trusted else 'point_correct'].sum())
            output.append(dict(split=split, method=method, macro_f1=macro_f1(tp, obs, positive, total),
                               observations=obs, candidate_rows=total, correct=tp,
                               definition='candidate-row binary macro F1, one positive predicted candidate per observation'))
    pd.DataFrame(output).to_csv(args.review / 'review_macro_f1.csv', index=False)
    print(pd.DataFrame(output).to_string(index=False))


if __name__ == '__main__':
    main()
