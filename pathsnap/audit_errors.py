"""Diagnose the original model and decoder before changing their targets."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import prepare, reference_labels
from .core import RoadGraph, baseline, decode, load_reference, load_trip, validate_candidates
from .pipeline import model_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('inputs/Student folder.zip'))
    parser.add_argument('--out', type=Path, default=Path('results/legacy_error_audit'))
    args = parser.parse_args()
    trip_id = '00000077'
    track, nodes, arcs = load_trip(args.data, 'student_test', trip_id)
    trip = model_trip(trip_id, track, nodes, arcs)
    labels = reference_labels(trip, load_reference(args.data, 'student_test', trip_id))
    _, features = prepare(trip)
    scores = pd.read_csv('results/student_test_probabilities.csv.gz', dtype={'trip_id': str},
                         float_precision='round_trip')
    rows = scores[scores.trip_id == trip_id].to_dict('records')
    graph = RoadGraph(nodes, arcs)
    groups = validate_candidates(track, arcs, rows)
    raw = baseline(track, groups)
    chosen, _, _ = decode(track, graph, groups)
    vectors = trip.nodes[trip.arcs[:, 1]] - trip.nodes[trip.arcs[:, 0]]
    vectors[:, 0] *= np.cos(np.deg2rad(np.median(trip.track[:, 1])))
    lengths = np.linalg.norm(vectors, axis=1)
    degrees = np.bincount(trip.arcs.ravel(), minlength=len(nodes))
    motion = features.drop_duplicates('observation_index').set_index('observation_index')
    records, summary = [], {'trip_id': trip_id, 'observations': len(track),
                             'labeled': int((labels >= 0).sum()), 'methods': {}}
    for name, guesses in [('classifier', raw), ('decoder', chosen)]:
        buckets, tags = Counter(), Counter()
        for i, (label, guess) in enumerate(zip(labels, guesses)):
            if label < 0 or label == guess:
                continue
            flags = []
            a = arcs[label]
            if not any(r['arc_id'] == label for r in groups.get(i, [])):
                flags.append('candidate_miss')
            if 0. < motion.at[i, 'elapsed_s'] <= 10. and motion.at[i, 'speed_mps'] <= 1.:
                flags.append('stationary_or_slow')
            if guess is not None:
                b = arcs[guess]
                if a == b[::-1]:
                    flags.append('reverse_arc')
                elif (not set(a) & set(b) and lengths[label] * lengths[guess] > 0 and
                      abs(vectors[label] @ vectors[guess]) / (lengths[label] * lengths[guess]) >= .94):
                    flags.append('parallel_road_proxy')
                if set(a) & set(b) and any(degrees[n] >= 5 for n in set(a) & set(b)):
                    flags.append('intersection_proxy')
            if (i > 0 and labels[i-1] < 0) or (i+1 < len(labels) and labels[i+1] < 0):
                flags.append('unlabelled_adjacent')
            bucket = flags[0] if flags else 'other'
            buckets[bucket] += 1
            tags.update(flags)
            records.append(dict(method=name, observation_index=i, reference_arc_id=label,
                                predicted_arc_id=guess, bucket=bucket, tags='|'.join(flags)))
        summary['methods'][name] = dict(errors=sum(buckets.values()), exclusive=dict(buckets),
                                        overlapping=dict(tags))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out / 'observations.csv', index=False)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
