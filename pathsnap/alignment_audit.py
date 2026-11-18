"""Inspect ordered-reference consistency and candidate coverage by trip."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .core import load_trip
from .grouped_training import read_split
from .pipeline import model_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    records = []
    for partition in ('train', 'student_test'):
        for trip_id, d in read_split(args.prepared / partition).items():
            trip = model_trip(trip_id, *load_trip(args.data, partition, trip_id))
            scale = 111195. * np.array([np.cos(np.deg2rad(np.median(trip.track[:, 1]))), 1.])
            endpoints = trip.nodes[trip.arcs[d.labels]]
            direction = (endpoints[:, 1] - endpoints[:, 0]) * scale
            offset = (trip.track[:, :2] - endpoints[:, 0]) * scale
            projection = np.clip((offset * direction).sum(axis=1) /
                                 np.maximum((direction ** 2).sum(axis=1), 1e-12), 0., 1.)
            distance = np.linalg.norm(offset - projection[:, None] * direction, axis=1)
            records.append(dict(partition=partition, trip_id=trip_id, observations=len(d.labels),
                                candidate_misses=len(d.labels)-int(d.target.sum()),
                                aligned_distance_median_m=np.median(distance),
                                aligned_distance_p95_m=np.quantile(distance, .95),
                                aligned_distance_above_100m=int((distance > 100.).sum())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
