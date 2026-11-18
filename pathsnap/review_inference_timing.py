"""Measure identical live candidate generation/scoring for review artifacts."""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .candidates import predict_trip
from .core import list_trip_ids, load_trip
from .pipeline import model_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    root = Path('results/review')
    artifacts = {name: joblib.load(root / filename) for name, filename in [
        ('LightGBM untuned', 'deployed.joblib'), ('LightGBM tuned', 'tuned_ranker.joblib'),
        ('Nearest road', 'nearest_rule.joblib')]}
    trips = {t: model_trip(t, *load_trip(args.data, 'student_test', t))
             for t in list_trip_ids(args.data, 'student_test')}
    rows = []
    with threadpool_limits(limits=1):
        for name, model in artifacts.items():
            durations = []
            for trip in trips.values():
                start = time.perf_counter()
                predict_trip(trip, model)
                durations.append(time.perf_counter() - start)
            rows.append(dict(method=name, trips=len(durations), observations=sum(len(t.track) for t in trips.values()),
                             total_seconds=sum(durations), median_ms_per_trip=float(1000*np.median(durations)),
                             ms_per_observation=float(1000*sum(durations)/sum(len(t.track) for t in trips.values())),
                             scope='feature generation and candidate scoring, excludes Viterbi'))
            print(rows[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
