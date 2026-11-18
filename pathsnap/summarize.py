"""Generate comparison figures and a decoder fixture from measured results."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathsnap.candidates import load_trip, predict_trip
from pathsnap.core import load_trip as load_extracted
from pathsnap.pipeline import model_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('results', type=Path)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    out = args.results
    inventory = pd.read_csv(args.data / 'dataset_inventory.csv', dtype={'trip_id': str})
    inventory.to_csv(out / 'dataset_inventory.csv', index=False)
    totals = inventory.groupby('partition')[['observations', 'labeled', 'misses', 'invalid_points', 'gaps']].sum()
    totals['label_coverage'] = totals.labeled / totals.observations
    totals['candidate_miss_rate'] = totals.misses / totals.labeled
    totals.to_csv(out / 'data_quality_summary.csv')
    print(totals.to_string())

    results = pd.read_csv(out / 'model_comparison.csv').query("phase == 'baseline'")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
    for ax, field, title in zip(axes, ['cv_macro_f1', 'student_top1', 'student_log_loss'],
                               ['Group CV macro F1', 'Student test top-1 accuracy', 'Student test log loss']):
        ax.barh(results.model, results[field], color='#255e80')
        ax.invert_yaxis()
        ax.set_title(title)
        ax.bar_label(ax.containers[0], fmt='%.3f', padding=3)
        ax.set_xlim(0, max(results[field]) * 1.15)
    fig.savefig(out / 'model_comparison.png', dpi=160)
    plt.close(fig)

    model = joblib.load(out / 'best_classifier.joblib')
    trip_id = '00000052'
    if args.archive.is_dir():
        trip = model_trip(trip_id, *load_extracted(args.archive, 'student_test', trip_id))
    else:
        with ZipFile(args.archive) as archive:
            trip = load_trip(archive, 'student_test', trip_id)
    observations, candidates = predict_trip(trip, model)
    stored = pd.read_csv(out / 'student_test_probabilities.csv.gz', dtype={'trip_id': str},
                         float_precision='round_trip')
    stored = stored[stored.trip_id == trip_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(stored, candidates, check_dtype=False, atol=1e-10, rtol=1e-10)
    observations.to_csv(out / 'example_observations.csv', index=False)
    candidates.to_csv(out / 'example_candidates.csv', index=False)
    top = candidates.loc[candidates.groupby('observation_index').probability.idxmax()]
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    net = trip.nodes[trip.arcs]
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(net, color='#c5cdd1', linewidth=.45, label='Road network'))
    ax.scatter(observations.lon, observations.lat, s=8, color='#3169a6', alpha=.65, label='GPS observations')
    ax.scatter(top.snapped_lon, top.snapped_lat, s=9, color='#b64b31', alpha=.7, label='Top-scoring snaps')
    ax.set(xlabel='Longitude', ylabel='Latitude', title=f'Student test trip {trip_id}')
    ax.set_aspect(1 / np.cos(np.deg2rad(observations.lat.median())))
    ax.legend(loc='best')
    fig.savefig(out / 'example_trip.png', dpi=160)
    plt.close(fig)
    print(f'Saved {len(observations)} observations and {len(candidates)} candidates. Inference matches stored scores.')


if __name__ == '__main__':
    main()
