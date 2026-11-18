"""Save all train and student-test observations with ordered route labels."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .core import list_trip_ids, load_reference, load_trip
from .grouped_features import CONTEXT_FEATURES, GROUP_FEATURES, make_candidates
from .ordered_labels import align
from .pipeline import model_trip


SETTINGS = {'radius': 100., 'limit': 8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--context', action='store_true')
    args = parser.parse_args()
    for partition in ('train', 'student_test'):
        folder = args.out / partition
        folder.mkdir(parents=True, exist_ok=True)
        for trip_id in list_trip_ids(args.data, partition):
            start = perf_counter()
            trip = model_trip(trip_id, *load_trip(args.data, partition, trip_id))
            reference = load_reference(args.data, partition, trip_id)
            labels, route_positions, residuals = align(trip, reference)
            _, frame = make_candidates(trip, context=args.context, **SETTINGS)
            obs = frame.observation_index.to_numpy(dtype=np.int32)
            arc = frame.arc_id.to_numpy(dtype=np.int32)
            projection_scale = 111195. * np.array([np.cos(np.deg2rad(np.median(trip.track[:, 1]))), 1.])
            ends = trip.nodes[trip.arcs[labels]]
            position = (trip.track[:, :2] - ends[:, 0]) * projection_scale
            direction = (ends[:, 1] - ends[:, 0]) * projection_scale
            fraction = np.clip((position*direction).sum(axis=1) /
                               np.maximum((direction**2).sum(axis=1), 1e-12), 0., 1.)
            distance = np.linalg.norm(position - fraction[:, None] * direction, axis=1)
            features = CONTEXT_FEATURES if args.context else GROUP_FEATURES
            np.savez(folder / (trip_id + '.npz'), features=frame[features].to_numpy(dtype=np.float32),
                     obs=obs, arc=arc, labels=labels, route_positions=route_positions, residuals=residuals,
                     snaps=frame[['snapped_lon', 'snapped_lat']].to_numpy(), timestamps=trip.track[:, 2],
                     alignment_distance=distance)
            print(partition, trip_id, len(labels), len(arc), f'{perf_counter()-start:.2f}s', flush=True)


if __name__ == '__main__':
    main()
