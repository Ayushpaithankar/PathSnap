"""Build development tables directly from the supplied ZIP."""

import argparse
from pathlib import Path
from zipfile import ZipFile
from contextlib import nullcontext

import numpy as np
import pandas as pd

from pathsnap.candidates import load_trip, prepare, trip_ids
from pathsnap.core import load_trip as load_extracted
from pathsnap.pipeline import model_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inventory = []
    extracted = args.archive.is_dir()
    with (nullcontext() if extracted else ZipFile(args.archive)) as archive:
        for partition in ('train', 'student_test'):
            ids = sorted(p.name for p in (args.archive / partition).glob('*/') if
                         (p / (p.name + '.track')).is_file()) if extracted else trip_ids(archive, partition)
            for trip_id in ids:
                if extracted:
                    trip = model_trip(trip_id, *load_extracted(args.archive, partition, trip_id))
                    route = np.loadtxt(args.archive / partition / trip_id / (trip_id + '.route'),
                                       dtype=int, ndmin=1)
                else:
                    trip = load_trip(archive, partition, trip_id)
                    with archive.open(f'Student folder/{partition}/{trip_id}/{trip_id}.route') as stream:
                        route = np.loadtxt(stream, dtype=int, ndmin=1)
                obs, rows = prepare(trip, route)
                target_ids = rows.loc[rows.target == 1, 'observation_index']
                eligible = obs.reference_arc_id >= 0
                inventory.append(dict(partition=partition, trip_id=trip_id, observations=len(obs),
                    nodes=len(trip.nodes), arcs=len(trip.arcs), reference_arcs=len(route),
                    longitude_min=obs.lon.min(), longitude_max=obs.lon.max(),
                    latitude_min=obs.lat.min(), latitude_max=obs.lat.max(),
                    invalid_points=int((~np.isfinite(trip.track).all(axis=1)).sum()),
                    gaps=int((obs.timestamp.diff() > 1).sum()), max_gap_s=obs.timestamp.diff().max(),
                    labeled=int(eligible.sum()), misses=int((eligible & ~obs.observation_index.isin(target_ids)).sum())))
                folder = args.out / partition
                folder.mkdir(exist_ok=True)
                obs.to_pickle(folder / f'{trip_id}-observations.pkl')
                rows.to_pickle(folder / f'{trip_id}-candidates.pkl')
                print(partition, trip_id, len(obs), len(rows), flush=True)
    pd.DataFrame(inventory).to_csv(args.out / 'dataset_inventory.csv', index=False)


if __name__ == '__main__':
    main()
