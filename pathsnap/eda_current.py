"""Reproducible raw-data quality inventory and development-trip trajectory plot."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from .core import list_trip_ids, load_trip


def components(nodes, arcs):
    parent = list(range(len(nodes)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in arcs:
        a, b = find(u), find(v)
        parent[a] = b
    return len({find(i) for i in range(len(nodes))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for partition in ('train', 'student_test'):
        for trip_id in list_trip_ids(args.data, partition):
            track, nodes, arcs = load_trip(args.data, partition, trip_id)
            xy = np.array([(r['lon'], r['lat']) for r in track])
            time = np.array([r['timestamp'] for r in track])
            dt = np.diff(time)
            degree = np.bincount(np.asarray(arcs).ravel(), minlength=len(nodes))
            rows.append(dict(partition=partition, trip_id=trip_id, observations=len(track),
                invalid_gps=int((~np.isfinite(xy).all(axis=1) |
                    (np.abs(xy[:, 0]) > 180) | (np.abs(xy[:, 1]) > 90)).sum()),
                invalid_timestamps=int((~np.isfinite(time)).sum()),
                nonpositive_steps=int((dt <= 0).sum()), gaps_over_30s=int((dt > 30).sum()),
                nodes=len(nodes), directed_arcs=len(arcs),
                weak_components=components(nodes, arcs), isolated_nodes=int((degree == 0).sum()),
                longitude_span_deg=float(np.ptp(xy[:, 0])), latitude_span_deg=float(np.ptp(xy[:, 1]))))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / 'eda_inventory.csv', index=False)
    summary = frame.groupby('partition')[['observations', 'invalid_gps', 'invalid_timestamps',
        'nonpositive_steps', 'gaps_over_30s', 'nodes', 'directed_arcs', 'weak_components',
        'isolated_nodes']].agg(['sum', 'median'])
    summary.to_csv(args.out / 'eda_summary.csv')
    print(summary.to_string())

    track, nodes, arcs = load_trip(args.data, 'student_test', '00000052')
    points = np.array([(r['lon'], r['lat']) for r in track])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.add_collection(LineCollection([[nodes[u], nodes[v]] for u,v in arcs],
                  colors='#a7b0bc', linewidths=.35, label='Directed road network'))
    ax.plot(points[:, 0], points[:, 1], color='#2457d6', linewidth=.7, alpha=.7,
            label='GPS trajectory')
    ax.scatter(points[:, 0], points[:, 1], c=np.arange(len(points)), cmap='viridis',
               s=6, zorder=3, label='GPS observations (time order)')
    ax.set(xlabel='Longitude', ylabel='Latitude', title='Student test trip 00000052: GPS and road network')
    ax.ticklabel_format(useOffset=False)
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.median(points[:, 1]))))
    fig.tight_layout()
    fig.savefig(args.out / 'eda_trajectory_00000052.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
