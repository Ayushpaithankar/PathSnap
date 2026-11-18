"""Load PathSnap trips and score directed arc candidates."""

from dataclasses import dataclass
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FEATURES = ['distance_m', 'arc_length_m', 'arc_fraction', 'heading_cos',
            'speed_mps', 'elapsed_s', 'distance_rank', 'candidate_count']
KEYS = ['trip_id', 'observation_index', 'arc_id']
HANDOFF = ['trip_id', 'observation_index', 'timestamp', 'arc_id', 'snapped_lon',
           'snapped_lat', 'distance_m', 'probability']


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray


def trip_ids(archive, partition):
    return sorted({p.split('/')[-2] for p in archive.namelist()
                   if p.endswith('.track') and p.split('/')[-3] == partition})


def load_trip(archive: ZipFile, partition, trip_id):
    prefix = f'Student folder/{partition}/{trip_id}/{trip_id}'
    def read(ext, dtype):
        with archive.open(f'{prefix}.{ext}') as stream:
            return np.loadtxt(stream, dtype=dtype, ndmin=2)

    return Trip(trip_id, read('track', float), read('nodes', float), read('arcs', int))


class SegmentIndex:
    """Sample segments for retrieval, then project onto the actual segments."""

    def __init__(self, starts, ends, step=35.):
        self.starts = starts
        self.delta = ends - starts
        self.length = np.linalg.norm(self.delta, axis=1)
        self.divisions = np.maximum(1, np.ceil(self.length / step).astype(int))
        self.ids = np.repeat(np.arange(len(starts)), self.divisions)
        fraction = (np.arange(len(self.ids)) - np.repeat(
            np.cumsum(self.divisions) - self.divisions, self.divisions) + .5) / self.divisions[self.ids]
        self.tree = cKDTree(starts[self.ids] + fraction[:, None] * self.delta[self.ids])
        self.pad = step / 2

    def project(self, points, radius, limit):
        nearest = self.tree.query(points)[0]
        radii = np.maximum(radius, nearest) + self.pad + 1e-6
        hits = self.tree.query_ball_point(points, radii)
        for point, hit in zip(points, hits):
            ids = np.unique(self.ids[hit])
            delta = self.delta[ids]
            fraction = np.clip(np.einsum('ij,ij->i', point - self.starts[ids], delta)
                               / np.maximum(self.length[ids] ** 2, 1e-12), 0., 1.)
            snapped = self.starts[ids] + fraction[:, None] * delta
            distances = np.linalg.norm(snapped - point, axis=1)
            order = np.argsort(distances, kind='stable')
            inside = order[distances[order] <= radius]
            keep = inside[:limit] if len(inside) else order[:1]
            yield ids[keep], fraction[keep], snapped[keep], distances[keep]


def prepare(trip, reference=None, radius=75., limit=5):
    """Return observations, candidates and optional development reference labels."""
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    origin = np.median(track[:, :2], axis=0)
    scale = 111195. * np.array([np.cos(np.deg2rad(origin[1])), 1.])
    points = (track[:, :2] - origin) * scale
    xy = (nodes - origin) * scale
    starts, ends = xy[arcs[:, 0]], xy[arcs[:, 1]]
    graph = SegmentIndex(starts, ends)
    dt = np.diff(track[:, 2], prepend=track[0, 2])
    movement = np.diff(points, axis=0, prepend=points[:1])
    displacement = np.linalg.norm(movement, axis=1)
    directional = (displacement >= 4.) & (dt > 0.) & (dt <= 10.)
    unit = movement / np.maximum(displacement[:, None], 1e-12)
    speed = np.divide(displacement, dt, out=np.zeros(len(track)), where=(dt > 0) & (dt <= 10))
    observations = pd.DataFrame(dict(trip_id=trip.trip_id, observation_index=np.arange(len(track)),
                                     lon=track[:, 0], lat=track[:, 1], timestamp=track[:, 2]))
    rows = []
    for i, (ids, fractions, snapped, distances) in enumerate(graph.project(points, radius, limit)):
        heading = graph.delta[ids] @ unit[i] / np.maximum(graph.length[ids], 1e-12) if directional[i] else np.zeros(len(ids))
        geographic = snapped / scale + origin
        for rank, arc in enumerate(ids):
            rows.append((i, track[i, 2], arc, *geographic[rank], distances[rank], graph.length[arc],
                         fractions[rank], heading[rank], speed[i], dt[i], rank, len(ids)))
    candidates = pd.DataFrame(rows, columns=['observation_index', 'timestamp', 'arc_id',
        'snapped_lon', 'snapped_lat', *FEATURES])
    candidates.insert(0, 'trip_id', trip.trip_id)
    if reference is not None:
        labels = reference_labels(trip, reference)
        observations['reference_arc_id'] = labels
        candidates['reference_arc_id'] = labels[candidates.observation_index]
        candidates['target'] = (candidates.arc_id == candidates.reference_arc_id).astype(int)
    return observations, candidates


def reference_labels(trip, reference):
    """Member 1's geometric proxy labels, calculated separately from inference."""
    track = trip.track
    labels = np.full(len(track), -1, dtype=int)
    ref_ids = np.unique(np.asarray(reference, dtype=int))
    if not len(ref_ids):
        return labels
    if ref_ids.min() < 0 or ref_ids.max() >= len(trip.arcs):
        raise ValueError('Reference arc ID out of range')
    origin = np.median(track[:, :2], axis=0)
    scale = 111195. * np.array([np.cos(np.deg2rad(origin[1])), 1.])
    points = (track[:, :2] - origin) * scale
    xy = (trip.nodes - origin) * scale
    ref_arcs = trip.arcs[ref_ids]
    graph = SegmentIndex(xy[ref_arcs[:, 0]], xy[ref_arcs[:, 1]])
    dt = np.diff(track[:, 2], prepend=track[0, 2])
    movement = np.diff(points, axis=0, prepend=points[:1])
    displacement = np.linalg.norm(movement, axis=1)
    directional = (displacement >= 4.) & (dt > 0.) & (dt <= 10.)
    unit = movement / np.maximum(displacement[:, None], 1e-12)
    for i, (ids, _, _, distance) in enumerate(graph.project(points, 75., 12)):
        direction = graph.delta[ids] @ unit[i] / np.maximum(graph.length[ids], 1e-12)
        costs = distance + (8. * (1 - direction) if directional[i] else 0.)
        order = np.argsort(costs, kind='stable')
        best = order[0]
        if distance[best] <= 45. and (len(order) == 1 or costs[order[1]] - costs[best] >= 1.):
            labels[i] = ref_ids[ids[best]]
    return labels


def predict_trip(trip, artifact):
    if artifact.get('version') == 'ordered-v2':
        from .grouped_features import RULES, baseline_scores, make_candidates
        from .grouped_models import model_scores, softmax
        observations, candidates = make_candidates(trip, **artifact['candidate_settings'])
        x = candidates[artifact['features']].to_numpy(dtype=np.float32)
        nearest, heading = (baseline_scores(rule, x) for rule in RULES)
        if artifact['model'] is None:
            scores = nearest if artifact['name'] == RULES[0] else heading
        else:
            scores = model_scores(artifact['model'], x)
        counts = np.bincount(candidates.observation_index, minlength=len(observations))
        candidates['probability'] = softmax(scores * artifact['calibration_slope'], counts)
        candidates['nearest_score'] = nearest
        candidates['heading_score'] = heading
        return observations, candidates[HANDOFF + ['nearest_score', 'heading_score']]
    observations, candidates = prepare(trip, **artifact['candidate_settings'])
    candidates['probability'] = artifact['model'].predict_proba(candidates[artifact['features']])[:, 1]
    return observations, candidates[HANDOFF]
