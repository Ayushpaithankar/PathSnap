"""Features available from GPS and the road graph before reference alignment."""
import numpy as np

from .candidates import FEATURES, prepare


EXTRA = ['smooth_heading_cos', 'stationary', 'distance_above_min', 'heading_rank',
         'reverse_twin', 'local_arc_support', 'trip_arc_support', 'from_degree',
         'to_degree', 'connects_previous', 'signed_offset_m']
GROUP_FEATURES = FEATURES + EXTRA
CONTEXT_FEATURES = GROUP_FEATURES + [
    'heading_15s', 'heading_30s', 'motion_15s_m', 'motion_30s_m',
    'incoming_support', 'outgoing_support', 'heading_incoming_support',
    'heading_outgoing_support', 'same_arc_adjacent',
]
RULES = ('Nearest road', 'Distance + heading')


def baseline_scores(name, x):
    distance = x[:, GROUP_FEATURES.index('distance_m')]
    if name == RULES[0]:
        return -distance / 12.
    cos = x[:, GROUP_FEATURES.index('heading_cos')]
    speed = x[:, GROUP_FEATURES.index('speed_mps')]
    elapsed = x[:, GROUP_FEATURES.index('elapsed_s')]
    return -(distance + np.where(speed * elapsed >= 4., 8. * (1. - cos), 0.)) / 12.


def make_candidates(trip, radius=100., limit=8, context=False):
    observations, frame = prepare(trip, radius=radius, limit=limit)
    idx = frame.observation_index.to_numpy()
    arc = frame.arc_id.to_numpy()
    u, v = trip.arcs[arc].T
    scale = 111195. * np.array([np.cos(np.deg2rad(np.median(trip.track[:, 1]))), 1.])
    points = trip.track[:, :2] * scale
    t = trip.track[:, 2]
    center = np.arange(len(points))
    lo, hi = np.maximum(0, center - 2), np.minimum(len(points) - 1, center + 2)
    vector = points[hi] - points[lo]
    motion = np.linalg.norm(vector, axis=1)
    duration = t[hi] - t[lo]
    stationary = (duration > 0) & (motion <= duration)
    informative = (motion >= 4.) & (duration > 0) & (duration <= 15.) & ~stationary
    directions = (trip.nodes[v] - trip.nodes[u]) * scale
    lengths = np.maximum(np.linalg.norm(directions, axis=1), 1e-9)
    unit = directions / lengths[:, None]
    heading = (unit * vector[idx]).sum(axis=1) / np.maximum(motion[idx], 1e-9)
    frame['smooth_heading_cos'] = np.where(informative[idx], heading, 0.)
    frame['stationary'] = stationary[idx].astype(int)
    groups = frame.groupby('observation_index', sort=False)
    frame['distance_above_min'] = frame.distance_m - groups.distance_m.transform('min')
    frame['heading_rank'] = groups.smooth_heading_cos.rank(ascending=False, method='min') - 1
    triples = set(zip(idx, u, v))
    frame['reverse_twin'] = [a != b and (i, b, a) in triples for i, a, b in zip(idx, u, v)]
    previous_heads = set(zip(idx, v))
    previous_arcs = set(zip(idx, arc))
    frame['connects_previous'] = [(i-1, a) in previous_heads or (i-1, edge) in previous_arcs
                                  for i, a, edge in zip(idx, u, arc)]
    degree = np.bincount(trip.arcs.ravel(), minlength=len(trip.nodes))
    frame['from_degree'], frame['to_degree'] = degree[u], degree[v]
    offset = points[idx] - trip.nodes[u] * scale
    frame['signed_offset_m'] = unit[:, 0] * offset[:, 1] - unit[:, 1] * offset[:, 0]
    local = np.empty(len(frame), dtype=float)
    support = np.empty(len(frame), dtype=float)
    for _, indices in frame.groupby('arc_id', sort=False).indices.items():
        observed = idx[indices]
        first = np.searchsorted(observed, observed - 8)
        last = np.searchsorted(observed, observed + 8, side='right')
        local[indices] = (last - first - 1) / 16.
        support[indices] = np.log1p(len(indices) - 1)
    frame['local_arc_support'], frame['trip_arc_support'] = local, support
    if context:
        for window in (15, 30):
            # Work within strictly increasing clock runs. Sparse gaps cannot invent motion.
            direction = np.zeros_like(points)
            boundaries = np.r_[0, np.flatnonzero((np.diff(t) <= 0) | (np.diff(t) > 10)) + 1, len(t)]
            for start, stop in zip(boundaries, boundaries[1:]):
                time = t[start:stop]
                before = np.minimum(np.searchsorted(time, time - window / 2), len(time) - 1)
                after = np.maximum(0, np.searchsorted(time, time + window / 2, side='right') - 1)
                direction[start:stop] = points[start + after] - points[start + before]
            distance = np.linalg.norm(direction, axis=1)
            frame[f'heading_{window}s'] = np.where(distance[idx] >= 4.,
                (direction[idx] * unit).sum(axis=1) / np.maximum(distance[idx], 1e-9), 0.)
            frame[f'motion_{window}s_m'] = distance[idx]
        starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1, len(idx)]
        # Nearby candidate direction and proximity are evidence for entering/leaving a directed arc.
        weights = np.exp(-frame.distance_above_min.to_numpy() / 12.)
        heading_weight = weights * (frame.smooth_heading_cos.to_numpy() + 1.) / 2.
        incoming = np.zeros(len(frame))
        outgoing = np.zeros(len(frame))
        directed_in = np.zeros(len(frame))
        directed_out = np.zeros(len(frame))
        same = np.zeros(len(frame))
        for i in range(len(t) - 1):
            if not 0 < t[i+1] - t[i] <= 10:
                continue
            a = slice(starts[i], starts[i+1])
            b = slice(starts[i+1], starts[i+2])
            repeated = arc[a, None] == arc[None, b]
            connects = (v[a, None] == u[None, b]) | repeated
            incoming[b], outgoing[a] = connects.T @ weights[a], connects @ weights[b]
            directed_in[b], directed_out[a] = connects.T @ heading_weight[a], connects @ heading_weight[b]
            same[b] += repeated.any(axis=0)
            same[a] += repeated.any(axis=1)
        frame['incoming_support'], frame['outgoing_support'] = incoming, outgoing
        frame['heading_incoming_support'], frame['heading_outgoing_support'] = directed_in, directed_out
        frame['same_arc_adjacent'] = same
    return observations, frame
