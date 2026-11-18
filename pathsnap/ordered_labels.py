"""Align GPS observations to ordered route occurrences, including revisited arcs."""
import numpy as np

from .candidates import SegmentIndex


def align(trip, reference, beam_size=24):
    """Beam Viterbi on monotone route positions; every observation gets a label."""
    reference = np.asarray(reference, dtype=int)
    if not len(reference) or reference.min() < 0 or reference.max() >= len(trip.arcs):
        raise ValueError('Ordered alignment requires a valid nonempty reference route')
    origin = np.median(trip.track[:, :2], axis=0)
    scale = 111195. * np.array([np.cos(np.deg2rad(origin[1])), 1.])
    points = (trip.track[:, :2] - origin) * scale
    nodes = (trip.nodes - origin) * scale
    ends = trip.arcs[reference]
    index = SegmentIndex(nodes[ends[:, 0]], nodes[ends[:, 1]])
    travel = np.r_[0., np.cumsum(index.length)]
    t = trip.track[:, 2]
    positions, progress, score = None, None, None
    states, parents, residuals = [], [], []
    for i, (nearest, _, _, _) in enumerate(index.project(points, 120., beam_size)):
        # Including prior states guarantees a feasible monotone path even for outliers.
        candidates = nearest if positions is None else np.union1d(
            nearest, np.minimum(len(reference) - 1, positions[:, None] + np.arange(3)).ravel())
        delta = index.delta[candidates]
        offset = points[i] - index.starts[candidates]
        fraction = np.clip((offset * delta).sum(axis=1) /
                           np.maximum(index.length[candidates] ** 2, 1e-12), 0., 1.)
        snap_distance = np.linalg.norm(offset - fraction[:, None] * delta, axis=1)
        next_progress = travel[candidates] + index.length[candidates] * fraction
        cost = snap_distance / 12.
        parent = np.full(len(candidates), -1, dtype=int)
        residual = np.zeros(len(candidates))
        if positions is not None:
            displacement = np.linalg.norm(points[i] - points[i-1])
            elapsed = max(1., t[i] - t[i-1])
            signed = next_progress[None, :] - progress[:, None]
            distances = np.maximum(0., signed)
            error = np.abs(distances - displacement)
            step = error / (12. + np.sqrt(elapsed)) + np.maximum(-signed - 8., 0.) / 8.
            step += np.maximum(distances - (50. * elapsed + 40.), 0.) / 30.
            step[candidates[None, :] < positions[:, None]] = np.inf
            total = score[:, None] + step
            parent = np.argmin(total, axis=0)
            columns = np.arange(len(candidates))
            residual = error[parent, columns]
            cost += total[parent, columns]
        keep = np.argsort(cost, kind='stable')[:beam_size]
        keep = keep[np.isfinite(cost[keep])]
        positions, progress, score = candidates[keep], next_progress[keep], cost[keep]
        score -= score.min()
        states.append(positions)
        parents.append(parent[keep])
        residuals.append(residual[keep])
    path = np.empty(len(points), dtype=int)
    error = np.empty(len(points))
    j = 0
    for i in range(len(points) - 1, -1, -1):
        path[i] = states[i][j]
        error[i] = residuals[i][j]
        j = parents[i][j]
    return reference[path], path, error
