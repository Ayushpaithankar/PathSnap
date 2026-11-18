"""Shared prediction and scoring for batch evaluation and the live API."""
from bisect import bisect_left
from collections import defaultdict
from itertools import groupby

import numpy as np

from .candidates import Trip, reference_labels
from .core import RoadGraph, baseline, continuity, decode, lines, validate_candidates


def model_trip(trip_id, track, nodes, arcs):
    return Trip(trip_id, np.asarray([(o['lon'], o['lat'], o['timestamp']) for o in track]),
                np.asarray(nodes), np.asarray(arcs))


def reconstruct(trip_id, track, nodes, arcs, rows, decoder=None):
    if any(row['trip_id'] != trip_id for row in rows):
        raise ValueError('Candidate trip ID mismatch')
    graph = RoadGraph(nodes, arcs)
    groups = validate_candidates(track, arcs, rows)
    raw = baseline(track, groups)
    decoded, connectors, warnings = decode(track, graph, groups, **(decoder or {}))
    empty = [[] for _ in track]
    network_ids = sorted({row['arc_id'] for row in rows} | {a for link in connectors for a in link})
    return {
        'trip_id': trip_id, 'observations': track, 'warnings': warnings,
        'road_network': lines(network_ids, [[] for _ in network_ids], graph),
        'baseline': {'arc_ids': raw, 'connecting_arc_ids': empty, 'segments': lines(raw, empty, graph),
                     'continuity': continuity(raw, graph)},
        'decoded': {'arc_ids': decoded, 'connecting_arc_ids': connectors,
                    'segments': lines(decoded, connectors, graph), 'continuity': continuity(decoded, graph)},
    }


def sequence(ids, links):
    traversals = (arc for a, link in zip(ids, links) for arc in [*link, a])
    return [arc for arc, _ in groupby(traversals) if arc is not None]


def lcs_length(predicted, reference):
    positions = defaultdict(list)
    for i, arc in enumerate(reference):
        positions[arc].append(i)
    tails = []
    for arc in predicted:
        for i in reversed(positions[arc]):
            index = bisect_left(tails, i)
            if index == len(tails):
                tails.append(i)
            else:
                tails[index] = i
    return len(tails)


def metrics(result, trip, rows, reference=None, expected_labels=None, ordered=False):
    output = {'observations': len(trip.track)}
    if reference:
        if ordered:
            from .ordered_labels import align
            labels, _, _ = align(trip, reference)
        else:
            labels = reference_labels(trip, reference)
        covered = {i: int(arc) for i, arc in enumerate(labels) if arc >= 0}
        candidates = {(r['observation_index'], r['arc_id']) for r in rows}
        if expected_labels is not None:
            expected = expected_labels.sort_values('observation_index')
            present = [(i, arc) in candidates for i, arc in covered.items()]
            if (expected.observation_index.tolist() != list(range(len(labels))) or
                    expected.reference_arc_id.tolist() != labels.tolist() or
                    expected.loc[expected.eligible, 'observation_index'].tolist() != list(covered)):
                raise ValueError(f'Reference label mismatch for {trip.trip_id}')
            frozen_present = expected.loc[expected.eligible, 'reference_in_candidates'].tolist()
            output['candidate_coverage_differences'] = sum(a != b for a, b in zip(frozen_present, present))
        output.update(labeled=len(covered), label_coverage=len(covered) / len(labels),
                      candidate_misses=sum(pair not in candidates for pair in covered.items()))
        ref_sequence = reference
        if ordered and rows and 'heading_score' in rows[0]:
            groups = defaultdict(list)
            for row in rows:
                groups[row['observation_index']].append(row)
            output['rules'] = {}
            for name, key in [('nearest_road', 'nearest_score'), ('distance_heading', 'heading_score')]:
                correct = sum(i in groups and max(groups[i], key=lambda r: r[key])['arc_id'] == arc
                              for i, arc in covered.items())
                output['rules'][name] = correct / len(covered) if covered else None
    for name in ('baseline', 'decoded'):
        method = result[name]
        ids = method['arc_ids']
        stats = dict(method['continuity'])
        if reference:
            ordered = sequence(ids, method['connecting_arc_ids'])
            match = lcs_length(ordered, ref_sequence)
            denominator = len(ordered) + len(ref_sequence)
            correct = sum(ids[i] == label for i, label in covered.items())
            stats.update(correct=correct, point_accuracy=correct / len(covered) if covered else None,
                         sequence_matches=match, sequence_denominator=denominator,
                         sequence_f1=2 * match / denominator if denominator else None)
        output[name] = stats
    return output
