import numpy as np

from pathsnap.candidates import Trip
from pathsnap.core import RoadGraph, decode
from pathsnap.grouped_models import observation_scores, softmax
from pathsnap.ordered_labels import align


def test_alignment_retains_reverse_traversal_and_revisit():
    trip = Trip('00000001', np.array([[x, 0., t * 3.] for t, x in enumerate(
        (.0001, .0004, .0009, .0008, .0004, .0001, .0002, .0006, .0009))]),
        np.array([[0., 0.], [.001, 0.]]), np.array([[0, 1], [1, 0]]))
    labels, occurrence, _ = align(trip, [0, 1, 0])
    assert len(labels) == len(trip.track)
    assert np.all(np.diff(occurrence) >= 0)
    assert occurrence[-1] == 2
    assert occurrence[4] == 1


def test_grouped_loss_counts_missed_candidate():
    p = softmax(np.array([2., 1., 100.]), np.array([2, 1]))
    assert np.allclose([p[:2].sum(), p[2]], [1., 1.])
    accuracy, loss = observation_scores(np.array([2., 1., 100.]),
                                        np.array([True, False, False]), np.array([2, 1]))
    assert accuracy == .5
    assert loss > 10


def test_long_gap_bridges_only_if_physically_plausible():
    graph = RoadGraph([(0., 0.), (.001, 0.), (.002, 0.)], [(0, 1), (1, 2)])
    track = [dict(lon=.0002, lat=0., timestamp=0.), dict(lon=.0018, lat=0., timestamp=40.)]
    rows = {i: [dict(arc_id=i, snapped_lon=o['lon'], snapped_lat=0., probability=1.)]
            for i, o in enumerate(track)}
    _, _, bridged = decode(track, graph, rows, bridge_gaps=True, max_speed_mps=20.)
    _, _, split = decode(track, graph, rows, bridge_gaps=True, max_speed_mps=2.)
    assert 'bridged' in bridged[0]
    assert 'disconnected' in split[0]
