from types import SimpleNamespace

import numpy as np

from pathsnap.candidates import Trip
from pathsnap.grouped_features import make_candidates
from pathsnap.review_experiment import point_stats


def test_covered_loss_excludes_a_miss_but_accuracy_counts_it():
    trip = SimpleNamespace(counts=np.array([2, 2]), target=np.array([True, False, False, False]))
    report = point_stats(np.zeros(4), trip, np.zeros(2), 1.)
    assert report['trusted_observations'] == 2
    assert report['trusted_point_correct'] == report['trusted_misses'] == report['covered_trusted'] == 1
    assert np.isclose(report['covered_log_loss_sum'], np.log(2.))
    assert report['all_log_loss_sum'] > 27.


def test_context_features_do_not_link_across_large_gap():
    trip = Trip('00000001', np.array([[0., 0., 0.], [.0001, 0., 1.], [.001, 0., 90.]]),
                np.array([[0., 0.], [.002, 0.]]), np.array([[0, 1], [1, 0]]))
    _, table = make_candidates(trip, context=True)
    before = table[table.observation_index == 1]
    after = table[table.observation_index == 2]
    assert (before.outgoing_support == 0).all()
    assert (after.incoming_support == 0).all()
    assert (after.motion_30s_m == 0).all()
