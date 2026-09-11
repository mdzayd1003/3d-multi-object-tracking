import numpy as np
import pytest

from mot.metrics import MOTAccumulator, amota, clear_mot
from mot.scenes import make_scene
from mot.tracker import TrackerConfig, run


def box(x, y=0.0):
    return np.array([x, y, 0.0, 0.0, 4.0, 2.0, 1.5])


def test_perfect_tracking_scores_one():
    acc = MOTAccumulator("dist")
    for t in range(10):
        acc.update({0: box(t), 1: box(t, 10.0)}, {5: box(t), 6: box(t, 10.0)})
    r = acc.report()
    assert r["mota"] == pytest.approx(1.0)
    assert r["ids"] == 0 and r["fp"] == 0 and r["fn"] == 0
    assert r["motp"] == pytest.approx(0.0)


def test_a_missing_hypothesis_is_a_miss_not_a_switch():
    acc = MOTAccumulator("dist")
    acc.update({0: box(0)}, {5: box(0)})
    acc.update({0: box(1)}, {})
    r = acc.report()
    assert r["fn"] == 1 and r["ids"] == 0


def test_reacquiring_the_same_id_after_a_gap_is_free():
    acc = MOTAccumulator("dist")
    acc.update({0: box(0)}, {5: box(0)})
    acc.update({0: box(1)}, {})
    acc.update({0: box(2)}, {5: box(2)})
    assert acc.report()["ids"] == 0


def test_reacquiring_a_different_id_is_a_switch():
    acc = MOTAccumulator("dist")
    acc.update({0: box(0)}, {5: box(0)})
    acc.update({0: box(1)}, {})
    acc.update({0: box(2)}, {9: box(2)})
    assert acc.report()["ids"] == 1


def test_an_extra_hypothesis_is_a_false_positive():
    acc = MOTAccumulator("dist")
    acc.update({0: box(0)}, {5: box(0), 6: box(30.0)})
    r = acc.report()
    assert r["fp"] == 1 and r["fn"] == 0 and r["ids"] == 0


def test_sticky_matching_suppresses_phantom_switches():
    """Two hypotheses that are near-equidistant must not trade identities.

    Both 5 and 6 sit within the match threshold of both ground truth objects,
    and which pairing is cheaper genuinely alternates: in one frame 5 is nearer
    to object 0, in the next it is nearer to object 1.  Re-solving from scratch
    follows that alternation and reports switches that no tracker committed;
    preferring the previous frame's pairing does not.
    """
    gt = {0: box(0.0), 1: box(1.0)}
    hyp_a = {5: box(0.4), 6: box(0.6)}    # cheapest pairing: 0-5, 1-6
    hyp_b = {5: box(0.6), 6: box(0.4)}    # cheapest pairing: 0-6, 1-5

    acc = MOTAccumulator("dist")
    for t in range(12):
        acc.update(gt, hyp_a if t % 2 == 0 else hyp_b)
    assert acc.report()["ids"] == 0

    # The same sequence scored without carrying the mapping forward: each frame
    # is solved independently, so an ID switch is recorded whenever the solver's
    # arbitrary choice flips.
    naive_prev, switches = {}, 0
    for t in range(12):
        hyp = hyp_a if t % 2 == 0 else hyp_b
        from mot.assign import associate
        from mot.boxes import center_distance
        gids, hids = sorted(gt), sorted(hyp)
        C = np.array([[center_distance(gt[g], hyp[h]) for h in hids] for g in gids])
        pairs, _, _ = associate(C, 2.0)
        mapping = {gids[i]: hids[j] for i, j in pairs}
        switches += sum(1 for g, h in mapping.items() if naive_prev.get(g, h) != h)
        naive_prev = mapping
    assert switches > 0


def test_motp_is_the_mean_error_over_matches():
    acc = MOTAccumulator("dist")
    acc.update({0: box(0.0)}, {5: box(0.5)})
    acc.update({0: box(0.0)}, {5: box(1.5)})
    assert acc.report()["motp"] == pytest.approx(1.0)


def test_mota_can_go_negative():
    acc = MOTAccumulator("dist")
    for t in range(5):
        acc.update({0: box(t)}, {k: box(50.0 + 10 * k) for k in range(4)})
    assert acc.report()["mota"] < 0.0


def test_scene_level_scores_are_sane():
    sc = make_scene("crossing", seed=7, noise=0.3)
    r = clear_mot(sc, run(sc, TrackerConfig(metric="dist")))
    assert 0.8 < r["mota"] <= 1.0
    assert r["gt"] == sc.total_gt()
    assert 0.0 <= r["recall"] <= 1.0 and 0.0 <= r["precision"] <= 1.0


def test_amota_is_bounded_and_reports_every_recall_level():
    sc = make_scene("crossing", seed=7, noise=0.5, n_frames=30)
    res = amota(sc, lambda t: run(sc, TrackerConfig(metric="dist", score_threshold=t)))
    assert 0.0 <= res["amota"] <= 1.0
    assert len(res["points"]) == 10
    assert all(0.0 <= p["motar"] <= 1.0 for p in res["points"])
    # The discrete sweep cannot hit every target recall exactly; the mismatch is
    # reported rather than silently absorbed into the score.
    assert res["max_recall_gap"] < 0.15


def test_amota_rewards_a_tracker_that_holds_up_across_the_sweep():
    # A tracker crippled at every operating point must not beat a healthy one.
    sc = make_scene("crossing", seed=7, noise=0.5, n_frames=30)
    good = amota(sc, lambda t: run(sc, TrackerConfig(metric="dist", score_threshold=t)))
    bad = amota(sc, lambda t: run(sc, TrackerConfig(metric="dist", score_threshold=t, max_age=0, min_hits=6)))
    assert good["amota"] > bad["amota"]
