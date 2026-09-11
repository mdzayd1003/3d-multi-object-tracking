"""The four claims in the README, asserted rather than trusted.

These are deliberately stated as inequalities with slack, not as pinned
numbers: the point is that the qualitative findings hold, and a test that
breaks whenever a constant moves in the fourth decimal teaches nothing.
"""
import pytest

from mot.experiments import age_tradeoff, amota_disagreement, cost_ceiling, solver_gap


@pytest.fixture(scope="module")
def ceiling():
    return cost_ceiling()


def test_iou_and_distance_are_equivalent_while_boxes_still_overlap(ceiling):
    for row in ceiling["rows"]:
        if row["noise"] <= 0.5:
            assert abs(row["iou"]["mota"] - row["dist"]["mota"]) < 0.02


def test_iou_collapses_first_as_localisation_degrades(ceiling):
    row = next(r for r in ceiling["rows"] if r["noise"] == 1.0)
    assert row["dist"]["mota"] - row["iou"]["mota"] > 0.2
    assert row["iou"]["ids"] > 4 * row["dist"]["ids"]


def test_the_collapse_is_not_explained_by_iou_reaching_zero(ceiling):
    # The usual story is that IoU stops working because true pairs stop
    # overlapping. Where the collapse happens, almost all of them still do.
    row = next(r for r in ceiling["rows"] if r["noise"] == 1.0)
    assert row["zero_iou_frac"] < 0.10
    assert row["beyond_dist_gate_frac"] < 0.01


def test_the_iou_gate_saturates_but_the_distance_gate_does_not(ceiling):
    sweep = ceiling["gate_sweep"]
    iou = {r["gate"]: r["mota"] for r in sweep["iou"]}
    dist = {r["gate"]: r["mota"] for r in sweep["dist"]}
    # 0.999 already admits any pair that overlaps at all, so going further
    # cannot buy anything -- that is the ceiling.
    assert abs(iou[0.9999] - iou[0.999]) < 0.03
    # The distance gate has no such bound, and the extra reach is worth a lot.
    assert max(dist.values()) - max(iou.values()) > 0.2
    # The loosest IoU gate is worth roughly a 3 m distance gate; the metre
    # beyond that is where the difference lives.
    assert abs(dist[3.0] - max(iou.values())) < 0.1


def test_both_costs_fail_once_the_geometry_genuinely_breaks(ceiling):
    # Not a defence of distance: at 2 m error the true pair is often outside
    # any sane gate and neither cost has anything left to work with.
    row = next(r for r in ceiling["rows"] if r["noise"] == 2.0)
    assert row["iou"]["mota"] < 0.0 and row["dist"]["mota"] < 0.0


def test_greedy_is_not_worse_than_hungarian_here():
    res = solver_gap()
    wins = sum(1 for r in res["rows"] if r["greedy"]["mota"] > r["hungarian"]["mota"])
    assert wins >= 4, [(r["scene"], r["noise"]) for r in res["rows"]]


def test_hungarian_gives_up_pairs_the_gate_would_have_kept():
    # The mechanism behind the previous test: Hungarian minimises the total cost
    # of a complete matching, including pairs that gating will discard, and pays
    # for that with pairs that would have survived.
    res = solver_gap()
    total_sacrificed = sum(r["sacrificed"] for r in res["rows"])
    assert total_sacrificed > 0
    for r in res["rows"]:
        assert r["gated_pairs_greedy"] >= r["gated_pairs_hungarian"]


def test_raising_max_age_stops_buying_recall_and_only_adds_clutter():
    res = age_tradeoff()
    rows = {r["max_age"]: r for r in res["rows"]}
    assert rows[0]["ids"] > 10 * rows[3]["ids"]          # coasting does fix switches
    assert rows[12]["fn"] == rows[3]["fn"]               # but misses stop improving
    assert rows[12]["fp"] > 3 * rows[3]["fp"]            # while clutter keeps growing
    best = max(res["rows"], key=lambda r: r["mota"])["max_age"]
    assert 1 <= best <= 3


def test_mota_and_amota_disagree_about_the_best_configuration():
    res = amota_disagreement()
    assert res["disagree"]
    assert res["best_by_mota"] != res["best_by_amota"]
    for r in res["rows"]:
        assert 0.0 <= r["amota"] <= 1.0
