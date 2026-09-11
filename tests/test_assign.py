import itertools

import numpy as np
import pytest

from mot.assign import associate, gate, greedy, hungarian


def brute_force(cost: np.ndarray) -> float:
    """Optimal total cost by enumeration -- the reference for small matrices."""
    n, m = cost.shape
    if n > m:
        cost = cost.T
        n, m = m, n
    return min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))


def test_matches_brute_force_on_random_matrices():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n, m = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        C = rng.random((n, m)) * 10
        total = sum(C[i, j] for i, j in hungarian(C))
        assert total == pytest.approx(brute_force(C))


def test_handles_both_orientations():
    rng = np.random.default_rng(1)
    C = rng.random((3, 6))
    tall = hungarian(C)
    wide = hungarian(C.T)
    assert len(tall) == len(wide) == 3
    assert sum(C[i, j] for i, j in tall) == pytest.approx(sum(C.T[i, j] for i, j in wide))


def test_pairs_are_a_matching():
    rng = np.random.default_rng(2)
    for _ in range(100):
        C = rng.random((int(rng.integers(1, 8)), int(rng.integers(1, 8))))
        pairs = hungarian(C)
        assert len({i for i, _ in pairs}) == len(pairs)
        assert len({j for _, j in pairs}) == len(pairs)
        assert len(pairs) == min(C.shape)


def test_returns_integers_not_numpy_scalars():
    # The browser port compares these as JSON; numpy ints serialise differently.
    for i, j in hungarian(np.random.default_rng(5).random((3, 4))):
        assert type(i) is int and type(j) is int


def test_empty_input():
    assert hungarian(np.zeros((0, 0))) == []
    assert greedy(np.zeros((0, 3))) == []


def test_greedy_can_be_beaten():
    # Greedy takes the 1.0 in the top-left and is then stuck with the 9.0.
    C = np.array([[1.0, 2.0], [1.0, 9.0]])
    assert sum(C[i, j] for i, j in greedy(C)) == pytest.approx(10.0)
    assert sum(C[i, j] for i, j in hungarian(C)) == pytest.approx(3.0)


def test_greedy_is_optimal_when_every_object_is_isolated():
    C = np.array([[0.1, 8.0, 9.0], [7.0, 0.2, 8.5], [9.0, 8.0, 0.3]])
    assert greedy(C) == hungarian(C)


def test_gate_drops_pairs_the_solver_had_to_invent():
    C = np.array([[0.5, 90.0], [90.0, 0.4]])
    assert len(gate(hungarian(C), C, 1.0)) == 2
    C2 = np.array([[0.5, 90.0], [90.0, 80.0]])
    assert gate(hungarian(C2), C2, 1.0) == [(0, 0)]


def test_associate_reports_what_was_left_over():
    C = np.array([[0.5, 90.0, 91.0], [90.0, 92.0, 93.0]])
    matched, un_t, un_d = associate(C, threshold=1.0)
    assert matched == [(0, 0)]
    assert un_t == [1]
    assert sorted(un_d) == [1, 2]


def test_associate_with_no_tracks_or_no_detections():
    matched, un_t, un_d = associate(np.zeros((0, 0)), threshold=1.0)
    assert matched == [] and un_t == [] and un_d == []


def test_tie_breaking_is_deterministic():
    # Every pair costs the same; the answer must still be reproducible, because
    # the JavaScript port is checked against it.
    C = np.ones((4, 4))
    first = hungarian(C)
    assert all(hungarian(C) == first for _ in range(10))
