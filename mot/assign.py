"""Assignment: an O(n^3) Hungarian solver, a greedy baseline, and gating.

The solver is written out rather than taken from scipy because the browser demo
has to produce the *same* assignment, not merely an equally good one.  On a cost
matrix with ties -- which is exactly what 1 - IoU produces once boxes stop
overlapping -- "equally good" covers many different identity assignments, so a
cross-language check is only meaningful if both sides break ties the same way.
Both ports visit rows and columns in ascending index order.
"""
from __future__ import annotations

import numpy as np

__all__ = ["hungarian", "greedy", "gate", "associate"]

_INF = float("inf")


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost perfect matching on the smaller dimension.

    Jonker-Volgenant style shortest augmenting path with potentials, which is
    the standard O(n^2 m) formulation.  Returns (row, col) pairs sorted by row.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    n, m = cost.shape
    transposed = n > m
    if transposed:
        cost = cost.T
        n, m = m, n

    # u, v are the dual potentials; way[j] reconstructs the augmenting path.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)   # p[j] = row matched to column j
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, _INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] > 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


def greedy(cost: np.ndarray) -> list[tuple[int, int]]:
    """Repeatedly take the globally cheapest free pair.

    Cheaper and easier to reason about than Hungarian, and identical to it when
    every object is far from every other.  Included so the crowding experiment
    has something to show a difference against.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    n, m = cost.shape
    order = np.argsort(cost, axis=None, kind="stable")
    rows_taken = set()
    cols_taken = set()
    pairs = []
    for flat in order:
        i, j = divmod(int(flat), m)
        if i in rows_taken or j in cols_taken:
            continue
        rows_taken.add(i)
        cols_taken.add(j)
        pairs.append((i, j))
        if len(pairs) == min(n, m):
            break
    return sorted(pairs)


def gate(pairs, cost: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Drop matches the solver made only because it had to fill the matching."""
    return [(i, j) for i, j in pairs if cost[i, j] <= threshold]


def associate(cost: np.ndarray, threshold: float, solver: str = "hungarian"):
    """Solve, gate, and report what was left over.

    Gating after solving rather than before is deliberate: an ungated solver has
    to place every row somewhere, and a track whose true detection is missing
    this frame will otherwise be handed someone else's.
    """
    n, m = (cost.shape if cost.size else (0, 0))
    pairs = (greedy if solver == "greedy" else hungarian)(cost)
    matched = gate(pairs, cost, threshold) if cost.size else []
    used_rows = {i for i, _ in matched}
    used_cols = {j for _, j in matched}
    return (
        matched,
        [i for i in range(n) if i not in used_rows],
        [j for j in range(m) if j not in used_cols],
    )
