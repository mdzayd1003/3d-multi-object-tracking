"""Command line: run one experiment or all of them, print a table, save JSON."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .experiments import ALL


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _print_cost_ceiling(r):
    print(f"\n== cost_ceiling ({r['scene']}) ==")
    print(f"{'noise':>6} {'IoU=0':>7} {'>gate':>7} | {'iou MOTA':>9} {'IDS':>4} | {'dist MOTA':>10} {'IDS':>4}")
    for row in r["rows"]:
        print(f"{row['noise']:6.2f} {100*row['zero_iou_frac']:6.1f}% {100*row['beyond_dist_gate_frac']:6.1f}%"
              f" | {row['iou']['mota']:+9.3f} {row['iou']['ids']:4d}"
              f" | {row['dist']['mota']:+10.3f} {row['dist']['ids']:4d}")
    gs = r["gate_sweep"]
    print(f"  gate sweep at noise {gs['noise']}:")
    for row in gs["iou"]:
        print(f"    iou  gate {row['gate']:<7} MOTA={row['mota']:+.3f}  IDS={row['ids']:3d}  FN={row['fn']:4d}")
    for row in gs["dist"]:
        print(f"    dist gate {row['gate']:<7} MOTA={row['mota']:+.3f}  IDS={row['ids']:3d}  FN={row['fn']:4d}")


def _print_solver_gap(r):
    print("\n== solver_gap ==")
    print(f"{'scene':>9} {'noise':>6} | {'hung':>7} {'greedy':>7} | {'disagreed':>9} {'sacrificed':>11}")
    for row in r["rows"]:
        print(f"{row['scene']:>9} {row['noise']:6.2f} | {row['hungarian']['mota']:+7.3f}"
              f" {row['greedy']['mota']:+7.3f} | {row['frames_disagreed']:9d} {row['sacrificed']:11d}")


def _print_age(r):
    print(f"\n== age_tradeoff ({r['scene']}, noise {r['noise']}) ==")
    print(f"{'max_age':>8} {'MOTA':>8} {'IDS':>5} {'FN':>6} {'FP':>6}")
    for row in r["rows"]:
        print(f"{row['max_age']:8d} {row['mota']:+8.3f} {row['ids']:5d} {row['fn']:6d} {row['fp']:6d}")


def _print_amota(r):
    print(f"\n== amota_disagreement ({r['scene']}, noise {r['noise']}) ==")
    print(f"{'max_age':>8} {'MOTA@0':>8} {'AMOTA':>8} {'AMOTP':>8}")
    for row in r["rows"]:
        print(f"{row['max_age']:8d} {row['mota_at_0']:+8.3f} {row['amota']:8.3f} {row['amotp']:8.3f}")
    print(f"  best by MOTA: max_age={r['best_by_mota']}   best by AMOTA: max_age={r['best_by_amota']}"
          f"   {'-> they disagree' if r['disagree'] else '-> they agree'}")


PRINTERS = {
    "cost_ceiling": _print_cost_ceiling,
    "solver_gap": _print_solver_gap,
    "age_tradeoff": _print_age,
    "amota_disagreement": _print_amota,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m mot")
    p.add_argument("experiment", nargs="?", choices=sorted(ALL), help="one experiment; omit with --all")
    p.add_argument("--all", action="store_true", help="run every experiment")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/smoke", help="directory for JSON output")
    a = p.parse_args(argv)

    if not a.all and not a.experiment:
        p.error("give an experiment name or --all")

    names = sorted(ALL) if a.all else [a.experiment]
    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for name in names:
        result = ALL[name](seed=a.seed)
        PRINTERS[name](result)
        path = outdir / f"{name}.json"
        path.write_text(json.dumps(result, indent=2, default=_jsonable))
    print(f"\nwrote {len(names)} file(s) to {outdir}/")
    return 0
