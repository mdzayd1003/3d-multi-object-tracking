# 3D multi-object tracking

Tracking-by-detection over oriented 3D boxes: a constant-velocity Kalman filter, two
assignment solvers, two association costs, and CLEAR MOT / AMOTA scoring — with the
choices between them measured rather than asserted.

**[Interactive demo →](https://mdzayd1003.github.io/3d-multi-object-tracking/)** — the whole
pipeline runs in the browser. Drag the noise slider and watch IoU association come apart.

```
make test         # 62 tests, ~15 s, no network, no GPU
make quick        # the four experiments below
make crosscheck   # hold the browser port to the Python
```

---

## The question

Nearly every 3D tracker is the same four steps — predict, associate, update, birth/death —
and nearly every design decision in it is folklore. This repository picks four of those
decisions and measures what they are actually worth on scenes where the answer is knowable.

All numbers below are produced by `python -m mot --all` and are pinned by
`tests/test_experiments.py`, so they cannot quietly stop being true.

---

## 1. IoU association has a ceiling, and it is not where you would guess

The standard cost is `1 − IoU`. The nuScenes benchmark uses centre distance instead. The
usual justification is that IoU goes to zero for non-overlapping boxes and stops
discriminating. That explanation is wrong here, and the real one is more restrictive.

Sweeping detector localisation error on the `crossing` scene:

| noise (m) | IoU MOTA | IDS | dist MOTA | IDS | true pairs with IoU = 0 |
|-----------|---------:|----:|----------:|----:|------------------------:|
| 0.10 | +0.969 | 0 | +0.969 | 0 | 0.0% |
| 0.50 | +0.960 | 0 | +0.960 | 0 | 0.0% |
| 0.75 | +0.785 | 10 | **+0.885** | 3 | 0.9% |
| 1.00 | +0.567 | 29 | **+0.854** | 4 | 3.7% |
| 1.25 | +0.090 | 45 | **+0.592** | 21 | 8.9% |
| 2.00 | −0.717 | 62 | −0.692 | 54 | 37.5% |

Three things to read off it.

**Below half a metre they are interchangeable, and IoU is marginally ahead.** Requiring
physical overlap rejects clutter that a 4 m distance gate admits — at noise 0.25 on the
`sparse` scene, IoU gives 0 false positives against distance's 12. IoU is not a bad cost.

**The collapse is not the zero-IoU story.** At noise 1.0, where IoU has already given up a
third of its MOTA and seven times the ID switches, only **3.7%** of true (object, detection)
pairs fail to overlap at all, and the cost still ranks the correct detection first more than
99% of the time. Ranking is not the problem.

**The problem is that the gate cannot be widened.** Sweeping the gate at noise 1.0:

```
iou  gate 0.7     MOTA=-0.098   iou  gate 0.99    MOTA=+0.517
iou  gate 0.9     MOTA=+0.194   iou  gate 0.999   MOTA=+0.567
iou  gate 0.95    MOTA=+0.312   iou  gate 0.9999  MOTA=+0.581   <- ceiling
dist gate 2.0     MOTA=-0.033   dist gate 4.0     MOTA=+0.854
dist gate 3.0     MOTA=+0.550   dist gate 6.0     MOTA=+0.825
```

`1 − IoU ≤ 0.999` already means "admit any pair of boxes that touch at all". There is
nothing past it: 0.9999 buys 0.014 MOTA and the curve is flat. For a 4×2 m car that ceiling
is worth roughly a **3 m** centre gate (0.550 against IoU's ceiling of 0.581) — and the metre
beyond it, which no IoU threshold can reach, is worth **+0.27 MOTA**. IoU's support is bounded
by the size of the objects being tracked; a distance gate is a free parameter. That is the
whole difference.

None of which rescues distance at 2 m error, where 14.8% of true pairs are outside any sane
gate and both costs go negative. The claim is about the range in between, which is where
real detectors sit.

## 2. Greedy assignment beats Hungarian here, for a structural reason

Hungarian finds the minimum-cost complete matching. Greedy repeatedly takes the cheapest
free pair. Hungarian is optimal, so it should win. It does not:

| scene | noise | Hungarian | greedy | frames they disagreed | pairs Hungarian gave up |
|-------|------:|----------:|-------:|----------------------:|------------------------:|
| sparse | 0.50 | +0.917 | **+0.958** | 6 | 6 |
| sparse | 1.00 | +0.803 | **+0.881** | 8 | 9 |
| crossing | 0.50 | **+0.960** | +0.956 | 4 | 5 |
| crossing | 1.00 | +0.854 | **+0.875** | 7 | 8 |
| dense | 0.50 | +0.944 | **+0.980** | 11 | 13 |
| dense | 1.00 | +0.883 | **+0.915** | 17 | 20 |

Hungarian is optimal for the problem it is given, and that problem is the wrong one. It
minimises the total cost of a *complete* matching, which forces every track to be paired with
something — including, in a frame where a track's real detection was missed, some piece of
clutter on the far side of the scene. Gating afterwards discards that pair, but the optimum
was already computed with its cost included, so Hungarian will trade away a genuinely cheap
pair to reduce the cost of two pairs that never survive gating. The last column counts
exactly those: pairs greedy kept, Hungarian dropped, and the gate would have accepted.

The fix is not "use greedy" — it is that the cost matrix should be extended with explicit
dummy rows priced at the gate, so that leaving a track unmatched is a move the solver can
choose rather than one imposed on it afterwards. This repository implements the naive
version and measures what it costs, because that is the version most trackers ship.

## 3. `max_age` buys ID switches, then only buys clutter

The usual description is that raising `max_age` trades false negatives for ID switches. Only
half of that survives contact (`crossing`, noise 0.6):

| max_age | MOTA | IDS | FN | FP |
|--------:|-----:|----:|---:|---:|
| 0 | +0.633 | 52 | 123 | 1 |
| 1 | +0.933 | 10 | 21 | 1 |
| 2 | **+0.946** | 2 | 11 | 13 |
| 3 | +0.940 | 1 | 13 | 15 |
| 8 | +0.890 | 1 | 13 | 39 |
| 12 | +0.865 | 1 | 13 | 51 |

Past `max_age = 2` the misses are **flat at 13** and the ID switches are **flat at 1**. Every
frame of extra coasting buys nothing and costs false positives, which grow to 51. The knee is
sharp and it is a clutter limit, not a recall/identity trade.

## 4. MOTA and AMOTA disagree about which tracker is better

MOTA is reported at one operating point. AMOTA sweeps the confidence threshold and averages
over recall. On the same three configurations they pick different winners:

| max_age | MOTA @ threshold 0 | AMOTA |
|--------:|-------------------:|------:|
| 1 | **+0.896** | 0.850 |
| 3 | +0.885 | **0.936** |
| 8 | +0.750 | 0.886 |

A short `max_age` looks best when every detection is kept, because clutter has no time to
accumulate. Raise the threshold and the picture inverts: the detections that survive are the
confident ones, misses become the binding constraint, and the configuration that can coast
through them wins. Reporting either number alone picks a different tracker.

---

## Implementation notes

Four things here are written out rather than imported, each because getting them wrong is
quiet rather than loud.

**Yaw is an angle.** The Kalman innovation on heading is wrapped into `[−π, π)` before use.
Without it, a track crossing the branch cut sees a residual of nearly 2π and the correction
drags the position state with it. `test_kalman.py::test_yaw_wraparound_does_not_throw_the_box`.

**Covariance uses the Joseph form.** `(I − KH)P` loses symmetry over the hundreds of
predict/update cycles a long track goes through. The test runs 500 steps and checks symmetry
every one of them, then checks the eigenvalues at the end.

**MOTA's identity accounting carries the previous frame's mapping forward.** Re-solving each
frame from scratch makes the ID switch count a function of solver tie-breaking rather than of
tracking quality — two near-equidistant hypotheses will trade places forever. The test
constructs exactly that case and shows the naive version reporting switches no tracker
committed.

**AMOTA's recall sweep is discrete, and the residual is reported.** MOTAR charges misses
against a `(1 − r)·P` allowance that assumes the operating point achieves recall `r` exactly.
A finite threshold sweep cannot, and when the achieved recall overshoots the target the
formula credits recall that was never asked for — MOTAR above 1. The sweep here is fine
enough to keep the overshoot small, the result is clipped to `[0, 1]`, and `max_recall_gap`
reports what is left instead of hiding it.

One thing was caught by a test rather than by design: the `crossing` scene originally gave
crossing objects a fixed lateral drift smaller than the lane spacing, so they never actually
changed lateral order and no solver was ever forced to choose — which would have made this
entire comparison vacuous. The drift is now scaled to lane spacing and sequence length, and
`test_scenes.py::test_crossing_objects_actually_cross` asserts the ordering really flips.

## The browser port

`docs/mot.js` is the whole deterministic pipeline in JavaScript — RNG, scene generation, box
geometry, the filter, both solvers, the tracker and CLEAR MOT — so the demo computes instead
of replaying a recording. CI checks it against the Python on every push, to two standards:

- **Exactly**: every assignment pair, the set of track ids reported in each frame, and the ID
  switch / false positive / miss counts. These are decisions, not measurements. If the two
  disagree about which detection belongs to which track, the demo is not showing the library.
- **To 1e−9**: the continuous values. The Kalman update inverts a 7×7 matrix and numpy's
  LAPACK inverse need not agree in the last bits with the Gauss-Jordan routine in the port.

Current result: **33,851 comparisons, every discrete output identical, worst continuous
difference 3.6e−15** — about four orders of magnitude inside the tolerance.

## Layout

```
mot/boxes.py       oriented 3D IoU via polygon clipping; both association costs
mot/kalman.py      constant-velocity filter, yaw wrapping, Joseph-form covariance
mot/assign.py      Hungarian (shortest augmenting path), greedy, gating
mot/tracker.py     predict / associate / update / birth / death
mot/scenes.py      synthetic scenes and a detector with tunable recall and clutter
mot/metrics.py     CLEAR MOT with sticky identity accounting; AMOTA / AMOTP
mot/experiments.py the four measurements above
docs/              the interactive demo and the JavaScript port
scripts/           reference generation and the cross-port check
```

## Scope

The scenes are synthetic. That is the point — ground truth is exact, the detector's recall,
clutter and localisation error are knobs rather than estimates, and every tracker error is
attributable. What it does not give you is a claim about nuScenes or KITTI: the numbers here
are about the *relationships* between these design choices, not about absolute performance on
real sensor data. The geometry, filter, solvers and metrics are the real ones and would run
unchanged on real detections.
