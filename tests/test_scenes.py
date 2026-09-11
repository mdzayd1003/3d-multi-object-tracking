import subprocess
import sys

import numpy as np
import pytest

from mot.scenes import SCENES, make_scene


def test_same_seed_gives_an_identical_scene():
    a = make_scene("crossing", seed=3)
    b = make_scene("crossing", seed=3)
    for fa, fb in zip(a.frames, b.frames):
        assert sorted(fa.gt) == sorted(fb.gt)
        for k in fa.gt:
            assert np.array_equal(fa.gt[k], fb.gt[k])
        assert len(fa.detections) == len(fb.detections)
        for da, db in zip(fa.detections, fb.detections):
            assert np.array_equal(da.box, db.box) and da.score == db.score


def test_different_seeds_give_different_scenes():
    a = make_scene("crossing", seed=3)
    b = make_scene("crossing", seed=4)
    assert not np.array_equal(a.frames[0].gt[0], b.frames[0].gt[0])


def test_determinism_survives_a_fresh_interpreter():
    # Nothing in the generator may depend on hash randomisation.
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from mot.scenes import make_scene;"
        "s = make_scene('crossing', seed=3);"
        "print(round(float(s.frames[10].detections[0].box[0]), 9))"
    )
    seen = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": h, "PATH": "/usr/bin:/bin"}).stdout.strip()
        for h in ("0", "1", "12345")
    }
    assert len(seen) == 1, seen


@pytest.mark.parametrize("name", sorted(SCENES))
def test_detector_rates_match_the_configuration(name):
    miss, fp_rate, n_frames = 0.2, 1.0, 80
    sc = make_scene(name, seed=11, n_frames=n_frames, miss_rate=miss, fp_per_frame=fp_rate)
    tp = sum(1 for f in sc.frames for d in f.detections if d.gt_id >= 0)
    fp = sum(1 for f in sc.frames for d in f.detections if d.gt_id < 0)
    assert tp / sc.total_gt() == pytest.approx(1 - miss, abs=0.05)
    assert fp / n_frames == pytest.approx(fp_rate, abs=0.15)


def test_zero_noise_detections_sit_exactly_on_the_truth():
    sc = make_scene("sparse", seed=5, noise=0.0, miss_rate=0.0, fp_per_frame=0.0)
    for f in sc.frames:
        for d in f.detections:
            assert np.allclose(d.box, f.gt[d.gt_id])


def test_every_object_is_present_in_every_frame():
    sc = make_scene("dense", seed=5)
    for f in sc.frames:
        assert len(f.gt) == sc.n_objects


def test_crossing_objects_actually_cross():
    # If they did not, association would never have to choose and the whole
    # comparison in this repository would be vacuous.
    sc = make_scene("crossing", seed=7)
    first, last = sc.frames[0], sc.frames[-1]
    order_first = sorted(first.gt, key=lambda k: first.gt[k][1])
    order_last = sorted(last.gt, key=lambda k: last.gt[k][1])
    assert order_first != order_last


def test_scores_separate_true_from_false_on_average_but_not_cleanly():
    sc = make_scene("crossing", seed=7, noise=0.4)
    tp = [d.score for f in sc.frames for d in f.detections if d.gt_id >= 0]
    fp = [d.score for f in sc.frames for d in f.detections if d.gt_id < 0]
    assert np.mean(tp) > np.mean(fp)
    # Overlapping ranges are the point: a single threshold cannot remove clutter
    # without also removing real objects, which is what AMOTA sweeps over.
    assert min(tp) < max(fp)


def test_unknown_scene_is_rejected():
    with pytest.raises(ValueError):
        make_scene("motorway")
