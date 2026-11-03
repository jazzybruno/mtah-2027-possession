"""Build a possession submission from the provided data, and score it on the public window.

    python -m possession.pipeline                              # writes evaluation/results/possession.csv
    python -m possession.pipeline --set cost_team=16           # override any Params field
    python -m possession.pipeline --sweep cost_team=8,12,16    # one public-window line per value

The public window is 17 possessions: use it to catch broken pipelines, not to pick parameters.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402

from possession import atd as atd_mod  # noqa: E402
from possession import tagging  # noqa: E402
from possession.config import Params, with_overrides  # noqa: E402
from possession.deadmodel import learned_logit  # noqa: E402
from possession.decode import clean_runs, decode  # noqa: E402
from possession.evidence import (apply_tags, dead_evidence, locate_restarts,  # noqa: E402
                                 team_evidence)
from possession.shape import shape_logit  # noqa: E402

log = logging.getLogger("possession")

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "possession.csv")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))


def public_gt_codes():
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT))


def match_context(game=GAME):
    """What predict needs to know about a match: graded halves, length, frame rate, pitch."""
    return {"blocks": half_video_blocks(game), "n_frames": source_frames("tactical", game),
            "fps": GAMES[game]["fps"], "pitch": GAMES[game]["pitch"]}


def predict(atd, tags, params, context=None, diagnostics=False):
    """Full-length H/A/D/X label array for one parameter set; with diagnostics, also a dict of
    the intermediate per-frame scores. context defaults to the challenge match."""
    context = context or match_context()
    fps, blocks = context["fps"], context["blocks"]
    length, width = context["pitch"]

    team_score, ballinfo = team_evidence(atd, params, fps)
    ball_says_nothing = team_score == 0
    proximity = team_score.copy()
    learned = shape = None
    if params.use_learned and tags is not None:
        learned = learned_logit(atd, ballinfo, tags, params, fps, blocks, length, width)
    dead_score = dead_evidence(atd, ballinfo, params, fps, blocks, length, width, learned)
    if tags is not None:
        apply_tags(team_score, dead_score, ballinfo["speed"], tags, params, fps, blocks)
    n_frames = context["n_frames"]
    if params.effective_play_min > 0:
        dead_score = dead_score + calibrated_offset(team_score, dead_score, params, blocks,
                                                    n_frames, fps)
    labels = decode(team_score, dead_score, blocks, n_frames, params.cost_team, params.cost_dead)

    if params.use_shape:
        # Second pass: team shape, trained on the confident first-pass frames, fills in where
        # the ball is missing or far from everyone.
        kickoffs = ([r for r, piece in locate_restarts(ballinfo["speed"], tags, params, fps)
                     if piece["type"] == "kick off (start)"] if tags is not None else [])
        shape = shape_logit(atd, ballinfo["ball"], proximity, labels, kickoffs, params, blocks,
                            length, fps)
        weight = np.where(ball_says_nothing, params.w_shape, params.w_shape * params.shape_on_ball)
        nudge = weight * np.clip(shape, -params.shape_clip, params.shape_clip)
        # Decoder scores are H +e/2, A -e/2: any team evidence lifts alive over dead by |e|/2.
        # Shape should pick the side, not argue the ball is in play, so cancel that lift.
        team_score = team_score + nudge
        dead_score = dead_score + np.abs(nudge) / 2
        labels = decode(team_score, dead_score, blocks, n_frames, params.cost_team,
                        params.cost_dead)
    if params.clean:
        labels = clean_runs(labels, blocks, int(params.min_dead_run_s * fps),
                            int(params.max_sandwich_s * fps))
    if diagnostics:
        return labels, {"learned": learned, "shape": shape, "ball_says_nothing": ball_says_nothing}
    return labels


def graded_frames(blocks):
    return np.concatenate([np.arange(first, last + 1) for _, first, last in blocks])


def calibrated_offset(team_score, dead_score, params, blocks, n_frames, fps):
    """Offset for dead_score so the decoded graded frames hold params.effective_play_min of
    ball-in-play time. Bisection: the dead share falls monotonically as the offset drops.

    The target is an external league statistic, never a ground-truth quantity.
    """
    graded = graded_frames(blocks)
    target = 1.0 - params.effective_play_min * 60 * fps / len(graded)
    low, high = -4.0, 4.0
    share = None
    for _ in range(params.calibration_steps):
        mid = (low + high) / 2
        labels = decode(team_score, dead_score + mid, blocks, n_frames, params.cost_team,
                        params.cost_dead)
        if params.clean:
            labels = clean_runs(labels, blocks, int(params.min_dead_run_s * fps),
                                int(params.max_sandwich_s * fps))
        share = float(np.mean(labels[graded] == "D"))
        if share < target:
            low = mid
        else:
            high = mid
    log.info("dead-ball calibration: target %.3f dead, reached %.3f with offset %+.3f",
             target, share, (low + high) / 2)
    return (low + high) / 2


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def summary(labels, game=GAME):
    """Possession count and mean duration over the graded halves -- our own output only."""
    fps = GAMES[game]["fps"]
    spells = []
    for _, first, last in half_video_blocks(game):
        window = labels[first:last + 1]
        change = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(window)]))
        spells += [b - a for a, b in zip(bounds[:-1], bounds[1:]) if window[a] in ("H", "A")]
    graded = np.concatenate([labels[f:l + 1] for _, f, l in half_video_blocks(game)])
    shares = {k: float(np.mean(graded == k)) for k in "HAD"}
    return "possessions {:4d}  mean {:5.1f}s  H {:.2f} A {:.2f} D {:.2f}".format(
        len(spells), np.mean(spells) / fps if spells else 0.0, shares["H"], shares["A"],
        shares["D"])


def score_public(labels):
    gt = public_gt_codes()
    result = pm.evaluate(gt, pm.label_codes(labels))
    return "public: spellF1 {:.4f}  macroF1 {:.4f}  poss {:3d} (gt {:3d})".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
        result["n_gt_spells"])


def tag_consistency(labels, tags, fps):
    """Agreement between predicted stoppages and Smart Tagging set pieces, as two fractions.

    Run it with --no-tags so the tags are an independent reference: recall is the share of set
    pieces with a predicted D within [-4 s, +2 s] of the tag start; hit rate is the share of
    predicted dead runs of 2 s or more with a set-piece tag within 10 s. Set pieces cover only part
    of all stoppages, so the hit rate is a floor, not a precision.
    """
    starts = np.array(sorted(piece["start_frame"] for piece in tags["set_pieces"]))
    dead = labels == "D"
    recall = np.mean([dead[max(0, s - 4 * fps):s + 2 * fps].any() for s in starts])

    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    runs = [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if b - a >= 2 * fps]
    hits = [np.any((starts >= a - 10 * fps) & (starts <= b + 10 * fps)) for a, b in runs]
    return "tags: set-piece recall {:.2f}  dead-run hit rate {:.2f}  ({} dead runs >= 2s)".format(
        recall, np.mean(hits) if hits else 0.0, len(runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    parser.add_argument("--sweep", metavar="NAME=V1,V2,...",
                        help="score each value on the public window; writes no submission")
    parser.add_argument("--no-tags", action="store_true", help="ignore Smart Tagging")
    parser.add_argument("--check", action="store_true",
                        help="also predict without tags and report agreement with them")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the tracking file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    started = time.time()
    params = with_overrides(Params(), args.set)
    atd = atd_mod.load(use_cache=not args.no_cache)
    tags = None if args.no_tags else tagging.load()

    if args.sweep:
        name, _, values = args.sweep.partition("=")
        for value in values.split(","):
            labels = predict(atd, tags, with_overrides(params, ["{}={}".format(name, value)]))
            print("{}={:<8} {}  |  {}".format(name, value, summary(labels),
                                              score_public(labels)))
        return

    if args.check:
        print(tag_consistency(predict(atd, None, params), tagging.load(), GAMES[GAME]["fps"]))

    labels = predict(atd, tags, params)
    write_submission(args.out, labels)
    print(summary(labels))
    print(score_public(labels))
    print("wrote {}  ({:.0f}s)".format(os.path.relpath(args.out, ROOT), time.time() - started))


if __name__ == "__main__":
    main()
