"""A learned dead-ball detector: logistic regression on per-frame context features.

Training labels come from two sources, never from the private ground truth:
  - Smart Tagging pseudo-labels: the seconds before a located set-piece restart, pre-kickoff and
    goal-to-kickoff stretches are dead; the core of in-possession phases is alive.
  - optionally the public 5-minute ground truth.
Plain numpy gradient descent with no randomness, so the same inputs give the same file.
"""

import logging
import warnings

import numpy as np

from possession.evidence import (hard_dead_mask, locate_restarts, moving_nanmean,
                                 outside_masks, player_motion)

log = logging.getLogger(__name__)

DEAD_CODE = 2   # possession_metrics: H 0, A 1, D 2, X 3
X_CODE = 3


def _window_mean(mask_or_values, k):
    return np.nan_to_num(moving_nanmean(np.asarray(mask_or_values, dtype=np.float64), k))


def _run_clock(mask, cap):
    """Frames since mask was last False, capped (how long a condition has held)."""
    out = np.zeros(len(mask))
    count = 0
    for i, m in enumerate(mask):
        count = count + 1 if m else 0
        out[i] = count
    return np.minimum(out, cap)


def features(atd, ballinfo, outside, half_blocks, fps, length, width):
    """(T, F) float matrix and the feature names."""
    ball, speed = ballinfo["ball"], ballinfo["speed"]
    seen = ~np.isnan(ball).any(axis=1)

    motion = player_motion(atd, fps)
    motion_rank = np.zeros(len(motion))
    for _, first, last in half_blocks:
        window = motion[first:last + 1]
        finite = np.isfinite(window)
        ranks = np.full(len(window), 0.5)
        if finite.any():
            order = np.argsort(np.argsort(window[finite]))
            ranks[finite] = order / max(1, finite.sum() - 1)
        motion_rank[first:last + 1] = ranks

    pos = atd["positions"].astype(np.float64)
    n_tracked = np.isfinite(pos[:, :, 0]).sum(axis=1).astype(np.float64)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no tracked players
        spread_x = np.nanstd(pos[:, :, 0], axis=1)
        spread_y = np.nanstd(pos[:, :, 1], axis=1)
        fastest = np.nanmax(atd["speed"].astype(np.float64), axis=1)
        dist = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
        nearest = np.where(np.isfinite(dist), dist, np.inf).min(axis=1)
        line_dist = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1],
                                       width - ball[:, 1]])
        still = seen & (speed < 0.5)
        slow_ball = seen & (speed < 2.0)
        fast_ball = seen & (speed > 8.0)

    nearest = np.where(np.isfinite(nearest), np.minimum(nearest, 20.0), np.nan)
    line_dist = np.clip(line_dist, -5.0, 20.0)
    fastest = np.where(np.isfinite(fastest), np.minimum(fastest, 12.0), np.nan)

    columns = {
        "motion_rank_0.5s": _window_mean(motion_rank, fps // 2),
        "motion_rank_2s": _window_mean(motion_rank, 2 * fps),
        "motion_rank_6s": _window_mean(motion_rank, 6 * fps),
        "motion_trend": (_window_mean(np.concatenate((motion_rank[2 * fps:],
                                                      np.full(2 * fps, 0.5))), fps)
                         - _window_mean(np.concatenate((np.full(2 * fps, 0.5),
                                                        motion_rank[:-2 * fps])), fps)),
        "fastest_player_2s": _window_mean(fastest, 2 * fps) / 12.0,
        "ball_seen_1s": _window_mean(seen, fps),
        "ball_seen_4s": _window_mean(seen, 4 * fps),
        "ball_unseen_clock": _run_clock(~seen, 10 * fps) / (10 * fps),
        "ball_still_1s": _window_mean(still, fps),
        "ball_still_clock": _run_clock(still, 5 * fps) / (5 * fps),
        "ball_slow_2s": _window_mean(slow_ball, 2 * fps),
        "ball_fast_2s": _window_mean(fast_ball, 2 * fps),
        "ball_outside_2s": _window_mean(outside, 2 * fps),
        "ball_line_dist": _window_mean(line_dist, fps) / 20.0,
        "nearest_player_1s": _window_mean(nearest, fps) / 20.0,
        "tracked_players": _window_mean(n_tracked, fps) / 22.0,
        "spread_x": _window_mean(spread_x, fps) / 30.0,
        "spread_y": _window_mean(spread_y, fps) / 20.0,
    }
    names = list(columns)
    return np.stack([columns[k] for k in names], axis=1), names


def pseudo_labels(n, restarts, tags, half_blocks, kickoff_dead, fps, calm=None, max_back_s=0):
    """Labels per frame: 1 dead, 0 alive, -1 unlabelled, from Smart Tagging alone.

    Dead labels cover the 2.5 s before each restart. With a `calm` mask and max_back_s > 0 they
    are extended further back while play stays calm, so they cover the stoppage rather than only
    its last moment.
    """
    labels = np.full(n, -1, dtype=np.int8)
    guard = np.zeros(n, dtype=bool)
    for r, _ in restarts:
        guard[max(0, r - 8 * fps):r + 3 * fps] = True

    for phase in tags["alive_phases"]:
        a, b = phase["start_frame"] + 2 * fps, phase["end_frame"] - 2 * fps
        if b > a:
            labels[a:b] = 0
    labels[guard & (labels == 0)] = -1

    for r, _ in restarts:
        start = max(0, r - int(2.5 * fps))
        if calm is not None and max_back_s > 0:
            floor = max(0, r - int(max_back_s * fps))
            while start > floor and calm[start - 1]:
                start -= 1
        labels[start:max(0, r - fps // 5)] = 1
    labels[kickoff_dead] = 1

    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_blocks:
        graded[first:last + 1] = True
    labels[~graded] = -1
    return labels


def fit(x, y, weight, l2=1e-2, steps=600, lr=0.5):
    """Weighted L2 logistic regression. Returns (mean, scale, coef, intercept)."""
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = (x - mean) / scale
    w_pos = weight * (y == 1)
    w_neg = weight * (y == 0)
    # Balance the classes so the intercept encodes an even prior; the decoder bias sets the rest.
    sample_w = w_pos / max(w_pos.sum(), 1e-9) + w_neg / max(w_neg.sum(), 1e-9)
    coef, intercept = np.zeros(z.shape[1]), 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(z @ coef + intercept)))
        g = sample_w * (p - y)
        coef -= lr * (z.T @ g + l2 * coef)
        intercept -= lr * g.sum()
    return mean, scale, coef, intercept


def logit(x, model):
    mean, scale, coef, intercept = model
    return ((x - mean) / scale) @ coef + intercept


def auc(scores, y):
    """Rank AUC of scores for y==1 against y==0."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def learned_logit(atd, ballinfo, tags, p, fps, half_blocks, length, width, gt_codes=None):
    """Per-frame dead-ball logit from a model fitted on Smart Tagging pseudo-labels, plus the
    public ground truth when gt_codes is given."""
    _, _, outside = outside_masks(ballinfo, p, fps, length, width)
    x, names = features(atd, ballinfo, outside, half_blocks, fps, length, width)
    n = len(x)
    restarts = locate_restarts(ballinfo["speed"], tags, p, fps)
    hard = hard_dead_mask(restarts, tags, half_blocks, n, fps)
    col = {name: i for i, name in enumerate(names)}
    calm = ((x[:, col["motion_rank_2s"]] < p.calm_motion_rank)
            | (x[:, col["ball_unseen_clock"]] > 0) | (x[:, col["ball_still_1s"]] > 0.5))
    y = pseudo_labels(n, restarts, tags, half_blocks, hard, fps, calm,
                      p.dead_label_back_s).astype(np.float64)
    weight = (y >= 0).astype(np.float64)
    if gt_codes is not None:
        graded = gt_codes != X_CODE
        y[graded] = (gt_codes[graded] == DEAD_CODE)
        weight[graded] = p.public_weight

    keep = weight > 0
    model = fit(x[keep], y[keep], weight[keep], l2=p.learned_l2)
    log.info("dead model: %d dead / %d alive training frames; top weights %s",
             int((y[keep] == 1).sum()), int((y[keep] == 0).sum()),
             ", ".join("{} {:+.2f}".format(names[i], model[2][i])
                       for i in np.argsort(-np.abs(model[2]))[:6]))
    scores = logit(x, model)
    # Centre on this match: the top (1 - learned_center_q) of graded frames lean dead. Class
    # balancing fixes the intercept for the training mix, not for the match's own frames.
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_blocks:
        graded[first:last + 1] = True
    return scores - np.quantile(scores[graded], p.learned_center_q)
