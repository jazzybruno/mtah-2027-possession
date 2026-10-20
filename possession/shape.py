"""Team evidence from the shape of both teams, for the frames where the ball says nothing.

Self-trained: the labels are our own first-pass decode on frames where the ball is visible and
the proximity evidence is clear. The model then speaks where the ball is missing. No ground
truth of any kind is involved.
"""

import logging
import warnings

import numpy as np

from possession.deadmodel import fit, logit

log = logging.getLogger(__name__)

HOME = 0


def _team_stats(pos, mask):
    """Per-frame nan-aware mean, std and quantiles of x and y for the slots in mask."""
    team = pos[:, mask, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no players of this team
        return {
            "mx": np.nanmean(team[:, :, 0], axis=1), "my": np.nanmean(team[:, :, 1], axis=1),
            "sx": np.nanstd(team[:, :, 0], axis=1), "sy": np.nanstd(team[:, :, 1], axis=1),
            "qlo": np.nanquantile(team[:, :, 0], 0.1, axis=1),
            "qhi": np.nanquantile(team[:, :, 0], 0.9, axis=1),
            "n": np.isfinite(team[:, :, 0]).sum(axis=1),
        }


def home_attack_direction(pos, team, kickoffs, half_blocks, fps):
    """{half: +1 if home attacks towards x = length, else -1}, read off the pre-kickoff lineup.

    Before a kickoff every player stands in their own half, so the team with the smaller mean x
    defends the x = 0 goal. Falls back to where each team's deepest player sits over the half.
    """
    home = team == HOME
    directions = {}
    for half, first, last in half_blocks:
        ks = [k for k in kickoffs if first - 10 * fps <= k <= last]
        sign = 0.0
        if ks:
            frames = slice(max(0, ks[0] - 2 * fps), ks[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sign = np.nanmean(pos[frames][:, ~home, 0]) - np.nanmean(pos[frames][:, home, 0])
        if not np.isfinite(sign) or sign == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                block = pos[first:last + 1]
                sign = (np.nanmedian(np.nanmin(block[:, ~home, 0], axis=1))
                        - np.nanmedian(np.nanmin(block[:, home, 0], axis=1)))
                sign = -sign  # the team whose deepest player sits at low x defends x = 0
            log.warning("half %d: no kickoff lineup, attack direction from deepest players", half)
        directions[half] = 1 if sign > 0 else -1
    log.info("home attacks towards %s", {h: "x=L" if d > 0 else "x=0"
                                         for h, d in directions.items()})
    return directions


def features(atd, ball, directions, half_blocks, length, fps):
    """(T, F) shape features, oriented so +x is home's attacking direction."""
    pos = atd["positions"].astype(np.float64).copy()
    n = len(pos)
    orient = np.zeros(n)
    for half, first, last in half_blocks:
        orient[first:last + 1] = directions[half]
    flip = orient < 0
    pos[flip, :, 0] = length - pos[flip, :, 0]

    home = atd["team"] == HOME
    h, a = _team_stats(pos, home), _team_stats(pos, ~home)

    seen = ~np.isnan(ball).any(axis=1)
    last_x = ball[:, 0].copy()
    last_x[flip] = length - last_x[flip]
    idx = np.where(seen, np.arange(n), -1)
    carry = np.maximum.accumulate(idx)
    last_x = np.where(carry >= 0, last_x[np.maximum(carry, 0)], length / 2)
    since = np.where(carry >= 0, np.arange(n) - carry, 10 * fps)

    columns = {
        "centroid_x": (h["mx"] + a["mx"]) / 2 / length - 0.5,
        "home_minus_away_x": (h["mx"] - a["mx"]) / 10.0,
        "home_back_line": h["qlo"] / length - 0.5,
        "away_back_line": a["qhi"] / length - 0.5,
        "home_front": h["qhi"] / length - 0.5,
        "away_front": a["qlo"] / length - 0.5,
        "spread_x_diff": (h["sx"] - a["sx"]) / 10.0,
        "spread_y_diff": (h["sy"] - a["sy"]) / 10.0,
        "last_ball_x": last_x / length - 0.5,
        "ball_staleness": np.minimum(since, 10 * fps) / (10 * fps),
    }
    names = list(columns)
    x = np.nan_to_num(np.stack([columns[k] for k in names], axis=1))
    return x, names


def shape_logit(atd, ball, team_score, labels, kickoffs, p, half_blocks, length, fps):
    """Home-vs-away logit per frame from team shape, trained on confident first-pass frames."""
    directions = home_attack_direction(atd["positions"], atd["team"], kickoffs, half_blocks, fps)
    x, names = features(atd, ball, directions, half_blocks, length, fps)
    confident = (np.abs(team_score) >= p.shape_label_min) & np.isin(labels, ("H", "A"))
    y = (labels[confident] == "H").astype(np.float64)
    model = fit(x[confident], y, np.ones(len(y)), l2=p.learned_l2)
    log.info("shape model: %d training frames (%.0f%% home); top weights %s", len(y),
             100 * y.mean() if len(y) else 0,
             ", ".join("{} {:+.2f}".format(names[i], model[2][i])
                       for i in np.argsort(-np.abs(model[2]))[:5]))
    return logit(x, model)
