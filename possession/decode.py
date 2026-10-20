"""Segment the match into H / A / D runs with a three-state Viterbi and switching costs.

Each frame contributes evidence; changing state costs a fixed amount. A possession therefore
changes hands only when the opponent's accumulated evidence outweighs the switching cost, which
is a direct translation of "a touch is not control, sustained control is".
"""

import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


def frame_scores(team_score, dead_score):
    """(T,3) per-state scores, higher is better."""
    return np.stack([team_score / 2.0, -team_score / 2.0, dead_score], axis=1)


def viterbi(scores, cost_team, cost_dead):
    """Best state path maximising sum(scores) - switching costs. scores is (T,3).

    Plain Python floats in the loop: three states make numpy's per-call overhead dominate.
    """
    n = len(scores)
    rows = scores.tolist()
    back = bytearray(3 * n)
    h, a, d = rows[0]
    for t in range(1, n):
        sh, sa, sd = rows[t]
        # to H
        if h >= a - cost_team and h >= d - cost_dead:
            nh, bh = h, H
        elif a - cost_team >= d - cost_dead:
            nh, bh = a - cost_team, A
        else:
            nh, bh = d - cost_dead, D
        # to A
        if a >= h - cost_team and a >= d - cost_dead:
            na, ba = a, A
        elif h - cost_team >= d - cost_dead:
            na, ba = h - cost_team, H
        else:
            na, ba = d - cost_dead, D
        # to D
        if d >= h - cost_dead and d >= a - cost_dead:
            nd, bd = d, D
        elif h >= a:
            nd, bd = h - cost_dead, H
        else:
            nd, bd = a - cost_dead, A
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = bh, ba, bd
        h, a, d = nh + sh, na + sa, nd + sd

    path = np.empty(n, dtype=np.int8)
    state = max(range(3), key=lambda k: (h, a, d)[k])
    path[-1] = state
    for t in range(n - 1, 0, -1):
        state = back[3 * t + state]
        path[t - 1] = state
    return path


def decode(team_score, dead_score, half_blocks, n_frames, cost_team, cost_dead):
    """Full-length label array: H/A/D inside each graded half, X elsewhere."""
    labels = np.full(n_frames, "X", dtype="<U1")
    scores = frame_scores(team_score, dead_score)
    for _, first, last in half_blocks:
        path = viterbi(scores[first:last + 1], cost_team, cost_dead)
        labels[first:last + 1] = LABELS[path]
    return labels
