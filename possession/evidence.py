"""Per-frame evidence for the decoder: team_score (>0 favours home) and dead_score (>0 favours
dead ball), both in log-odds-like units the switching costs are expressed in."""

import logging
import warnings

import numpy as np

log = logging.getLogger(__name__)

HOME = 0


def interpolate_ball(xy, max_gap):
    """(filled xy, interpolated mask). Linear fill of NaN gaps no longer than max_gap frames."""
    seen = ~np.isnan(xy).any(axis=1)
    out = xy.astype(np.float64).copy()
    interpolated = np.zeros(len(xy), dtype=bool)
    idx = np.flatnonzero(seen)
    gaps = np.diff(idx)
    for i in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        a, b = idx[i], idx[i + 1]
        t = ((np.arange(a + 1, b) - a) / (b - a))[:, None]
        out[a + 1:b] = out[a] + t * (out[b] - out[a])
        interpolated[a + 1:b] = True
    return out, interpolated


def ball_speed(xy, half_window, fps):
    """Central-difference speed in m/s; NaN where either end is unknown."""
    k = half_window
    speed = np.full(len(xy), np.nan)
    if len(xy) > 2 * k:
        delta = xy[2 * k:] - xy[:-2 * k]
        speed[k:-k] = np.hypot(delta[:, 0], delta[:, 1]) * fps / (2 * k)
    return speed


def moving_nanmean(values, half_window):
    """Centred mean over +/- half_window frames, ignoring NaN; NaN where the window is empty."""
    finite = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(values), index + half_window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def long_runs(mask, min_length):
    """mask restricted to runs of True at least min_length long."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(mask), dtype=bool)
    for start, stop in zip(starts, stops):
        if stop - start >= min_length:
            out[start:stop] = True
    return out


def player_motion(atd, fps):
    """Mean speed of tracked players per frame, from the tracker or from positions if absent."""
    speed = atd["speed"].astype(np.float64)
    if np.isfinite(speed).mean() < 0.05:
        log.warning("tracker speed channel empty; deriving player speed from positions")
        pos = atd["positions"].astype(np.float64)
        step = np.full(speed.shape, np.nan)
        step[1:] = np.hypot(*(pos[1:] - pos[:-1]).transpose(2, 0, 1)) * fps
        speed = step
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames: no players tracked
        return np.nanmean(speed, axis=1)


def team_evidence(atd, p, fps):
    """team_score from who is nearest to the ball, discounted for distance, ball speed and
    interpolated ball positions. Also returns the ball arrays the dead-ball evidence reuses."""
    ball, interpolated = interpolate_ball(atd["ball_xy"], p.ball_max_gap)
    speed = ball_speed(ball, p.speed_half_window, fps)

    pos = atd["positions"].astype(np.float64)
    dist = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    home = atd["team"] == HOME
    d_home = dist[:, home].min(axis=1) if home.any() else np.full(len(dist), np.inf)
    d_away = dist[:, ~home].min(axis=1) if (~home).any() else np.full(len(dist), np.inf)
    d_min = np.minimum(d_home, d_away)

    informative = np.isfinite(d_min) & (d_min < p.r_info)
    with np.errstate(invalid="ignore"):
        contrast = np.clip((d_away - d_home) / p.contrast_scale, -1.0, 1.0)
    closeness = np.clip((p.r_info - d_min) / (p.r_info - p.r_control), 0.0, 1.0)
    slow = np.clip((p.v_fast - speed) / (p.v_fast - p.v_slow), p.fast_floor, 1.0)
    slow = np.where(np.isnan(slow), p.fast_floor, slow)
    weight = np.where(interpolated, p.interp_weight, 1.0)

    score = np.where(informative,
                     p.w_team * np.nan_to_num(contrast) * closeness * slow * weight, 0.0)
    return score, {"ball": ball, "interpolated": interpolated, "speed": speed}


def outside_masks(ballinfo, p, fps, length, width):
    """(observed, raw_outside, outside): ball seen (not interpolated), seen beyond a line, and
    seen beyond a line without being seen inside within p.out_confirm_s."""
    ball, interpolated = ballinfo["ball"], ballinfo["interpolated"]
    observed = ~np.isnan(ball).any(axis=1) & ~interpolated
    m = p.pitch_margin
    with np.errstate(invalid="ignore"):
        raw_outside = observed & ((ball[:, 0] < -m) | (ball[:, 0] > length + m)
                                  | (ball[:, 1] < -m) | (ball[:, 1] > width + m))
    # A high ball projected onto the ground can look outside for a few frames; a real exit keeps
    # the ball out of the pitch (or out of view) for longer.
    outside = raw_outside & ~_seen_within(observed & ~raw_outside, int(p.out_confirm_s * fps))
    return observed, raw_outside, outside


def dead_evidence(atd, ballinfo, p, fps, half_blocks, length, width, learned=None):
    """dead_score: hand-set per-frame terms (ball leaving the pitch, collective slowness, a
    stationary or unseen ball), plus the learned logit if given, plus exit -> restart events."""
    ball, speed = ballinfo["ball"], ballinfo["speed"]
    observed, raw_outside, outside = outside_masks(ballinfo, p, fps, length, width)

    motion = moving_nanmean(player_motion(atd, fps), p.motion_window)
    slowness = np.zeros(len(motion))
    for _, first, last in half_blocks:
        window = motion[first:last + 1]
        if np.isfinite(window).any():
            boundary = np.nanquantile(window, p.dead_quantile)
            slowness[first:last + 1] = np.clip((boundary - window) / p.motion_scale, -1.0, 1.0)
    slowness = np.nan_to_num(slowness)

    with np.errstate(invalid="ignore"):
        still = long_runs(observed & (speed < p.v_still), p.still_min_frames)
    unseen = long_runs(np.isnan(ball).any(axis=1), p.unseen_min_frames)

    score = (p.dead_bias + p.w_out * outside + p.w_motion * slowness + p.w_still * still
             + p.w_unseen * unseen)
    if learned is not None:
        # Blended, not substituted: the two rank dead frames better together than apart.
        score = score + p.dead_bias_learned + p.w_learned * learned

    confirmed, inferred = exit_events(ball, observed, outside, p, fps, length, width)
    restarts = restart_events(speed, observed, outside, raw_outside, p, fps)
    post = int(p.restart_post_s * fps)
    for t in restarts:
        score[t:t + post] -= p.w_restart_alive

    # A confirmed exit (ball seen outside) is trusted up to max_dead_s. An inferred one (ball lost
    # heading over a line) must also show the players slowing down before it is believed.
    counts = {"confirmed": 0, "inferred": 0}
    for exits, limit_s, kind in ((confirmed, p.max_dead_s, "confirmed"),
                                 (inferred, p.max_dead_inferred_s, "inferred")):
        for e in exits:
            r = _next_after(restarts, e + int(p.min_dead_s * fps))
            if r is None or r - e > int(limit_s * fps):
                continue
            if kind == "inferred" and slowness[e:r].mean() <= 0:
                continue
            score[e:r] += p.w_interval
            counts[kind] += 1
    log.info("dead-ball events: %d confirmed + %d inferred exits, %d restarts; paired %d + %d",
             len(confirmed), len(inferred), len(restarts), counts["confirmed"],
             counts["inferred"])
    return score


def _seen_within(mask, k):
    """True where mask is True anywhere in [t - k, t + k]."""
    total = np.concatenate(([0], np.cumsum(mask)))
    idx = np.arange(len(mask))
    return (total[np.minimum(idx + k + 1, len(mask))] - total[np.maximum(idx - k, 0)]) > 0


def _next_after(sorted_frames, t):
    """First element of sorted_frames >= t, or None."""
    i = np.searchsorted(sorted_frames, t)
    return int(sorted_frames[i]) if i < len(sorted_frames) else None


def exit_events(ball, observed, outside, p, fps, length, width):
    """(confirmed, inferred) frames where the ball leaves the pitch. Confirmed: seen outside.
    Inferred: seen heading over a nearby line and then lost. One event per p.event_gap_s."""
    n = len(ball)
    velocity = np.full_like(ball, np.nan)
    k = p.speed_half_window
    velocity[k:-k] = (ball[2 * k:] - ball[:-2 * k]) * fps / (2 * k)

    x, y = ball[:, 0], ball[:, 1]
    vx, vy = velocity[:, 0], velocity[:, 1]
    with np.errstate(invalid="ignore"):
        # Outward speed towards the nearest line, positive when heading out.
        to_line = np.stack([x, length - x, y, width - y], axis=1)
        outward = np.stack([-vx, vx, -vy, vy], axis=1)
        nearest = np.nanargmin(np.where(np.isnan(to_line), np.inf, to_line), axis=1)
        dist = to_line[np.arange(n), nearest]
        out_speed = outward[np.arange(n), nearest]
        heading_out = observed & ~outside & (dist < p.exit_line_dist) & (out_speed > p.exit_speed)

    seen = observed.astype(np.int32)
    seen_ahead = np.concatenate((np.cumsum(seen[::-1])[::-1], [0]))
    look = p.exit_lookahead
    idx = np.arange(n)
    lost_after = (seen_ahead[np.minimum(idx + 1, n)] - seen_ahead[np.minimum(idx + 1 + look, n)]) == 0

    outside_soon = np.zeros(n, dtype=bool)
    outside_idx = np.flatnonzero(outside)
    j = np.searchsorted(outside_idx, idx + 1)
    has = j < len(outside_idx)
    outside_soon[has] = outside_idx[j[has]] <= idx[has] + look

    became_outside = np.zeros(n, dtype=bool)
    became_outside[1:] = outside[1:] & ~outside[:-1]
    gap = int(p.event_gap_s * fps)
    confirmed = _dedupe(np.flatnonzero((heading_out & outside_soon) | became_outside), gap)
    inferred = _dedupe(np.flatnonzero(heading_out & lost_after & ~outside_soon), gap)
    # An inferred exit within a second of a confirmed one is the same event.
    if len(confirmed) and len(inferred):
        j = np.clip(np.searchsorted(confirmed, inferred), 1, len(confirmed)) - 1
        near = np.abs(inferred - confirmed[j]) < gap
        j2 = np.minimum(j + 1, len(confirmed) - 1)
        near |= np.abs(inferred - confirmed[j2]) < gap
        inferred = inferred[~near]
    return confirmed, inferred


def restart_events(speed, observed, outside, raw_outside, p, fps):
    """Frames where the ball is kicked after resting still, or comes back in from outside.

    `outside` is the confirmed (sustained) mask and decides that the ball really was out;
    `raw_outside` is the per-frame one and locates the moment it comes back in.
    """
    n = len(speed)
    with np.errstate(invalid="ignore"):
        still = (observed & (speed < p.v_still)).astype(np.int32)
        kick = observed & ~outside & (speed > p.v_kick)
    window = p.restart_still_window
    still_sum = np.concatenate(([0], np.cumsum(still)))
    idx = np.arange(n)
    still_before = still_sum[idx] - still_sum[np.maximum(idx - window, 0)]
    from_rest = kick & (still_before >= p.restart_still_frames)

    outside_sum = np.concatenate(([0], np.cumsum(outside)))
    recent = int(p.restart_from_out_s * fps)
    was_outside = (outside_sum[idx] - outside_sum[np.maximum(idx - recent, 0)]) > 0
    # Last observed frame before each frame: was the ball outside there?
    last_seen = np.maximum.accumulate(np.where(observed, idx, -1))
    prev_seen = np.concatenate(([-1], last_seen[:-1]))
    prev_outside = (prev_seen >= 0) & raw_outside[np.maximum(prev_seen, 0)]
    back_in = observed & ~raw_outside & was_outside & prev_outside

    return _dedupe(np.flatnonzero(from_rest | back_in), int(p.event_gap_s * fps))


def _dedupe(frames, gap):
    """Keep the first frame of every cluster closer together than gap."""
    kept = []
    for f in frames:
        if not kept or f - kept[-1] >= gap:
            kept.append(int(f))
    return np.array(kept, dtype=np.int64)


def detect_restart(speed, first, last, v_kick, settle=12, settle_speed=2.0):
    """First frame in [first, last) where the ball is kicked after being near-still or unseen.

    None if no such frame; the caller decides the fallback.
    """
    first, last = max(first, settle), min(last, len(speed))
    calm = np.nan_to_num(speed, nan=0.0)
    for t in range(first, last):
        if speed[t] > v_kick and calm[t - settle:t].min() < settle_speed:
            return t
    return None


def locate_restarts(speed, tags, p, fps):
    """[(restart frame, set piece)] for every tagged set piece, the kick found in its window."""
    n, pre = len(speed), int(p.restart_pre_s * fps)
    restarts = []
    for piece in tags["set_pieces"]:
        lo, hi = max(0, piece["start_frame"] - pre), min(n, piece["end_frame"])
        restart = detect_restart(speed, lo, hi, p.v_kick)
        if restart is None:
            restart = (piece["start_frame"] + piece["end_frame"]) // 2
        restarts.append((restart, piece))
    return restarts


def hard_dead_mask(restarts, tags, half_blocks, n, fps):
    """Frames that are certainly dead: a half before its kickoff, a goal until the restart."""
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_blocks:
        kicks = [r for r, piece in restarts
                 if piece["type"] == "kick off (start)" and first - fps * 10 <= r <= last]
        if kicks:
            mask[first:min(kicks)] = True
    after_goal = sorted(r for r, piece in restarts if piece["type"] == "kick off (after goal)")
    for goal in tags["goals"]:
        nxt = [r for r in after_goal if r > goal["end_frame"]]
        if nxt and nxt[0] - goal["end_frame"] < fps * 180:
            mask[goal["end_frame"]:nxt[0]] = True
    return mask


def apply_tags(team_score, dead_score, speed, tags, p, fps, half_blocks):
    """Add Smart Tagging evidence in place: phase hints, set-piece restarts, kickoffs, goals."""
    n = len(team_score)
    for phase in tags["phases"]:
        team_score[max(0, phase["start_frame"]):min(n, phase["end_frame"])] += (
            p.w_phase * phase["sign"])

    pre, post = int(p.restart_pre_s * fps), int(p.restart_post_s * fps)
    restarts = locate_restarts(speed, tags, p, fps)
    for restart, piece in restarts:
        dead_score[max(0, piece["start_frame"] - pre):restart] += p.w_restart_dead
        dead_score[restart:restart + post] -= p.w_restart_alive
        if piece["sign"]:
            team_score[restart:restart + post] += p.w_restart_team * piece["sign"]

    dead_score[hard_dead_mask(restarts, tags, half_blocks, n, fps)] += p.w_hard
    for goal in tags["goals"]:
        midpoint = (goal["start_frame"] + goal["end_frame"]) // 2
        dead_score[midpoint:goal["end_frame"]] += p.w_restart_dead
    log.info("tags applied: %d restarts located", len(restarts))
