"""Every tunable number in the pipeline, in one place.

Nothing here is fitted to the challenge ground truth or video. The team-evidence group, the team
shape weights and cost_team were tuned on the Metrica open sample games (possession/external.py).
The dead-ball group keeps its reasoning-based values: the sample tracking loses the ball on almost
every dead frame, far more than the challenge tracking does, so dead-ball weights tuned there do
not transfer (TECHNICAL.md, section 5).
"""

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Params:
    # Ball preprocessing
    ball_max_gap: int = 12            # frames; linearly interpolate ball gaps up to 0.48 s
    interp_weight: float = 0.5        # evidence weight on interpolated ball frames
    speed_half_window: int = 2        # frames each side for the ball velocity estimate

    # Team evidence from ball proximity (metres, m/s)
    r_control: float = 0.75           # a player this close to the ball can be controlling it
    r_info: float = 2.5               # beyond this nobody is near enough to say anything
    contrast_scale: float = 0.5       # metres of nearest-distance gap that count as decisive
    v_slow: float = 4.0               # below this the ball can be controlled
    v_fast: float = 18.0              # above this the ball is in flight (pass, shot, clearance)
    fast_floor: float = 0.0           # evidence kept for a fast ball (or unknown speed) near a player
    w_team: float = 1.0               # per-frame weight of a clear, slow, close touch

    # Dead-ball evidence (log-odds dead vs alive, per frame)
    pitch_margin: float = 0.5         # metres beyond the line before the ball counts as out
    w_out: float = 2.0                # ball seen outside the pitch
    out_confirm_s: float = 1.0        # ...and not seen inside within this many seconds
    motion_window: int = 25           # frames each side for smoothing mean player speed
    dead_quantile: float = 0.30       # per-half quantile of player speed treated as the boundary
    motion_scale: float = 0.6         # m/s from that boundary for full motion evidence
    w_motion: float = 0.6             # weight of slow / fast collective movement
    v_still: float = 0.5              # ball speed below this counts as stationary
    still_min_frames: int = 38        # a stationary ball for 1.5 s suggests a set piece
    w_still: float = 0.8
    dead_bias: float = -0.3           # prior tilt towards alive; superseded by the calibration below
    unseen_min_frames: int = 75       # ball unseen for 3 s on a wide camera is usually off-pitch
    w_unseen: float = 0.3

    # Dead-ball events: ball leaves the pitch ... restart kick
    exit_line_dist: float = 2.0       # metres from a line for a "heading out" ball
    exit_speed: float = 2.0           # m/s outward
    exit_lookahead: int = 15          # frames to confirm: seen outside, or lost
    restart_still_window: int = 25    # frames before a kick that are checked for a resting ball
    restart_still_frames: int = 15    # of those, how many must show the ball still
    restart_from_out_s: float = 4.0   # ball seen outside this recently, now back in: a throw-in
    event_gap_s: float = 1.0          # merge events closer than this
    min_dead_s: float = 1.0           # shortest exit -> restart stoppage
    max_dead_s: float = 120.0         # longest confirmed-exit stoppage we pair
    max_dead_inferred_s: float = 45.0  # longest inferred-exit stoppage we pair
    unpaired_dead_s: float = 3.0      # dead stretch after an exit with no restart found
    w_interval: float = 1.0           # per-frame dead evidence between paired events

    # Learned dead-ball model (possession/deadmodel.py)
    use_learned: int = 0              # 1: add the learned logit to the hand-set per-frame terms
    learned_l2: float = 1e-2
    w_learned: float = 0.5            # logit scale: correlated frames, so damp it
    dead_bias_learned: float = 0.0
    learned_center_q: float = 0.55    # ~45% dead, LaLiga effective-play norms
    dead_label_back_s: float = 60.0   # extend dead labels back from a restart while play is calm
    calm_motion_rank: float = 0.4     # player-motion rank below which play counts as calm

    # Team shape model (possession/shape.py), second decoding pass
    use_shape: int = 1
    shape_label_min: float = 0.5      # |proximity evidence| that makes a frame a training label
    w_shape: float = 0.1              # per-frame weight where the ball says nothing
    shape_on_ball: float = 1.0        # fraction of that weight where the ball is informative
    shape_clip: float = 3.0

    # Smart Tagging
    w_phase: float = 0.08             # per-frame team nudge from phase intervals
    w_restart_dead: float = 1.5       # dead evidence before a detected set-piece restart
    w_restart_alive: float = 1.5      # alive evidence just after it
    w_restart_team: float = 0.6       # team evidence after a restart with a known taker
    restart_pre_s: float = 2.0        # look this far before the tag start for the restart
    restart_post_s: float = 2.0       # seconds after the restart that are surely alive
    v_kick: float = 5.0               # ball speed that marks the restart kick
    w_hard: float = 50.0              # pre-kickoff and goal-to-kickoff stretches

    # Decoder switching costs, in the same units as the evidence
    cost_team: float = 12.0           # H <-> A; about half a second of clear, slow, close control
    cost_dead: float = 8.0            # alive <-> D

    # Dead-ball calibration to an external statistic: the dead-ball offset is searched so the
    # graded frames carry this much ball-in-play time. 55:58 is the average effective playing
    # time of Barcelona's 2024-25 LaLiga matches (LaLiga figures as reported by El Diario Alerta,
    # 2025-05-27). 0 disables the search and uses dead_bias as is.
    effective_play_min: float = 55.97
    calibration_steps: int = 16

    # Run clean-up after decoding (decode.clean_runs)
    clean: int = 1
    min_dead_run_s: float = 1.0       # shorter D runs are not stoppages
    max_sandwich_s: float = 4.0       # shorter H/A runs between two D runs are part of the stoppage


def with_overrides(params, overrides):
    """Params with `name=value` strings applied, for sweeps from the command line."""
    known = {f.name: f.type for f in fields(Params)}
    changes = {}
    for item in overrides:
        name, _, value = item.partition("=")
        if name not in known:
            raise KeyError("unknown parameter {!r}; have {}".format(name, sorted(known)))
        changes[name] = int(value) if known[name] in (int, "int") else float(value)
    return replace(params, **changes)
