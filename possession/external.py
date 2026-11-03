"""Tune the pipeline on EXTERNAL data: Metrica Sports' open sample matches.

The rules require any tuning to use external data, never the challenge ground truth or video.
This module turns the three public Metrica sample games (github.com/metrica-sports/sample-data,
Metrica Sports, acknowledged with thanks) into the same inputs predict() takes, derives
frame-level H/A/D labels from their event data, degrades their clean tracking so it looks like
the challenge's automatic tracking, and scores parameter sets on them.

    git clone https://github.com/metrica-sports/sample-data /path/to/sample-data
    python -m possession.external --data /path/to/sample-data/data
    python -m possession.external --data ... "cost_team=8" "cost_team=16"
    python -m possession.external --data ... --search            # coordinate search

No Smart Tagging exists for these games, so tag-driven evidence is not exercised here.
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import fields

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

from possession import atd as atd_mod  # noqa: E402
from possession.config import Params, with_overrides  # noqa: E402
from possession.pipeline import predict  # noqa: E402

log = logging.getLogger("possession.external")

FPS = 25
PITCH = (105.0, 68.0)
HOME, AWAY = 0, 1

# Tracking quality, from aggregates only. In the sample games the ball is missing on 92-100% of
# dead frames and 4-12% of live ones. In the challenge tracking it is missing on 73% of the frames
# 2-10 s before Smart Tagging restarts and 17% of open-play phase cores (checked with Smart
# Tagging, no ground truth), and about 20.5 of 22 players are tracked per frame. So degradation
# adds ball gaps during live play only, and player dropouts.
BALL_GAP = (60.0, 1.0)          # lognormal (median frames, sigma), live play only
BALL_SEEN = (900.0, 1.0)
PLAYER_GAP = (100.0, 0.8)
PLAYER_SEEN = (1500.0, 0.8)
SEEDS = (11, 12)

# Events that show a team on the ball at their start frame.
ON_BALL = {"PASS", "SHOT", "SET PIECE", "BALL LOST", "CARRY", "RECOVERY"}
# Controlled possession, from the proxies in CHALLENGE.md: a team takes over once it strings
# together this many on-ball events, or keeps the ball this long.
CONTROL_EVENTS = 3
CONTROL_S = 3.0


def _speed_from_positions(positions, fps):
    speed = np.full(positions.shape[:2], np.nan, dtype=np.float32)
    step = positions[1:] - positions[:-1]
    speed[1:] = np.hypot(step[..., 0], step[..., 1]) * fps
    return speed


def load_csv_game(folder, name):
    """Sample games 1 and 2: one tracking CSV per team (players then ball), events CSV."""
    teams, ball = [], None
    for side in ("Home", "Away"):
        path = os.path.join(folder, "{}_RawTrackingData_{}_Team.csv".format(name, side))
        with open(path) as handle:
            rows = list(csv.reader(handle))
        data = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r]
                         for r in rows[3:]], dtype=np.float64)
        coords = data[:, 3:-2]
        teams.append(coords.reshape(len(data), -1, 2))
        ball = data[:, -2:]
    positions = np.concatenate(teams, axis=1)
    team = np.array([HOME] * teams[0].shape[1] + [AWAY] * teams[1].shape[1], dtype=np.int8)

    events = []
    with open(os.path.join(folder, "{}_RawEventsData.csv".format(name))) as handle:
        for row in csv.DictReader(handle):
            events.append({"team": HOME if row["Team"] == "Home" else AWAY,
                           "type": row["Type"], "subtype": row["Subtype"] or "",
                           "period": int(row["Period"]),
                           "start": int(row["Start Frame"]) - 1,
                           "end": int(row["End Frame"]) - 1})
    return positions, ball, team, events


def _parse_multi_spec_epts(meta, raw):
    """(positions (T,P,2) normalised, ball (T,2), team (P,)) for an EPTS export whose line layout
    changes between frame ranges (one DataFormatSpecification per lineup, as in sample game 3).
    The challenge file has a single layout and goes through atd.parse_layout instead."""
    import xml.etree.ElementTree as ET
    root = ET.parse(meta).getroot()
    teams = atd_mod._find_all(root, "Team")
    home_team = teams[0].get("id")
    team_of_player = {p.get("id"): p.get("teamId") for p in atd_mod._find_all(root, "Player")}
    channel_of = {pc.get("id"): (pc.get("playerId"), pc.get("channelId"))
                  for pc in atd_mod._find_all(root, "PlayerChannel")}

    specs = []
    for spec in atd_mod._find_all(root, "DataFormatSpecification"):
        slots, player_channels, ball_channels = atd_mod._spec_layout(spec, channel_of)
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots,
                      list(player_channels), list(ball_channels)))
    players = sorted({pid for _, _, slots, _, _ in specs for pid in slots})
    column = {pid: i for i, pid in enumerate(players)}

    with open(raw) as handle:
        lines = handle.readlines()
    n = len(lines)
    positions = np.full((n, len(players), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for line in lines:
        frame_part, player_part, ball_part = line.rstrip("\r\n").split(":")
        frame_id = int(frame_part)
        spec = next((sp for sp in specs if sp[0] <= frame_id <= sp[1]), None)
        if spec is None or not 1 <= frame_id <= n:
            continue
        _, _, slots, pch, bch = spec
        px, py = atd_mod._channel_index(pch, "x"), atd_mod._channel_index(pch, "y")
        row = frame_id - 1
        for pid, group in zip(slots, [g for g in player_part.split(";") if g]):
            values = group.split(",")
            if values[0] != "NaN":
                positions[row, column[pid]] = (float(values[px]), float(values[py]))
        values = ball_part.split(",")
        bx, by = atd_mod._channel_index(bch, "x"), atd_mod._channel_index(bch, "y")
        if values[bx] not in ("NaN", ""):
            ball[row] = (float(values[bx]), float(values[by]))
    team = np.array([HOME if team_of_player.get(pid) == home_team else AWAY for pid in players],
                    dtype=np.int8)
    return positions, ball, team, home_team


def load_epts_game(folder, name):
    """Sample game 3: FIFA EPTS tracking (the challenge's format) and JSON events."""
    meta = os.path.join(folder, "{}_metadata.xml".format(name))
    raw = os.path.join(folder, "{}_tracking.txt".format(name))
    positions, ball, team, home_id = _parse_multi_spec_epts(meta, raw)

    with open(os.path.join(folder, "{}_events.json".format(name))) as handle:
        raw_events = json.load(handle)["data"]
    events = [{"team": HOME if e["team"]["id"] == home_id else AWAY,
               "type": e["type"]["name"],
               "subtype": (e.get("subtypes") or {}).get("name", "")
               if isinstance(e.get("subtypes"), dict) else "",
               "period": int(e["period"]),
               "start": int(e["start"]["frame"]) - 1,
               "end": int(e["end"]["frame"] or e["start"]["frame"]) - 1}
              for e in raw_events]
    return positions, ball, team, events


def _controlled(run, min_events, min_s):
    """Whether a team's run of consecutive on-ball events shows controlled possession."""
    if run[0]["type"] == "SET PIECE":
        return True
    held = max(max(e["start"], e["end"]) for e in run) - run[0]["start"]
    return len(run) >= min_events or held >= min_s * FPS


def labels_from_events(events, n_frames, min_events=CONTROL_EVENTS, min_s=CONTROL_S):
    """(labels, blocks): H/A/D per frame from events, X outside each period's events.

    The ball is dead from a ball-out, a foul or a goal until the next on-ball event (the restart),
    which always sets the team. Otherwise a team takes over at the first of a run of its on-ball
    events only if the run shows control (CHALLENGE.md: a single touch, deflection or won duel does
    not change possession); until then possession stays with the last team that had it.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    blocks = []
    for period in sorted({e["period"] for e in events}):
        period_events = sorted((e for e in events if e["period"] == period),
                               key=lambda e: e["start"])
        marks = []
        for e in period_events:
            sub = e["subtype"].upper()
            if e["type"] in ON_BALL:
                marks.append((e["start"], "H" if e["team"] == HOME else "A", e))
            if e["type"] == "BALL OUT":
                marks.append((max(e["start"], e["end"]), "D", None))
            elif e["type"] == "FAULT RECEIVED" or (e["type"] == "CHALLENGE" and "FAULT" in sub):
                marks.append((e["start"], "D", None))
            elif e["type"] == "SHOT" and ("GOAL" in sub or "OUT" in sub):
                marks.append((max(e["start"], e["end"]), "D", None))
        marks.sort(key=lambda m: m[0])

        changes, state, i = [], None, 0
        while i < len(marks):
            frame, label, event = marks[i]
            j = i + 1
            if label != "D":
                while j < len(marks) and marks[j][1] == label:
                    j += 1
            if label == "D" or state in (None, "D") or (
                    label != state and _controlled([m[2] for m in marks[i:j]], min_events, min_s)):
                if label != state:
                    changes.append((frame, label))
                    state = label
            i = j
        first = changes[0][0]
        last = min(n_frames - 1, max(max(e["start"], e["end"]) for e in period_events))
        for (frame, label), nxt in zip(changes, changes[1:] + [(last + 1, None)]):
            labels[frame:min(nxt[0], last + 1)] = label
        blocks.append((period, first, last))
    return labels, blocks


def _renewal_mask(n, seen, gap, rng):
    """True where missing: alternating lognormal visible / missing stretches."""
    mask = np.zeros(n, dtype=bool)
    t = int(rng.lognormal(np.log(seen[0]), seen[1]) * rng.random())
    while t < n:
        length = max(1, int(rng.lognormal(np.log(gap[0]), gap[1])))
        mask[t:t + length] = True
        t += length + max(1, int(rng.lognormal(np.log(seen[0]), seen[1])))
    return mask


def degrade(positions, ball, labels, seed, ball_seen=BALL_SEEN):
    """Copies with extra ball gaps during live play and player dropouts, like the challenge
    tracking. A shorter `ball_seen` median stresses the missing-ball terms."""
    rng = np.random.default_rng(seed)
    positions, ball = positions.copy(), ball.copy()
    live = np.isin(labels, ("H", "A"))
    ball[_renewal_mask(len(ball), ball_seen, BALL_GAP, rng) & live] = np.nan
    for slot in range(positions.shape[1]):
        positions[_renewal_mask(len(positions), PLAYER_SEEN, PLAYER_GAP, rng), slot] = np.nan
    return positions, ball


def to_atd(positions, ball, team):
    scale = np.array(PITCH)
    pos_m = (positions * scale).astype(np.float32)
    return {"positions": pos_m, "speed": _speed_from_positions(pos_m, FPS), "team": team,
            "ball_xy": (ball * scale).astype(np.float32)}


def load_games(data_dir):
    games = []
    for name, loader in (("Sample_Game_1", load_csv_game), ("Sample_Game_2", load_csv_game),
                         ("Sample_Game_3", load_epts_game)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            log.warning("missing %s", folder)
            continue
        positions, ball, team, events = loader(folder, name)
        labels, blocks = labels_from_events(events, len(ball))
        games.append({"name": name, "positions": positions, "ball": ball, "team": team,
                      "labels": labels, "blocks": blocks})
        graded = labels != "X"
        log.info("%s: %d frames graded, dead %.2f, %d possessions", name, int(graded.sum()),
                 float(np.mean(labels[graded] == "D")),
                 len(pm.possession_spells(pm.label_codes(labels), pm.valid_runs(
                     pm.label_codes(labels)))))
    return games


def prepare(games, clean=False, ball_seen=BALL_SEEN):
    """[(name, atd, context, gt codes, effective play minutes)] per game and degradation seed."""
    out = []
    for game in games:
        codes = pm.label_codes(game["labels"])
        context = {"blocks": game["blocks"], "n_frames": len(game["ball"]), "fps": FPS,
                   "pitch": PITCH}
        alive_min = float(np.isin(game["labels"], ("H", "A")).sum()) / FPS / 60
        for seed in ((None,) if clean else SEEDS):
            positions, ball = ((game["positions"], game["ball"]) if seed is None
                               else degrade(game["positions"], game["ball"], game["labels"],
                                            seed, ball_seen))
            out.append((game["name"], to_atd(positions, ball, game["team"]), context, codes,
                        alive_min))
    return out


def score(prepared, params):
    """Mean spellF1 and macroF1 over the prepared games, with per-game spellF1."""
    spells, macros, per_game = [], [], []
    for name, atd, context, codes, alive_min in prepared:
        # Calibrate to this game's own effective playing time, as the challenge pipeline does
        # with the league figure.
        p = with_overrides(params, ["effective_play_min={}".format(alive_min)])
        labels = predict(atd, None, p, context=context)
        result = pm.evaluate(codes, pm.label_codes(labels))
        spells.append(result["spell_f1_mean"])
        macros.append(result["macro_f1"])
        per_game.append("{} {:.3f}".format(name[-1], result["spell_f1_mean"]))
    return float(np.mean(spells)), float(np.mean(macros)), per_game


# Parameters searched by --search, with candidate values around the reasoning-based defaults.
SEARCH = {
    "cost_team": [2, 4, 6, 8, 12, 16, 24, 32],
    "cost_dead": [2, 4, 6, 8, 12, 16],
    "r_control": [0.75, 1.0, 1.5],
    "r_info": [2.0, 2.5, 3.0, 4.0],
    "contrast_scale": [0.5, 0.75, 1.0, 1.5],
    "v_fast": [12.0, 18.0, 25.0, 40.0],
    "fast_floor": [0.0, 0.15],
    "use_shape": [0, 1],
    "w_shape": [0.02, 0.035, 0.05, 0.1],
    "shape_on_ball": [0.5, 1.0, 1.5, 2.0],
    "max_sandwich_s": [0.0, 1.0, 2.0, 4.0],
    "w_motion": [0.0, 0.15, 0.3, 0.6],
    "w_still": [0.0, 0.4],
    "w_out": [0.0, 0.5, 1.0, 2.0],
    "w_unseen": [0.8, 1.5, 3.0, 5.0],
    "unseen_min_frames": [12, 25, 50, 75],
}


def search(prepared, params, rounds=2):
    """Coordinate search on mean spellF1; ties broken by macroF1."""
    best = score(prepared, params)
    print("start {:.4f} / {:.4f}".format(best[0], best[1]))
    for _ in range(rounds):
        improved = False
        for name, values in SEARCH.items():
            for value in values:
                trial = with_overrides(params, ["{}={}".format(name, value)])
                result = score(prepared, trial)
                if result[0] > best[0] + 1e-4:
                    params, best, improved = trial, result, True
                    print("  {}={:<6} spell {:.4f} macro {:.4f}  [{}]".format(
                        name, value, result[0], result[1], ", ".join(result[2])))
        if not improved:
            break
    defaults = Params()
    changed = {f.name: getattr(params, f.name) for f in fields(Params)
               if getattr(params, f.name) != getattr(defaults, f.name)}
    print("best spell {:.4f} macro {:.4f}; changed: {}".format(best[0], best[1], changed))
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="sample-data/data folder")
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--clean", action="store_true", help="score on undegraded tracking")
    parser.add_argument("--ball-seen", type=float, default=BALL_SEEN[0],
                        help="median live frames between ball gaps; lower is harsher")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    for name in ("possession", "possession.atd", "possession.evidence", "possession.shape",
                 "possession.deadmodel"):
        logging.getLogger(name).setLevel(logging.WARNING)

    prepared = prepare(load_games(args.data), clean=args.clean,
                       ball_seen=(args.ball_seen, BALL_SEEN[1]))
    if args.search:
        search(prepared, Params())
        return
    for variant in ["default"] + args.variants:
        overrides = [] if variant == "default" else variant.split()
        spell, macro, per_game = score(prepared, with_overrides(Params(), overrides))
        print("{:<32} spell {:.4f} macro {:.4f}  [{}]".format(variant, spell, macro,
                                                             ", ".join(per_game)))


if __name__ == "__main__":
    main()
