"""Smart Tagging intervals turned into the few things the decoder uses: phase-team hints,
set-piece windows with their taker, kickoffs and goals."""

import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME  # noqa: E402
from load_smart_tagging import load as load_events  # noqa: E402

log = logging.getLogger(__name__)

HOME_SIGN, AWAY_SIGN = 1, -1

# Phases performed by the team with the ball, and by the team without it. The Team tag names
# the team performing the phase, so a defending phase points at the other side.
IN_POSSESSION = {"build up", "progression", "creating chances", "attacking transition",
                 "ball in final third", "ball in the box", "long ball"}
OUT_OF_POSSESSION = {"defending in attacking third", "defending in middle third",
                     "defending in defensive third", "defensive transition"}
# Open-play phases whose core is certainly ball-in-play: training labels for the dead-ball model.
ALIVE_PHASES = {"build up", "progression", "creating chances", "attacking transition"}
SET_PIECES = "set pieces"
GOALS = "goals"
KICKOFF_START = "kick off (start)"
KICKOFF_AFTER_GOAL = "kick off (after goal)"


def team_sign(team_tag):
    """+1 for Barcelona (home), -1 for Athletic (away), 0 for N/A or anything unknown."""
    tag = (team_tag or "").strip().lower()
    if "barcelona" in tag:
        return HOME_SIGN
    if "athletic" in tag or "atletic" in tag:
        return AWAY_SIGN
    return 0


def load(game=GAME, data_dir=None):
    """Events grouped by role, each a dict with start_frame, end_frame (exclusive) and sign."""
    events = load_events(game, data_dir)
    phases, set_pieces, goals, alive_phases = [], [], [], []
    for event in events:
        code = (event["code"] or "").strip().lower()
        sign = team_sign(event["tags"].get("Team"))
        item = {"start_frame": event["start_frame"], "end_frame": event["end_frame"],
                "sign": sign, "type": (event["tags"].get("Type") or "").strip().lower()}
        if code in ALIVE_PHASES:
            alive_phases.append(item)
        if code in IN_POSSESSION and sign:
            phases.append(dict(item, sign=sign))
        elif code in OUT_OF_POSSESSION and sign:
            phases.append(dict(item, sign=-sign))
        elif code == SET_PIECES:
            set_pieces.append(item)
        elif code == GOALS:
            goals.append(item)
    log.info("smart tagging: %d phase hints, %d set pieces, %d goals",
             len(phases), len(set_pieces), len(goals))
    return {"phases": phases, "set_pieces": set_pieces, "goals": goals,
            "alive_phases": alive_phases}
