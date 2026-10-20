"""Read the FIFA/EPTS tracking export into dense numpy arrays, cached as .npz.

The layout of every line is taken from the metadata XML rather than assumed, so a ball z or
owner channel, if the export has one, is picked up by name instead of being silently dropped.
"""

import logging
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

log = logging.getLogger(__name__)

HOME, AWAY = 0, 1
CACHE_DIR = os.path.join(ROOT, "cache")


def _local(tag):
    """Tag name without an XML namespace."""
    return tag.rsplit("}", 1)[-1]


def _find_all(root, name):
    return [el for el in root.iter() if _local(el.tag) == name]


def parse_layout(meta_path):
    """Line layout and team of every track slot, from the metadata XML.

    Returns (team_of_slot int8 array, player_channels, ball_channels): the channel lists name the
    comma-separated fields of one player group and of the ball group, in file order.
    """
    root = ET.parse(meta_path).getroot()

    teams = _find_all(root, "Team")
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_team_id = teams[0].get("id")
    team_of_player = {p.get("id"): p.get("teamId") for p in _find_all(root, "Player")}

    channel_of_player_channel = {pc.get("id"): (pc.get("playerId"), pc.get("channelId"))
                                 for pc in _find_all(root, "PlayerChannel")}

    specs = _find_all(root, "DataFormatSpecification")
    if not specs:
        raise ValueError("no DataFormatSpecification in {}".format(meta_path))
    layouts = {_spec_layout(spec, channel_of_player_channel) for spec in specs}
    if len(layouts) != 1:
        raise ValueError("{} data format specifications disagree; not supported".format(
            len(layouts)))
    slot_players, player_channels, ball_channels = layouts.pop()
    if None in slot_players:
        # No PlayerChannel mapping: fall back to the starter baseline's convention, slot i is
        # the i-th Player element.
        players_in_order = [p.get("id") for p in _find_all(root, "Player")]
        if len(players_in_order) != len(slot_players):
            raise ValueError("cannot map {} slots to {} players".format(
                len(slot_players), len(players_in_order)))
        log.warning("slot->player mapping missing in the spec; using Player element order")
        slot_players = tuple(players_in_order)

    team_of_slot = np.array([HOME if team_of_player.get(pid) == home_team_id else AWAY
                             for pid in slot_players], dtype=np.int8)
    names = [t.findtext("Name") or t.findtext("{*}Name") for t in teams]
    log.info("teams: H = %s, A = %s; %d slots (%d home); player fields %s; ball fields %s",
             names[0], names[1], len(slot_players), int((team_of_slot == HOME).sum()),
             list(player_channels), list(ball_channels))
    return team_of_slot, list(player_channels), list(ball_channels)


def _spec_layout(spec, channel_of_player_channel):
    """(slot player ids, player channel names, ball channel names) as a hashable tuple."""
    slot_players, player_channels, ball_channels = [], None, []
    for register in spec:
        if _local(register.tag) != "SplitRegister":
            continue
        groups = [g for g in register if _local(g.tag) == "SplitRegister"]
        if groups:
            for group in groups:
                refs = [r.get("playerChannelId") for r in group
                        if _local(r.tag) == "PlayerChannelRef"]
                pairs = [channel_of_player_channel.get(ref, (None, ref)) for ref in refs]
                channels = tuple(ch for _, ch in pairs)
                if player_channels is None:
                    player_channels = channels
                elif channels != player_channels:
                    raise ValueError("player groups have different channels")
                slot_players.append(pairs[0][0])
        else:
            ball_channels = [r.get("channelId") for r in register
                             if _local(r.tag) == "BallChannelRef"]
    if player_channels is None or not ball_channels:
        raise ValueError("could not find player and ball registers in the data format spec")
    return tuple(slot_players), tuple(player_channels), tuple(ball_channels)


def _floats(text, n):
    """n floats from a comma-separated group; NaN-padded if the group is shorter."""
    values = [float(v) for v in text.split(",") if v != ""]
    if len(values) < n:
        values.extend([np.nan] * (n - len(values)))
    return values[:n]


def parse_raw(raw_path, n_slots, player_channels, ball_channels, n_frames):
    """(players (T,S,C) float32, ball (T,B) float32), indexed by video frame, NaN where unseen."""
    n_pc, n_bc = len(player_channels), len(ball_channels)
    players = np.full((n_frames, n_slots, n_pc), np.nan, dtype=np.float32)
    ball = np.full((n_frames, n_bc), np.nan, dtype=np.float32)
    nan_group = [np.nan] * n_pc
    bad_lines = 0

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                bad_lines += 1
                continue
            video_frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            if not 0 <= video_frame < n_frames:
                bad_lines += 1
                continue
            groups = [g for g in parts[1].split(";") if g != ""]
            if len(groups) != n_slots:
                bad_lines += 1
                continue
            players[video_frame] = [nan_group if g[0] == "N" else _floats(g, n_pc)
                                    for g in groups]
            ball[video_frame] = _floats(parts[2], n_bc)

    if bad_lines:
        log.warning("skipped %d malformed tracking lines", bad_lines)
    return players, ball


def load(game=GAME, data_dir=None, use_cache=True):
    """Tracking arrays in metres, keyed by name. See the returned dict for the fields.

    positions (T,S,2) and ball_xy (T,2) are metres from the (0,0) corner; speed (T,S) is the
    tracker's own player speed in m/s; team (S,) is HOME/AWAY; ball_extra (T,k) carries any ball
    channels beyond x and y, named in ball_extra_names.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache_path = os.path.join(CACHE_DIR, "{}_atd.npz".format(game))
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path))])
    if use_cache and os.path.isfile(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if np.array_equal(cached["stamp"], stamp):
            log.info("tracking from cache %s", os.path.relpath(cache_path, ROOT))
            return {k: cached[k] for k in cached.files}

    team, player_channels, ball_channels = parse_layout(meta_path)
    n_frames = source_frames("tactical", game)
    log.info("parsing %s (one-off, then cached) ...", os.path.basename(raw_path))
    players, ball = parse_raw(raw_path, len(team), player_channels, ball_channels, n_frames)

    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width], dtype=np.float32)
    px, py = _channel_index(player_channels, "x"), _channel_index(player_channels, "y")
    bx, by = _channel_index(ball_channels, "x"), _channel_index(ball_channels, "y")
    ps = _channel_index(player_channels, ("speed", "s"), required=False)

    extra_idx = [i for i in range(len(ball_channels)) if i not in (bx, by)]
    out = {
        "positions": players[:, :, [px, py]] * scale,
        "speed": (players[:, :, ps] if ps is not None
                  else np.full(players.shape[:2], np.nan, dtype=np.float32)),
        "team": team,
        "ball_xy": ball[:, [bx, by]] * scale,
        "ball_extra": ball[:, extra_idx],
        "ball_extra_names": np.array([ball_channels[i] for i in extra_idx], dtype="U32"),
        "stamp": stamp,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, **out)
    log.info("cached to %s", os.path.relpath(cache_path, ROOT))
    return out


def _channel_index(channels, names, required=True):
    """Index of the first channel whose name is, or ends with, one of `names` (case-insensitive).

    The Metrica export names player speed "s"; other EPTS exports use "speed".
    """
    names = (names,) if isinstance(names, str) else names
    lowered = [str(c).lower() for c in channels]
    for name in names:
        for i, channel in enumerate(lowered):
            if channel == name or channel.endswith("_" + name):
                return i
    if required:
        raise KeyError("no {} channel among {}".format(names, channels))
    return None
