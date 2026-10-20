"""Synthetic-data tests for the possession pipeline. No match data needed.

    python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import atd  # noqa: E402
from possession.config import Params, with_overrides  # noqa: E402
from possession.decode import decode, viterbi  # noqa: E402
from possession.evidence import (detect_restart, interpolate_ball, long_runs,  # noqa: E402
                                 moving_nanmean, team_evidence)

META = """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <Teams>
      <Team id="T1"><Name>FC Barcelona</Name></Team>
      <Team id="T2"><Name>Athletic Club</Name></Team>
    </Teams>
    <Players>
      <Player id="P1" teamId="T1"/>
      <Player id="P2" teamId="T2"/>
    </Players>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification separator=":" startFrame="1" endFrame="3">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="p2_x"/>
          <PlayerChannelRef playerChannelId="p2_y"/>
          <PlayerChannelRef playerChannelId="p2_s"/>
        </SplitRegister>
        <SplitRegister separator=",">
          <PlayerChannelRef playerChannelId="p1_x"/>
          <PlayerChannelRef playerChannelId="p1_y"/>
          <PlayerChannelRef playerChannelId="p1_s"/>
        </SplitRegister>
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
        <BallChannelRef channelId="z"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
  <Sensors>
    <PlayerChannels>
      <PlayerChannel id="p1_x" playerId="P1" channelId="x"/>
      <PlayerChannel id="p1_y" playerId="P1" channelId="y"/>
      <PlayerChannel id="p1_s" playerId="P1" channelId="speed"/>
      <PlayerChannel id="p2_x" playerId="P2" channelId="x"/>
      <PlayerChannel id="p2_y" playerId="P2" channelId="y"/>
      <PlayerChannel id="p2_s" playerId="P2" channelId="speed"/>
    </PlayerChannels>
  </Sensors>
</main>
"""

RAW = ("1:0.1,0.2,1.5;0.3,0.4,2.0:0.5,0.5,0.0\n"
       "2:NaN,NaN,NaN;0.3,0.4,2.0:NaN,NaN,NaN\n"
       "3:0.1,0.2,1.5;0.3,0.4,2.0;:0.6,0.5,0.1\n")


class TestAtdParsing(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.meta = os.path.join(self.dir, "meta.xml")
        self.raw = os.path.join(self.dir, "raw.txt")
        with open(self.meta, "w") as f:
            f.write(META)
        with open(self.raw, "w") as f:
            f.write(RAW)

    def test_layout_follows_channel_refs(self):
        team, player_channels, ball_channels = atd.parse_layout(self.meta)
        # Slot 0 is P2 (away) because the spec lists it first, not the Player element order.
        self.assertEqual(team.tolist(), [atd.AWAY, atd.HOME])
        self.assertEqual(player_channels, ["x", "y", "speed"])
        self.assertEqual(ball_channels, ["x", "y", "z"])

    def test_raw_rows_land_on_video_frames(self):
        team, pc, bc = atd.parse_layout(self.meta)
        players, ball = atd.parse_raw(self.raw, len(team), pc, bc, 3)
        self.assertAlmostEqual(float(players[0, 0, 0]), 0.1, places=6)
        self.assertTrue(np.isnan(players[1, 0]).all())
        self.assertTrue(np.isnan(ball[1]).all())
        self.assertAlmostEqual(float(ball[2, 0]), 0.6, places=6)


class TestEvidence(unittest.TestCase):
    def test_interpolation_fills_short_gaps_only(self):
        xy = np.array([[0, 0], [np.nan, np.nan], [2, 2], [np.nan, np.nan], [np.nan, np.nan],
                       [np.nan, np.nan], [6, 6]], dtype=float)
        filled, interp = interpolate_ball(xy, max_gap=2)
        self.assertEqual(filled[1].tolist(), [1, 1])
        self.assertTrue(np.isnan(filled[3]).all())
        self.assertEqual(interp.tolist(), [False, True, False, False, False, False, False])

    def test_long_runs(self):
        mask = np.array([1, 1, 0, 1, 1, 1, 0], dtype=bool)
        self.assertEqual(long_runs(mask, 3).tolist(), [0, 0, 0, 1, 1, 1, 0])

    def test_moving_nanmean_ignores_nan(self):
        values = np.array([1.0, np.nan, 3.0])
        self.assertAlmostEqual(moving_nanmean(values, 1)[1], 2.0)

    def test_detect_restart(self):
        speed = np.array([0.0] * 20 + [8.0] * 5)
        self.assertEqual(detect_restart(speed, 0, 25, v_kick=5.0), 20)
        self.assertIsNone(detect_restart(np.full(25, 8.0), 0, 25, v_kick=5.0))

    def test_team_evidence_favours_nearest_team(self):
        n = 10
        data = {
            "positions": np.array([[[10.0, 10.0], [20.0, 20.0]]] * n, dtype=np.float32),
            "team": np.array([0, 1], dtype=np.int8),
            "ball_xy": np.array([[10.5, 10.0]] * n, dtype=np.float32),
            "speed": np.zeros((n, 2), dtype=np.float32),
        }
        score, _ = team_evidence(data, Params(), fps=25)
        self.assertTrue((score > 0).all())


class TestDeadModel(unittest.TestCase):
    def test_fit_separates_and_auc(self):
        from possession.deadmodel import auc, fit, logit
        x = np.concatenate([np.zeros((50, 2)), np.ones((50, 2))]) + np.linspace(0, 0.1, 100)[:, None]
        y = np.array([0] * 50 + [1] * 50, dtype=float)
        model = fit(x, y, np.ones(100))
        self.assertGreater(auc(logit(x, model), y), 0.99)
        self.assertEqual(auc(np.arange(4.0), np.array([0, 0, 1, 1])), 1.0)


class TestShape(unittest.TestCase):
    def test_attack_direction_from_kickoff_lineup(self):
        from possession.shape import home_attack_direction
        n = 200
        pos = np.zeros((n, 4, 2))
        pos[:, :2, 0] = 30.0   # home in the low-x half before kickoff
        pos[:, 2:, 0] = 70.0
        team = np.array([0, 0, 1, 1], dtype=np.int8)
        dirs = home_attack_direction(pos, team, [100], [(1, 60, 199)], fps=25)
        self.assertEqual(dirs, {1: 1})


class TestDecoder(unittest.TestCase):
    def test_short_blip_does_not_switch(self):
        team = np.array([1.0] * 50 + [-1.0] * 5 + [1.0] * 50)
        scores = np.stack([team / 2, -team / 2, np.full(len(team), -1.0)], axis=1)
        path = viterbi(scores, cost_team=10.0, cost_dead=10.0)
        self.assertTrue((path == 0).all())

    def test_sustained_change_switches(self):
        team = np.array([1.0] * 50 + [-1.0] * 50)
        scores = np.stack([team / 2, -team / 2, np.full(len(team), -1.0)], axis=1)
        path = viterbi(scores, cost_team=10.0, cost_dead=10.0)
        self.assertEqual(path[:45].tolist(), [0] * 45)
        self.assertEqual(path[55:].tolist(), [1] * 45)

    def test_decode_writes_x_outside_halves(self):
        team = np.ones(30)
        dead = np.full(30, -1.0)
        labels = decode(team, dead, [(1, 5, 14), (2, 20, 24)], 30, 10.0, 10.0)
        self.assertEqual("".join(labels), "X" * 5 + "H" * 10 + "X" * 5 + "H" * 5 + "X" * 5)

    def test_clean_runs(self):
        from possession.decode import clean_runs
        labels = np.array(list("HHHHDHHHH" + "DDDDAADDDD" + "AAAA"))
        out = clean_runs(labels, [(1, 0, len(labels) - 1)], min_dead=2, max_sandwich=3)
        self.assertEqual("".join(out), "HHHHHHHHH" + "DDDDDDDDDD" + "AAAA")

    def test_overrides(self):
        params = with_overrides(Params(), ["cost_team=3", "still_min_frames=10"])
        self.assertEqual(params.cost_team, 3.0)
        self.assertEqual(params.still_min_frames, 10)
        self.assertIsInstance(params.still_min_frames, int)


if __name__ == "__main__":
    unittest.main()
