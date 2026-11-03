# Frame-level ball possession — technical document

More than a Hack 2027, Phase 1. One label per frame of the tactical video: `H`, `A`, `D` (`X`
outside the graded halves).

## 1. Approach

Possession is an interval, and the ranking metric (`spellF1`) scores intervals. So the system does
not classify frames independently. It turns every frame into **evidence**, then **segments** the
match with a three-state Viterbi decoder that charges a fixed cost for every change of state:

```
tracking ─┬─> team evidence  (who is near a slow ball)            ┐
          ├─> dead evidence  (ball out, players slow, ball still)  ├─> dead-ball calibration ─> Viterbi H/A/D ─> run clean-up ─> CSV
Smart Tagging ─> restarts, kickoffs, goals, phase hints           ┘   (external league statistic)
```

A possession changes hands only when the opponent's accumulated evidence exceeds the switching
cost. That is a direct translation of the challenge rule: a touch is not a change of possession;
sustained control is. Deflections, a ball flying past an opponent, and one-frame tracking errors
all fail to pay the cost, so they do not split possessions.

Why this rather than a learned per-frame classifier: we have 5 minutes of labels (17
possessions). A segmenter with a handful of physically meaningful parameters can be reasoned
about; a classifier trained on 7,500 highly correlated frames cannot be trusted on the other 101
minutes.

## 2. Assets used and dropped

| asset | used for | status |
|---|---|---|
| tactical ATD (players + ball) | all evidence | **used** |
| Smart Tagging | set-piece restarts, kickoffs, goals; small team hint from phases; attack direction | **used** |
| public 5-minute ground truth | checking the file is well formed and the pipeline runs end to end | nothing is trained, fitted, tuned or calibrated on it |
| tactical homography | — | dropped: the ATD is already in pitch coordinates |
| broadcast video / ATD / homography | — | dropped: not frame-aligned to tactical, and a third of it has no usable homography |
| tactical video | — | dropped: no detection model needed on top of the provided tracking |

The tracking metadata XML is parsed for the line layout rather than assuming it. That is how we
found that the player speed channel is called `s`, and that the ball carries a speed channel but no
owner or alive flag.

## 3. Evidence

All distances in metres (normalised coordinates × 105.3 × 68.0), 25 fps.

**Team evidence** (positive = home). On each frame where the ball is known, take the distance
from the ball to the nearest home and nearest away player. The evidence is their difference,
clipped, then multiplied by:
- **closeness:** full within 1.5 m of the ball, zero beyond 4 m;
- **ball speed:** full below 4 m/s, down to 15% above 12 m/s, because a passing ball says little
  about control;
- **interpolation:** half weight on interpolated frames.

**Dead-ball evidence** (positive = dead), summed per frame:
- **ball outside the pitch:** seen more than 0.5 m beyond a line, and *not seen inside within
  1 s*. Without that confirmation, high balls projected onto the ground looked like exits and
  created false stoppages.
- **collective slowness:** mean player speed, smoothed over ±1 s and compared with a per-half
  quantile (30%), so no absolute speed threshold is needed.
- **stationary ball:** observed still for 1.5 s or more, as when a free kick is being set.
- **unseen ball:** missing for 3 s or more. On a wide tactical camera that usually means it is off
  the pitch.
- **events:** a confirmed exit is paired with the next restart (a kick after a resting ball, or
  the ball coming back in from outside) within 120 s, and everything between them leans dead.
  Every detected restart leans alive for the 2 s after it.

**Smart Tagging.** For each set piece, the restart kick is located inside its window: the first
frame above 5 m/s after a calm ball. Before that kick the ball leans dead, after it alive, and the
tagged taker's team gets a nudge. Everything in a half before its kickoff, and from a goal to the
following kickoff, is forced dead. Phase tags (Build Up, Progression, Defending in …) add a
0.08-per-frame hint towards the possessing side. The intervals are phases, not possessions, so
the hint is deliberately weak.

**Dead-ball calibration.** One offset is added to the dead-ball score of every frame, searched
by bisection so that the decoded graded frames contain 55 min 58 s of ball-in-play time. That is
the average effective playing time of Barcelona's 2024-25 LaLiga matches (LaLiga figures, as
reported by El Diario Alerta on 2025-05-27). For this match that means 47.2% of graded frames
dead. The target is an external statistic, not a property of the ground truth.

**Optional components, off in the submission.** Two small models remain in the code, switched
off with `use_learned=0` and `use_shape=0`:
- a learned dead-ball model (`possession/deadmodel.py`), trained on Smart Tagging pseudo-labels;
- a team-shape model (`possession/shape.py`), self-trained on the first-pass decode.

Neither uses the ground truth or the video. They are off because whether to use them, and at
what weight, could only be judged against the public ground truth, which the rules forbid (see
section 5).

## 4. Possession rules implemented

- **Restarts.** The restart frame is where the ball is kicked from rest, or comes back in from
  outside. The team on the ball at that moment starts the possession.
- **Loose and contested balls.** When both teams are near the ball, the evidence nets to roughly
  zero, so the previous team keeps possession, as the rules require.
- **Change of possession.** The new team must out-evidence the old one by the switching cost
  (12 units, about half a second of clear, slow, close control).
- **Interceptions vs deflections.** A fast ball near an opponent carries 15% weight, so a
  deflection does not pay the cost; a ball that slows down at an opponent's feet does.
- **Ball not visible (41% of graded frames).** Gaps up to 0.48 s are interpolated; beyond that
  the decoder keeps the current state until new evidence arrives.
- **Dead ball** is always `D`. We never try to say who "owns" a stopped ball.
- **Run clean-up** after decoding. A dead run under 1 s is not a stoppage and takes the previous
  label. A possession under 4 s with dead ball on both sides is a player holding the ball during
  one stoppage (a throw-in or goal kick being taken), so it becomes `D`.

## 5. Compliance: where every parameter comes from

The organisers' rules clarification forbids training, fitting or calibrating on the ground-truth
possession data or on the video, and requires any tuning to use external data. The submitted
configuration follows it:

| parameter group | source |
|---|---|
| distances, speeds, durations (control radius, pass speed, stationary-ball time, …) | football and tracking physics, set before any scoring |
| switching costs (12 team, 8 dead) | about half a second of clear control; set before any scoring |
| dead-ball level | calibrated to an external statistic (effective playing time, section 3) |
| per-half adaptive thresholds (player-speed quantile) | computed automatically from this match's tracking; no labels involved |
| run clean-up (1 s, 4 s) | football reasoning: no stoppage is shorter than 1 s; a possession of under 4 s between two stoppages is a restart being taken |

**What was reverted.** Earlier versions of this repository, written before the clarification,
used the public 5-minute ground truth to choose some settings: the switching cost was set to
match the public window's mean possession length, and the two optional models above were enabled
at weights picked from public-window scores. After the clarification those choices were reverted
to their original, reasoning-based values, and the code path that could train on the public
window was deleted. No hand labels were ever created.

**What the public window is still used for.** Only to confirm the file is well formed and the
pipeline runs: public spellF1 0.3875, macroF1 0.8533, against 0.3256 / 0.6207 for the reference
baseline. Nothing is chosen from these numbers.

## 6. Limitations

- **Very short possessions** (under ~2 s) are still absorbed by the switching cost. The truth
  contains some.
- **A player holding the ball during a stoppage** (throw-in, goal kick) produces strong team
  evidence. The clean-up merges splits under 4 s; longer ones remain.
- **Exits are rarely seen.** The ball usually leaves the camera's view as it goes out, so most
  dead-ball time comes from collective slowness rather than from exit → restart pairs.
- **Stoppages without a tag.** Smart Tagging has 59 set pieces, against roughly 100+ stoppages in
  a match.
- **No player identities**, so there is no notion of a carrier's own run.

With more time: tuning the switching costs and the optional models on external tracking data with
possession labels (subject to the organisers' confirmation of a dataset's licence); a
duration-aware (semi-Markov) decoder; and using the broadcast feed to confirm stoppages around
replays.

## 7. Reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-possession.txt     # numpy, pandas, pinned
export MTAH_DATA_DIR=/path/to/data                         # contains tactical/ and broadcast/
.venv/bin/python -m possession.pipeline --out evaluation/results/<team_name>.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/<team_name>.csv
.venv/bin/python -m unittest discover tests                # synthetic tests, no data needed
```

The pipeline is deterministic (no random state, exact Viterbi decoding, a fixed-step bisection
for the calibration), and two runs produce byte-identical files. It was developed on Python
3.14; the pinned numpy/pandas need Python 3.11 or newer. The first run parses the tracking file once into
`cache/` (gitignored); a full run, parse included, takes about 15 s on a laptop. Every parameter is in
`possession/config.py`, and any of them can be overridden with `--set name=value`.

Everything that touches the gated data stays on the participant's machine: `data/`, `cache/` and
`evaluation/results/*.csv` are gitignored.
