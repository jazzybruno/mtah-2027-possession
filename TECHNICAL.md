# Frame-level ball possession — technical document

More than a Hack 2027, Phase 1. One label per frame of the tactical video: `H`, `A`, `D` (`X`
outside the graded halves).

## 1. Approach

Possession is an interval, and the ranking metric (`spellF1`) scores intervals. So the system does
not classify frames independently. It turns every frame into **evidence**, then **segments** the
match with a three-state Viterbi decoder that charges a fixed cost for every change of state:

```
tracking ─┬─> team evidence  (who is near a slow ball)                        ┐
          ├─> dead evidence  (hand-set terms + learned dead-ball model)       ├─> Viterbi ─> shape model ─> Viterbi ─> run clean-up ─> CSV
Smart Tagging ─> restarts, kickoffs, goals, phase hints, model training labels┘   (pass 1)   (self-trained)   (pass 2)
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
| public 5-minute ground truth | pipeline checks, out-of-sample tests | used for checking only; not trained on |
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

**Learned dead-ball model.** A logistic regression on 18 context features, added to the hand-set
dead-ball score at a small weight (0.15). The features cover player motion at three time scales,
ball visibility, stillness and speed, the ball outside or near a line, nearest-player distance,
player spread and the number of tracked players. It trains only on **Smart Tagging
pseudo-labels**:
- **dead:** from each located restart back to where the stoppage began. The label is extended
  backwards while play stays calm (players slow, or the ball unseen or still), up to 60 s, plus
  pre-kickoff and goal-to-kickoff stretches;
- **alive:** the core of open-play phases (Build Up, Progression, Creating Chances, Attacking
  Transition), away from any restart.

The public window is deliberately left out of training, so it stays an out-of-sample test. The
logit is centred on this match, so the top ~45% of frames lean dead.

**Team shape (second pass).** A logistic regression on 10 shape features. They are oriented so
+x is home's attacking direction, read off the pre-kickoff lineup:
- where the play is (the centroid of all players);
- each team's front and back lines and the gap between the teams;
- the difference in width and length between the teams;
- where the ball was last seen.

It is **self-trained**: the labels are our own first-pass decode on about 33k frames where the
ball is visible and the proximity evidence is clear. Its output feeds a second decode, at full
weight where the ball is silent and half weight elsewhere. Any team evidence also lifts "alive"
over "dead" in the decoder; that lift is cancelled for the shape term, so shape only picks the
side.

## 4. Possession rules implemented

- **Restarts.** The restart frame is where the ball is kicked from rest, or comes back in from
  outside. The team on the ball at that moment starts the possession.
- **Loose and contested balls.** When both teams are near the ball, the evidence nets to roughly
  zero, so the previous team keeps possession, as the rules require.
- **Change of possession.** The new team must out-evidence the old one by the switching cost
  (8 units, roughly a third of a second of clear, slow, close control).
- **Interceptions vs deflections.** A fast ball near an opponent carries 15% weight, so a
  deflection does not pay the cost; a ball that slows down at an opponent's feet does.
- **Ball not visible (41% of graded frames).** Gaps up to 0.48 s are interpolated; beyond that
  the decoder keeps the current state and the team-shape model adds evidence.
- **Dead ball** is always `D`. We never try to say who "owns" a stopped ball.
- **Run clean-up** after decoding. A dead run under 1 s is not a stoppage and takes the previous
  label. A possession under 4 s with dead ball on both sides is a player holding the ball during
  one stoppage (a throw-in or goal kick being taken), so it becomes `D`.

## 5. How choices were made without overfitting 17 possessions

The public window was used to catch bugs and as an out-of-sample test, never as the objective.
Parameters come from football priors, with three checks that do not touch the private ground
truth:

1. **Dead-ball share.** LaLiga effective playing time is about 52–55 of the 106 graded minutes, so
   about 40–50% of graded frames should be dead. We predict 46%.
2. **Possession count.** The public window averages 14.2 s per possession. Over ~3,400 s of live
   play that implies about 240 possessions. The switching cost was set to give 239 (mean 14.3 s).
3. **Smart Tagging agreement** (`--check`). With tags switched off, it measures how many set
   pieces fall on a predicted stoppage. Tags are independent of the tracking signals, so this is a
   check that needs no ground truth.

Out-of-sample results on the public window:

| component | metric | value |
|---|---|---|
| hand-set dead-ball score | frame AUC, dead vs alive | 0.877 |
| learned dead-ball model, labels from the 2.5 s before restarts | frame AUC, dead vs alive | 0.868 |
| learned dead-ball model, labels extended back over the stoppage (**used**) | frame AUC, dead vs alive | **0.919** |
| team-shape model (self-trained) | frame AUC, home vs away | 0.956 |

| pipeline stage | public spellF1 | public macroF1 |
|---|---|---|
| reference baseline (`smooth50`) | 0.3256 | 0.6207 |
| v1: proximity + hand-set dead ball | 0.3875 | 0.8523 |
| + event-based stoppages | 0.3875 | 0.8526 |
| + team shape, second pass | 0.4000 | 0.8635 |
| + switching cost set from the possession-length prior | 0.4444 | 0.8334 |
| + run clean-up (4 s) | 0.4444 | 0.8334 |
| + learned dead-ball model, tags-only training (**submitted**) | **0.4800** | 0.8369 |

The public window is 17 possessions, and one possession moves `spellF1` by about 6 points. These
numbers show the pipeline is sound, not how it ranks.

**What changed the learned model's verdict.** Its first version labelled only the 2.5 s before
each restart as dead. It learned "about to restart" rather than "stoppage in progress", ranked
frames no better than the hand-set score, and fragmented the decode, so it was switched off.
Extending the dead labels back over the calm stretch before each restart moved its out-of-sample
AUC from 0.868 to 0.919 and made it worth adding at a small weight. Larger weights (0.3, 0.5)
lowered macroF1, so it stays a correction to the hand-set score rather than a replacement.

**Choices not taken from the public window.** A 7 s clean-up threshold would have fixed a specific
split stoppage in the public window and scored higher there. 4 s was kept because it rests on the
football argument, not on that one example.

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

With more time: a duration-aware (semi-Markov) decoder with separate minimum lengths for
possessions and stoppages; pseudo-labels for team control as well as for dead ball; and using
the broadcast feed to confirm stoppages around replays.

## 7. Reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-possession.txt     # numpy, pandas, pinned
export MTAH_DATA_DIR=/path/to/data                         # contains tactical/ and broadcast/
.venv/bin/python -m possession.pipeline --out evaluation/results/<team_name>.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/<team_name>.csv
.venv/bin/python -m unittest discover tests                # synthetic tests, no data needed
```

The pipeline is deterministic (no random state, exact Viterbi decoding, fixed-step gradient
descent for both learned models), and two runs produce byte-identical files. It was developed on Python 3.14 with the
pinned versions above. The first run parses the tracking file once (about a minute) into
`cache/`, which is gitignored; later runs take seconds. Every parameter is in
`possession/config.py`, and any of them can be overridden with `--set name=value`.

Everything that touches the gated data stays on the participant's machine: `data/`, `cache/` and
`evaluation/results/*.csv` are gitignored.
