# Planarian behavior ethogram (white-7MP rig)

> **UPDATE 2026-06-26 — v1 model trained on clean data.** Production model
> `realtime_runs/behavior_clf.joblib` has 5 classes: **contracted, gliding,
> resting, turning, wig_wag** (RandomForest, LOO-CV macro-F1 0.84). The
> **"scrunching" entry below was renamed `contracted`** for worm_run_01: the
> undisturbed worm rests in a contracted pear/oval posture, which is NOT the
> scrunching escape gait. True scrunching (the rhythmic escape lurch defined
> below) is absent in undisturbed footage — keep the definition for a future
> stimulus session (it would be a separate class then). peristalsis/reversing had
> too few examples to model and are deferred. Definitions below remain the
> labeling reference. System status: see SESSION_HANDOFF.md.

Operational definitions for labeling worm behavior in the 3 s windows shown by
`behavior_label_tool.py`. The goal is INTER-LABELER CONSISTENCY: two people (or
the same person on two days) should assign the same label to the same window.
Fuzzy boundaries — especially gliding vs turning — cap how well any model can do,
so the criteria below are deliberately explicit and measurable.

Each behavior lists: what it is, the observable cue, and a measurable proxy (the
`behavior_features.py` feature that tracks it, with a rough threshold from the
observed white-7MP distribution / the `behavior_rules.py` constants). Thresholds
are guidance for edge calls, not hard gates — judge the video first.

Labeling is MULTI-LABEL: if a window genuinely shows two behaviors (e.g. the
worm glides and turns), select both. Pick the behavior(s) that dominate the 3 s.

## Decision order (most specific first)
Resolve a window by going down this list and taking the first that clearly fits;
then add any second behavior that also clearly applies.

1. **scrunching** — escape gait: rhythmic body-length oscillation with strong
   contraction, little net travel. The whole body shortens and elongates
   repeatedly (a caterpillar-like lurch), often in response to a noxious cue.
   - cue: body visibly contracts/elongates >=2 times in the window
   - proxy: `bodylen_cycles >= 1` AND `bodylen_contract >= 0.20`
   - vs peristalsis: scrunch is bigger-amplitude, more abrupt, less net progress.

2. **peristalsis** — normal locomotor body waves: low-amplitude length changes
   traveling along the body, usually WITH smooth gliding. Subtle.
   - cue: gentle, regular body-length ripple; worm still makes headway
   - proxy: `bodylen_cycles >= 1` AND `bodylen_contract < 0.20`
   - note: often co-occurs with gliding — multi-label both if so.

3. **reversing** — the worm moves while its body heading flips ~180 deg (it backs
   up / leads with the other end). RARE; label only if clearly real on the video.
   - cue: direction of travel reverses relative to body axis
   - proxy: moving (`speed >= 0.1`) AND `heading_change_deg >= 140`
   - CAUTION: a sudden 180 deg flip in the tracker overlay can be a head/tail
     tracking artifact, not a real reversal. You are labeling the RAW video — only
     call reversing if the worm itself visibly backs up.

4. **turning** — directed travel that changes DIRECTION: the worm is translating
   and its path/heading curves meaningfully over the window.
   - cue: the worm moves AND its travel direction noticeably changes (a curved
     path, or a pivot), beyond a straight glide
   - proxy: moving (`speed >= 0.1`) AND (`heading_change_deg >= 40` over the
     window OR a visibly curved path / elevated `path_curv_deg_mm`,
     `ang_vel_p90_deg_s`)
   - vs gliding: see the boundary note below. This is the key confusion.

5. **wig_wag** — head sweeping side to side with LITTLE net travel (scanning).
   - cue: head swings left-right repeatedly while the body stays roughly in place
   - proxy: `head_osc_deg >= 20` AND `head_reversals >= 1` AND NOT translating
     (`speed < ~0.1`)
   - vs turning: wig_wag is head movement WITHOUT net travel; turning is a change
     of TRAVEL direction while moving.

6. **gliding** — steady, directed translation in a roughly straight line (cilia-
   driven smooth crawl). The default "moving normally" behavior.
   - cue: worm advances smoothly, heading roughly constant
   - proxy: `speed >= 0.10-0.15 mm/s`, low `heading_change_deg` (< ~40),
     low body-length oscillation

7. **resting** — essentially stationary, no organized head/body movement.
   - cue: worm holds position; only sub-pixel jitter
   - proxy: `speed < 0.10 mm/s`, no head oscillation, no length cycling

8. **unknown** — only if the worm is not visible / not enough of the window is
   trackable to judge. Avoid using this as a "not sure" escape — make a call.

## The gliding vs turning boundary (the main source of label noise)
Both involve the worm MOVING, so speed does not separate them. The distinction is
whether the DIRECTION OF TRAVEL changes:
- straight, steady advance over the 3 s  -> **gliding**
- the path clearly curves, or the worm pivots/redirects -> **turning**
Rule of thumb: if the net change in heading across the window is under ~30-40 deg
and the path looks straight, call it gliding; if it's a clear arc or pivot
(>~40 deg, or a visibly curved track), call it turning. A long, gentle drift that
barely curves is gliding, not turning. When a window has a straight glide that
ends in a turn, multi-label BOTH.

## Notes specific to this dataset
- The 2026-06-02 baseline session is dominated by gliding and resting. Escape/
  stress gaits (scrunching, peristalsis, reversing) are rare here because they
  are typically EVOKED (noxious stimulus / drug). Expect few of them; do not
  force a window into a rare class to "find" it.
- Label the behavior you SEE in the raw video. The tracker's head/tail marker and
  position can occasionally glitch (position jitter, rare head/tail flips); ignore
  the overlay and judge the animal.
