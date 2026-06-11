# RedlineBench

An RL environment for AI contract negotiation, scored by a verifiable outcome instead of an AI judge.

## The idea
Math and code are easy to grade because the answer checks itself. Legal negotiation is not. There is no answer key for "good deal." RedlineBench scores the final negotiated terms against the client's priorities, producing a hard, reproducible number with no judge in the loop.

## What's here so far
- A scorekeeper that grades a finished negotiation from the client's point of view
- Two negotiating sides: our AI versus an opposing-counsel "vendor"
- A multi-round back-and-forth that produces a settled deal
- A baseline: the untrained model's average score across several situations

## Status
Environment and baseline complete. Baseline score: 0.46. Next: training the model against the environment and measuring whether that score climbs.

