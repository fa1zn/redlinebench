<div align="center">

# RedlineBench

**An RL environment where an AI lawyer negotiates a contract liability cap, scored by a verifiable reward instead of a judge model.**

[![Environment](https://img.shields.io/badge/Prime_Intellect-redline--negotiate-2b57e0)](https://app.primeintellect.ai/dashboard/environments/fa1zvn/redline-negotiate)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.10+-3776ab)
&nbsp;
![Training](https://img.shields.io/badge/Method-GRPO-555)

</div>

RedlineBench is a small RL environment for contract-clause negotiation. An AI lawyer argues for a liability cap against a deterministic opposing counsel, and the reward is calculated from the agreed number rather than scored by another model, so every outcome is verifiable.

I ran GRPO on it end to end with Tinker. The first training curve came back flat, and chasing down why turned into the actual result: three reward formulations, two of which can't train anything, and one that can.

## Start here

- **[redlinebench_writeup.md](redlinebench_writeup.md)** — the full story: the flat curve, the two ways a reward kills its own gradient, and the smooth penalty that finally produces a learnable signal. Read this one.
- **[redlinebench_curves.png](redlinebench_curves.png)** — the three reward curves on one chart.
- **[redlinebench/](redlinebench/)** — the environment, scorer, baseline, and training configs.

<div align="center">
<img src="https://raw.githubusercontent.com/fa1zn/redlinebench/main/redlinebench_curves.png" width="820" alt="Three reward curves: naive saturates at the ceiling, hard penalty pins to a floor, smooth penalty climbs about 20x">
</div>

## The setup

A liability cap is the ceiling on what one party owes if a contract goes wrong, and it gets fought over in almost every commercial deal. The client wants it high so they can recover if the vendor fails. The vendor wants it low so they are not exposed. They argue toward a single number.

The policy plays the buyer's counsel and tries to land the cap as high as it can without losing the deal. Because the whole thing comes down to one dollar figure, the result can be graded without a human or a judge model in the loop.

## The reward

`score_outcome` grades the final agreed cap against the client's interests. A cap at or below the walk-away point is worthless (0.0), a cap at the client's ideal is perfect (1.0), and it is linear in between. The vendor is a fixed rule, not a second model: it accepts anything at or below its threshold, otherwise concedes a fixed fraction of the gap per round and settles at the midpoint. Keeping the opponent deterministic means the only thing changing across a run is the policy.

## The result

I trained Qwen3-8B with GRPO, group size 8, ten batches, across three versions of the reward:

- **Naive reward** saturates immediately. The base model anchors absurdly high, clears the ceiling, and scores ~0.96 at batch 0. Every sample scores the same, the advantages go to zero, and there is nothing to learn from.
- **Hard penalty** collapses the curve to a floor instead of a ceiling. Overshoot gets punished, but the clamp means a 6M anchor and a 60M anchor score identically, so the group still has no spread.
- **Smooth penalty** removes the clamp and lets the penalty decay continuously, so a smaller overshoot always scores strictly higher than a larger one. Now the eight anchors in a group land on eight different scores, GRPO has a direction to push, and the reward climbs about 20x over the run.

Both flat curves were the same bug wearing different clothes. A reward is trainable not because it is harder, but because it varies with a real slope across the region the policy actually explores. The full diagnosis is in the [writeup](redlinebench_writeup.md).

## Next (v2)

- Hide the vendor's walk-away point and vary it per episode, so the optimal anchor depends on state the model has to infer rather than a fixed constant
- Add an LLM counterparty in place of the fixed rule
- Score against a held-out eval set for a clean before and after
- Sweep across models

## Run it

```bash
prime env install fa1zvn/redline-negotiate
```
