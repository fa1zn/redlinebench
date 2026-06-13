<div align="center">

# RedlineBench

**An RL environment where an AI lawyer negotiates a contract liability cap, scored by a verifiable reward.**

[![Environment](https://img.shields.io/badge/Prime_Intellect-redline--negotiate-2b57e0)](https://app.primeintellect.ai/dashboard/environments/fa1zvn/redline-negotiate)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.10+-3776ab)
&nbsp;
![Training](https://img.shields.io/badge/Method-GRPO-555)

</div>

RedlineBench is a benchmark for AI contract negotiation. An AI lawyer argues for a liability cap against opposing counsel, and the reward is calculated rather than judged by another model, so every outcome is verifiable.

The environment, `redline-negotiate`, is published on the [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments/fa1zvn/redline-negotiate).

<div align="center">
<img src="https://raw.githubusercontent.com/fa1zn/redlinebench/master/redline-diagram.png" width="820" alt="One negotiation episode: the model proposes a cap, a fixed-rule vendor accepts or counters, and the scorekeeper turns the outcome into a reward">
</div>

## The setup

A liability cap is the ceiling on what one party owes if a contract goes wrong. It gets fought over in almost every commercial deal. The client wants it high so they can recover if the vendor fails, and the vendor wants it low so they are not exposed. They argue toward a number.

The model plays the client and tries to land the cap as high as it can without losing the deal. I used this term because it comes down to a single number, which means I can grade the result without a human or a judge model in the loop.

## The reward

The scorekeeper grades the agreed cap on a fixed scale:

- No deal: -1
- Cap at the client's walkaway, $100K: 0
- Cap at the client's target, $1M: 1.0
- Linear between those points

## The opponent

The vendor is a fixed rule, not a second model. It accepts any offer at or below $200K. Above that it rejects and counters, conceding a quarter of the gap each round. If it rejects every round, the deal dies and the model scores -1.

Keeping the opponent fixed means the only thing changing across a run is the model itself.

## Training

I trained Qwen3.5-0.8B with GRPO, hosted on Prime Intellect. The reward climbs as the model learns to push the cap higher and still close the deal.

<div align="center">
<img src="https://raw.githubusercontent.com/fa1zn/redlinebench/main/redlinebench_reward_curve.png" width="820" alt="Reward per step climbing from 0.45 to 0.95 over a five-step run">
</div>

## What broke, and why it matters

A longer run of thirty steps failed at step 12. That failure is the result worth reporting.

GRPO learns from the spread between attempts. It runs a scenario several times, reinforces the attempts that beat the batch average, and pushes down the ones below it. With no spread, there is nothing to learn from.

By step 12 the model was winning every attempt by the same margin. The batches went flat, the signal collapsed, and training stopped itself after ten dead batches in a row.

<div align="center">
<img src="https://raw.githubusercontent.com/fa1zn/redlinebench/main/redlinebench_saturation.png" width="820" alt="Reward saturates by step 12, then training halts because the learning signal collapses">
</div>

It got there by demanding huge numbers, pushing the agreed cap toward $80M, sitting right on the edge of the reward and the vendor rule. It was not negotiating better, it was exploiting the grader.

This is the failure mode that makes reward design hard in legal AI. A model optimizing a verifiable reward will find the cheapest way to max it, and if the environment allows an absurd anchor, it takes it.

## What this says about the environment

The environment is too easy. Anything that saturates in twelve steps is not testing much. A real test needs a harder opponent or a reward that does not pay out for absurd anchors.

## Next

- Evaluate the trained checkpoint against the fixed baseline on held-out scenarios, for a clean before and after
- Make the vendor reject absurd anchors so the model cannot win by overreaching
- Reshape the reward so demanding $80M earns nothing
- Reimplement the GRPO loop by hand on the Tinker API and rerun against the hardened environment

## Run it

```bash
prime env install fa1zvn/redline-negotiate
```
