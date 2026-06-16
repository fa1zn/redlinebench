redlinebench_curves.png](redlinebench_curves.png</div>

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
