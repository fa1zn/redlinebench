# redline-negotiate

### Overview
- **Environment ID**: `redline-negotiate`
- **Short description**: Multi-turn contract negotiation where a model negotiates liability caps against an opposing-counsel AI, scored by a verifiable reward function.
- **Tags**: multi-turn, negotiation, legal, train, eval

### Datasets
- **Primary dataset(s)**: Procedurally generated negotiation scenarios with varying vendor starting offers
- **Source links**: Generated in-environment from scenario configurations
- **Split sizes**: Train: 7 scenarios, Eval: 8 scenarios (different difficulty distributions)

### Task
- **Type**: multi-turn
- **Output format expectations**: Dollar amount (e.g., "$750,000" or "750000")
- **Rubric overview**:
  - `negotiation_reward`: Main reward based on final agreed cap (0.0-1.0 linear scale)
  - `agreed_cap_metric`: Tracks the final settlement amount
  - `rounds_completed_metric`: Tracks negotiation duration

### Reward Function (Verifiable)

The reward is calculated deterministically based on the final agreed price:

| Outcome | Reward |
|---------|--------|
| No deal reached | -1.0 |
| Cap ≤ $100,000 (walkaway) | 0.0 |
| Cap ≥ $1,000,000 (ideal) | 1.0 |
| Between walkaway and ideal | Linear interpolation (0.0 - 1.0) |

Settlement = midpoint of final buyer and vendor offers.

### Quickstart

Install the environment:
```bash
prime env install redline-negotiate
```

Run an evaluation with default settings:
```bash
prime eval run redline-negotiate
```

Configure model and sampling:
```bash
prime eval run redline-negotiate \
  -m openai/gpt-4.1-mini \
  -n 8 -r 3 \
  -a '{"total_rounds": 3, "opponent_model": "claude-haiku-4-5-20251001"}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `total_rounds` | int | `3` | Number of negotiation rounds |
| `opponent_model` | str | `claude-haiku-4-5-20251001` | Model used for opposing counsel AI |
| `scenarios` | list[int] | See below | Custom list of vendor starting offers |

Default training scenarios: `[50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000]`

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `negotiation_reward` | Main scalar reward (0.0-1.0 based on final cap) |
| `agreed_cap_metric` | Final agreed dollar amount |
| `rounds_completed_metric` | Number of negotiation rounds completed |

### How Negotiation Works

```
Round 1: Model counter-offers → Vendor AI responds with counter
Round 2: Model counters → Vendor AI responds with counter
Round 3: Model makes final offer → Settlement = midpoint
```

### Environment Variables

The opposing counsel AI uses the OpenAI-compatible API:
- `OPENAI_API_KEY` - Required for opponent model calls
