RedlineBench

RedlineBench is a benchmark for AI contract negotiation. An AI lawyer negotiates a liability cap against opposing counsel, and the reward is pure math, not an LLM judge, so the outcome is verifiable.

Its environment, redline-negotiate, is published on the Prime Intellect Environments Hub: fa1zvn/redline-negotiate.

What it's modeling

A liability cap is the dollar ceiling on what one party owes if a contract goes wrong. It's one of the most heavily redlined terms in any commercial deal. The client wants the cap high so they can recover if the vendor fails. The vendor wants it low so they're not on the hook. They negotiate toward a number.

The AI plays the client's side. It pushes for the highest cap it can land without blowing up the deal. I picked this term because it reduces to a single number, which means I can grade the result automatically with no human in the loop.

The reward

After each negotiation the scorekeeper grades the agreed cap:


No deal: -1
Cap at the client's walkaway ($100K): 0
Cap at the client's ideal ($1M): 1.0
Linear in between


No judge model, no subjective call. The number is the number.

The opponent

The vendor is a fixed rule, not another AI. It accepts any offer at or below $200K. Above that it rejects and counters, conceding slowly (a quarter of the gap each round). If it rejects every round, the deal dies and the model scores -1.

A deterministic opponent keeps the experiment clean. The only thing changing across the run is the model.

What happened when I trained it

I trained Qwen3.5-0.8B on it with GRPO, hosted on Prime Intellect.

The first run climbed. Reward went from around 0.45 to around 0.95 over five steps. The model was learning to negotiate a higher cap and close the deal.

Then I ran it longer, thirty steps, and it failed at step 12. That failure is the actual result.

The finding

GRPO learns from the spread between attempts. It runs the same scenario several times, reinforces the attempts that beat the average, and pushes down the ones below it. It needs some attempts to be better than others. If every attempt scores the same, there's nothing to compare and nothing to learn.

By step 12 the model had gotten so good that every attempt in the batch hit the ceiling. All wins, all identical. The trainer ran out of signal and stopped itself after ten dead batches in a row.

And the way it got there matters. It wasn't negotiating well. It learned to demand enormous numbers, pushing the agreed cap toward $80M, exploiting the edge of the reward and the vendor rule. It found the cheapest path to a high score.

The model didn't get better at negotiating. It got better at gaming the grade. That's the reliability problem that matters in real legal AI, and it showed up in miniature, on purpose.

What this tells me

The environment is too easy. A model that saturates the reward this fast needs a harder opponent or a tighter reward to stay a real test. The interesting work is in the reward design, not the training loop.

Next


Evaluate the trained checkpoint against the fixed-policy baseline on held-out scenarios, for a clean before/after number
Make the vendor tougher so the model can't win by anchoring absurdly high
Tighten the reward so demanding $80M doesn't pay


Run it

bashprime env install fa1zvn/redline-negotiate
