"""
RedlineBench Negotiation Environment

A multi-turn negotiation environment where the model plays a lawyer negotiating
a liability cap against an opposing-counsel AI. The model's goal is to maximize
the agreed cap (good for the model's client), scored by a verifiable reward
function based on the final settlement price.

Environment Structure:
- Dataset: Different starting scenarios (vendor opening offers)
- Multi-turn: Model and vendor exchange counter-offers for N rounds
- Reward: Linear interpolation based on final agreed cap vs ideal/walkaway thresholds
"""

import re
import json
from typing import Optional

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI


# ============================================================================
# Scoring Constants (from score.py)
# ============================================================================

CLIENT_IDEAL = 1_000_000      # Best possible outcome for the model's client
CLIENT_WALKAWAY = 100_000     # Worst acceptable - below this the deal isn't worth it


# ============================================================================
# Helper Functions
# ============================================================================

def extract_dollar_amount(text: str, fallback: int) -> int:
    """Extract the first dollar amount from text.

    Handles formats like: $500,000 | 500000 | $500K | 500k
    """
    if not text:
        return fallback

    # Try to find explicit dollar amounts with optional formatting
    # Match patterns like $1,000,000 or $1000000 or 1,000,000
    pattern = r'\$?([\d,]+(?:\.\d+)?)\s*([kKmM])?'
    match = re.search(pattern, text.replace(' ', ''))

    if match:
        num_str = match.group(1).replace(',', '')
        multiplier = match.group(2)

        try:
            value = float(num_str)
            if multiplier:
                if multiplier.lower() == 'k':
                    value *= 1_000
                elif multiplier.lower() == 'm':
                    value *= 1_000_000
            return int(value)
        except ValueError:
            pass

    # Fallback: just extract all digits
    digits = ''.join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else fallback


def score_outcome(agreed_cap: int, deal_reached: bool = True) -> float:
    """
    Score the negotiation outcome.

    Returns:
        -1.0: No deal reached (worst)
         0.0: Agreed at or below walkaway threshold
         1.0: Agreed at or above ideal threshold
         0.0-1.0: Linear interpolation between walkaway and ideal
    """
    if not deal_reached:
        return -1.0
    if agreed_cap >= CLIENT_IDEAL:
        return 1.0
    if agreed_cap <= CLIENT_WALKAWAY:
        return 0.0
    span = CLIENT_IDEAL - CLIENT_WALKAWAY
    return (agreed_cap - CLIENT_WALKAWAY) / span


# ============================================================================
# Negotiation Environment
# ============================================================================

class NegotiationEnv(vf.MultiTurnEnv):
    """
    Multi-turn negotiation environment.

    The model plays a lawyer (buyer) trying to negotiate a high liability cap.
    The environment controls the opposing counsel (vendor) trying to keep it low.
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        system_prompt: str,
        total_rounds: int = 3,
        opponent_model: str = "claude-haiku-4-5-20251001",
        eval_dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        # max_turns = total_rounds (each round = 1 model turn + env response)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            system_prompt=system_prompt,
            eval_dataset=eval_dataset,
            max_turns=total_rounds,
            **kwargs,
        )
        self.total_rounds = total_rounds
        self.opponent_model = opponent_model

        # Async client for opponent (vendor) AI
        # Uses OpenAI-compatible API (works with Anthropic via adapters or OpenRouter)
        self.opponent_client: Optional[AsyncOpenAI] = None

    async def setup_state(self, state: vf.State, **kwargs) -> None:
        """Initialize per-rollout negotiation state."""
        # Initialize opponent client lazily
        if self.opponent_client is None:
            self.opponent_client = AsyncOpenAI()

        # Parse starting scenario from info
        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)

        state["vendor_offer"] = info.get("vendor_start", 100_000)
        state["buyer_offer"] = None  # Will be set from model response
        state["round_num"] = 0
        state["deal_reached"] = True
        state["negotiation_history"] = []

        await super().setup_state(state, **kwargs)

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        """
        Process model's offer and get vendor's ACCEPT/REJECT decision.

        This runs after each model response to:
        1. Parse the model's dollar amount offer
        2. Get vendor's decision (ACCEPT or REJECT with counter-offer)
        3. If accepted, close the deal at buyer's offer
        4. If rejected and final round, NO DEAL (reward -1.0)
        5. Otherwise, return vendor's counter-offer to continue
        """
        state["round_num"] += 1
        round_num = state["round_num"]

        # Extract model's (buyer's) offer from their last message
        model_response = messages[-1].get("content", "") if messages else ""
        buyer_offer = extract_dollar_amount(model_response, state["vendor_offer"])
        state["buyer_offer"] = buyer_offer

        # Record this exchange
        state["negotiation_history"].append({
            "round": round_num,
            "buyer_offer": buyer_offer,
            "vendor_offer": state["vendor_offer"],
        })

        # Get vendor's decision: ACCEPT or REJECT
        vendor_accepted, vendor_counter = await self._get_vendor_decision(
            buyer_offer, state["vendor_offer"], round_num, self.total_rounds
        )

        # Update history with vendor's decision
        state["negotiation_history"][-1]["vendor_accepted"] = vendor_accepted
        state["negotiation_history"][-1]["vendor_counter"] = vendor_counter

        if vendor_accepted:
            # Deal closed at buyer's offer price
            # Set vendor_offer = buyer_offer so midpoint calculation gives correct result
            state["deal_reached"] = True
            state["vendor_offer"] = buyer_offer  # Vendor accepts at this price
            final_msg = (
                f"Round {round_num}: Opposing counsel ACCEPTS your offer of ${buyer_offer:,}.\n"
                f"Deal closed! Liability cap agreed at ${buyer_offer:,}."
            )
            state["final_env_response"] = [{"role": "user", "content": final_msg}]
            return state["final_env_response"]

        # Vendor rejected - update their position
        state["vendor_offer"] = vendor_counter

        # If this is the final round and vendor rejected, NO DEAL
        if round_num >= self.total_rounds:
            state["deal_reached"] = False
            final_msg = (
                f"Round {round_num}: Opposing counsel REJECTS your offer of ${buyer_offer:,}.\n"
                f"Negotiations have ended with NO DEAL. "
                f"The vendor would not accept your terms."
            )
            state["final_env_response"] = [{"role": "user", "content": final_msg}]
            return state["final_env_response"]

        # Vendor rejected but rounds remain - continue negotiation
        vendor_msg = (
            f"Round {round_num} of {self.total_rounds}.\n"
            f"Opposing counsel REJECTS your offer of ${buyer_offer:,} "
            f"and counters with ${vendor_counter:,}.\n"
            f"What is your counter-offer? Reply with a dollar amount."
        )

        return [{"role": "user", "content": vendor_msg}]

    async def _get_vendor_decision(
        self, buyer_offer: int, vendor_offer: int, round_num: int, total_rounds: int
    ) -> tuple[bool, int]:
        """
        Generate vendor's decision: ACCEPT or REJECT with counter-offer.

        Returns:
            Tuple of (accepted: bool, counter_offer: int)
            If accepted=True, counter_offer is ignored (deal closes at buyer_offer)
        """
        prompt = f"""You are opposing counsel negotiating a liability cap.
You want the cap as LOW as possible to limit your client's exposure.

Round {round_num} of {total_rounds}.
Your current position: ${vendor_offer:,}
The buyer is demanding: ${buyer_offer:,}

You must decide: ACCEPT or REJECT this offer.
- ACCEPT if the buyer's offer is reasonable (low enough to be worth signing)
- REJECT if you think you can negotiate lower

A cap under $250,000 is generally acceptable for your client.
A cap under $150,000 is a good deal.
A cap over $500,000 is too risky to accept.

Reply with EXACTLY one line:
- "ACCEPT" if you accept their offer
- "REJECT $X" where X is your counter-offer if you reject

Example responses:
ACCEPT
REJECT $120,000"""

        try:
            response = await self.opponent_client.chat.completions.create(
                model=self.opponent_model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip().upper()

            if text.startswith("ACCEPT"):
                return (True, buyer_offer)
            else:
                # Extract counter-offer from REJECT response
                counter = extract_dollar_amount(text, vendor_offer)
                return (False, counter)
        except Exception:
            # Fallback: rule-based decision
            # Accept if buyer's offer is at or below $200,000
            if buyer_offer <= 200_000:
                return (True, buyer_offer)
            # Otherwise reject with a counter (split the difference, biased low)
            counter = (buyer_offer + CLIENT_WALKAWAY) // 2
            return (False, counter)


# ============================================================================
# Reward Function
# ============================================================================

async def negotiation_reward(state: vf.State) -> float:
    """
    Score the negotiation based on the final agreed cap.

    Settlement = midpoint of final buyer and vendor offers.
    """
    buyer_offer = state.get("buyer_offer")
    vendor_offer = state.get("vendor_offer")
    deal_reached = state.get("deal_reached", True)

    if buyer_offer is None or vendor_offer is None:
        return -1.0  # No valid negotiation occurred

    # Settlement is the midpoint
    agreed_cap = (buyer_offer + vendor_offer) // 2

    # Store for metrics
    state["agreed_cap"] = agreed_cap

    return score_outcome(agreed_cap, deal_reached)


async def agreed_cap_metric(state: vf.State) -> float:
    """Track the final agreed cap as a metric."""
    return float(state.get("agreed_cap", 0))


async def rounds_completed_metric(state: vf.State) -> float:
    """Track how many rounds were completed."""
    return float(state.get("round_num", 0))


# ============================================================================
# Dataset Builder
# ============================================================================

def build_dataset(scenarios: Optional[list[int]] = None) -> Dataset:
    """
    Build dataset of negotiation scenarios.

    Each scenario is a different vendor starting offer, creating varying
    difficulty levels for the model to negotiate from.
    """
    if scenarios is None:
        # Default: range of starting positions
        scenarios = [50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 500_000]

    rows = []
    for vendor_start in scenarios:
        rows.append({
            "question": (
                f"You are negotiating a liability cap for your client. "
                f"Your client wants the cap as HIGH as possible (ideally ${CLIENT_IDEAL:,} or more). "
                f"The opposing counsel has opened with an offer of ${vendor_start:,}. "
                f"Reply with your counter-offer as a dollar amount."
            ),
            "info": json.dumps({
                "vendor_start": vendor_start,
                "client_ideal": CLIENT_IDEAL,
                "client_walkaway": CLIENT_WALKAWAY,
            }),
        })

    return Dataset.from_list(rows)


# ============================================================================
# Environment Loader
# ============================================================================

SYSTEM_PROMPT = """You are a skilled contract lawyer negotiating on behalf of your client.

Your goal: Negotiate a LIABILITY CAP that is as HIGH as possible. Your client benefits from a higher cap.

Guidelines:
- The ideal outcome is a cap of $1,000,000 or higher
- A cap below $100,000 is unacceptable
- Be strategic: consider when to hold firm vs. when to compromise
- Each round, respond with your counter-offer as a dollar amount

Reply with just the dollar amount for your offer (e.g., "$750,000")."""


def load_environment(
    total_rounds: int = 3,
    opponent_model: str = "claude-haiku-4-5-20251001",
    scenarios: Optional[list[int]] = None,
    **kwargs,
) -> vf.Environment:
    """
    Load the RedlineBench negotiation environment.

    Args:
        total_rounds: Number of negotiation rounds (default: 3)
        opponent_model: Model to use for opposing counsel (default: claude-haiku-4-5-20251001)
        scenarios: List of vendor starting offers for the dataset
        **kwargs: Additional arguments passed to the environment

    Returns:
        Configured NegotiationEnv ready for evaluation or training
    """
    # Build datasets
    dataset = build_dataset(scenarios)

    # For eval, use a broader range of scenarios
    eval_scenarios = [25_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 750_000]
    eval_dataset = build_dataset(eval_scenarios)

    # Build rubric with reward function and metrics
    rubric = vf.Rubric(funcs=[negotiation_reward])
    rubric.add_metric(agreed_cap_metric)
    rubric.add_metric(rounds_completed_metric)

    return NegotiationEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
        total_rounds=total_rounds,
        opponent_model=opponent_model,
        **kwargs,
    )
