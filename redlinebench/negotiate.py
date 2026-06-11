import anthropic

client = anthropic.Anthropic()

# --- the scorekeeper ---
CLIENT_IDEAL    = 1_000_000
CLIENT_WALKAWAY =   100_000

def score_outcome(agreed_cap, deal_reached=True):
    if not deal_reached:
        return -1.0
    if agreed_cap >= CLIENT_IDEAL:
        return 1.0
    if agreed_cap <= CLIENT_WALKAWAY:
        return 0.0
    span = CLIENT_IDEAL - CLIENT_WALKAWAY
    return (agreed_cap - CLIENT_WALKAWAY) / span

# --- our AI side (the buyer) ---
def ai_offer(their_offer, round_num, total_rounds):
    prompt = f"""You are negotiating a liability cap for your client.
Your client wants the cap as HIGH as possible.
This is round {round_num} of {total_rounds}.
The other side just offered ${their_offer}.
Reply with ONLY a dollar number for your counter-offer, no words."""
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else their_offer

# --- the back-and-forth ---
TOTAL_ROUNDS = 3
vendor_offer = 100_000   # vendor starts low

for round_num in range(1, TOTAL_ROUNDS + 1):
    our_offer = ai_offer(vendor_offer, round_num, TOTAL_ROUNDS)
    print(f"Round {round_num}: vendor ${vendor_offer:,}  |  our AI ${our_offer:,}")
    if round_num < TOTAL_ROUNDS:
        vendor_offer = vendor_offer + 100_000   # vendor concedes upward each round

# they settle in the middle of the final two offers
agreed = (our_offer + vendor_offer) // 2
print()
print(f"They settle at: ${agreed:,}")
print("Scorekeeper grade:", score_outcome(agreed))
