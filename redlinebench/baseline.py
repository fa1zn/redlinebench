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

# --- one full negotiation, returns the grade ---
def run_negotiation(vendor_start, total_rounds=3):
    vendor_offer = vendor_start
    for round_num in range(1, total_rounds + 1):
        our_offer = ai_offer(vendor_offer, round_num, total_rounds)
        if round_num < total_rounds:
            vendor_offer = vendor_offer + 100_000
    agreed = (our_offer + vendor_offer) // 2
    return score_outcome(agreed)

# --- the baseline: run several situations and average them ---
situations = [50_000, 100_000, 200_000, 300_000, 400_000]

grades = []
for start in situations:
    grade = run_negotiation(start)
    grades.append(grade)
    print(f"Vendor starts at ${start:,}  ->  grade {grade:.2f}")

average = sum(grades) / len(grades)
print()
print(f"BASELINE (average grade): {average:.2f}")
