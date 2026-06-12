import anthropic

client = anthropic.Anthropic()

# --- the scorekeeper (unchanged) ---
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

# --- helper: pull the first dollar number out of AI text ---
def extract_number(text, fallback):
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else fallback

# --- our AI side (the buyer): wants the cap HIGH ---
def buyer_offer(vendor_offer, round_num, total_rounds):
    prompt = f"""You are negotiating a liability cap for your client.
Your client wants the cap as HIGH as possible.
Round {round_num} of {total_rounds}. The vendor just offered ${vendor_offer}.
Reply with ONLY a dollar number for your counter-offer, no words."""
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_number(msg.content[0].text, vendor_offer)

# --- the vendor (NEW: a real AI): wants the cap LOW ---
def vendor_counter(buyer_offer_amount, round_num, total_rounds):
    prompt = f"""You are opposing counsel negotiating a liability cap.
You want the cap as LOW as possible to limit your exposure.
Round {round_num} of {total_rounds}. The buyer just demanded ${buyer_offer_amount}.
Hold firm when you can. Reply with ONLY a dollar number for your counter, no words."""
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_number(msg.content[0].text, buyer_offer_amount)

# --- the back-and-forth: now BOTH sides are AI ---
TOTAL_ROUNDS = 3
vendor_offer = 100_000   # vendor opens low

for round_num in range(1, TOTAL_ROUNDS + 1):
    buyer = buyer_offer(vendor_offer, round_num, TOTAL_ROUNDS)
    vendor_offer = vendor_counter(buyer, round_num, TOTAL_ROUNDS)
    print(f"Round {round_num}: buyer ${buyer:,}  |  vendor ${vendor_offer:,}")

# they settle in the middle of the final two offers
agreed = (buyer + vendor_offer) // 2
print()
print(f"They settle at: ${agreed:,}")
print("Scorekeeper grade:", round(score_outcome(agreed), 2))

