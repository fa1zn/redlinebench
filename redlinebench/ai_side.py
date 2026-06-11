import anthropic

client = anthropic.Anthropic()

def ai_offer(their_offer):
    prompt = f"""You are negotiating a liability cap for your client.
Your client wants the cap as HIGH as possible.
The other side just offered ${their_offer}.
Reply with ONLY a dollar number, no words."""
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

print("Vendor offered: $100,000")
print("AI counter-offer:", ai_offer(100_000))
