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

buyer_offer  = 1_000_000
vendor_offer =   100_000
agreed = (buyer_offer + vendor_offer) // 2

print("They agreed at: $", agreed)
print("Scorekeeper grade:", score_outcome(agreed))
