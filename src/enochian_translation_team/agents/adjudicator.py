class Adjudicator:
    def decide(self, hypothesis: dict, critique: dict) -> dict:
        decision = "accept" if hypothesis["score"] > 0.5 and critique["low_count"] < 2 else "reject"
        return {
            "verdict": decision,
            "confidence": hypothesis["score"],
            "notes": f"Accepted based on strong cluster, rejected due to {critique['low_count']} weak items." if decision == "reject" else "Strong semantic cohesion."
        }
