class Skeptic:
    def challenge(self, hypothesis: dict, cluster: list) -> dict:
        low_score_items = [c for c in cluster if c.get("semantic", 0) < 0.3]
        return {
            "counter": f"{len(low_score_items)} items in the cluster have weak semantic alignment.",
            "low_count": len(low_score_items),
        }
