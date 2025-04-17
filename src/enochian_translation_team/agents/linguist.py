class Linguist:
    def analyze(self, root, cluster) -> dict:
        return {
            "root": root,
            "hypothesis": f"The root '{root}' appears to group words with related meanings.",
            "justification": f"{len(cluster)} candidates with shared gloss elements.",
            "score": round(sum(c.get("semantic", 0) for c in cluster) / len(cluster), 3),
        }
