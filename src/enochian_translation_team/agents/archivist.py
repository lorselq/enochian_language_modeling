class Archivist:
    def summarize(self, root, hypothesis, critique, verdict):
        status = verdict["verdict"].capitalize()
        score = hypothesis.get("score", 0.0)
        weak = critique.get("low_count", 0)

        return f"{status}: Root '{root}' scored {score:.2f}. {hypothesis['justification']} Skeptic flagged {weak} weak items. Adjudicator: {verdict['notes']}"
