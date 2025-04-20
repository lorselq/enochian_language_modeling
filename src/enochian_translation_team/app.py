import argparse
from collections import defaultdict
from enochian_translation_team.crew.root_extraction_crew import RootExtractionCrew

# Buffers for streaming
token_buffers = defaultdict(str)
log_entries = []


def stream_callback(role, message):
    badge = {
        "Linguist": "🥸",
        "Skeptic": "🤔",
        "Adjudicator": "👩‍⚖️",
        "Glossator": "🧐",
        "Archivist": "📚",
        "Maestro": "🪄",
    }.get(role, "👤")

    is_first_token = not token_buffers[role]

    # Special formatting for prompt-like starter messages
    if message.strip().startswith(">>>"):
        formatted = f"\033[3;90m{message.replace('>>>', '').strip()}\033[0m\n"
        print(formatted, end="", flush=True)
        return

    token_buffers[role] += message

    if is_first_token:
        print(f"\n{badge} {role}:\n", end="", flush=True)

    print(message, end="", flush=True)

    # Update log entries for markdown
    for i, (r, text) in enumerate(log_entries):
        if r == role:
            log_entries[i] = (role, token_buffers[role])
            break
    else:
        log_entries.append((role, token_buffers[role]))


def main():
    max_words = None
    while max_words is None:
        try:
            max_words_input = input(
                "How many root words should I process? (0 for all of them): "
            )
            max_words = int(max_words_input)
        except ValueError:
            print("Invalid number. Try typing a digit.")

    print("🪄 Initializing semantic tribunal...\n")
    crew = RootExtractionCrew()
    crew.run_with_streaming(max_words=max_words, stream_callback=stream_callback)
    print("\n\n🎉 The research team has completed their assigned task(s)!")


if __name__ == "__main__":
    main()
