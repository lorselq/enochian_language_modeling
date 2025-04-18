import re
import datetime
from typing import Optional
from pathlib import Path
from collections import defaultdict
from enochian_translation_team.crew.root_extraction_crew import RootExtractionCrew

# Buffers for streaming
token_buffers = defaultdict(str)
log_entries = []


def save_log_to_md(log_entries):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp}_log.md"

    header_re = re.compile(r"^\*\*(.+?)\*\*:\s*", flags=re.MULTILINE)

    with open(log_path, "w", encoding="utf-8") as f:
        for role, message in log_entries:
            # Clean and format for Markdown
            header_re = re.compile(r"^\*\*(.+?):\*\*", re.MULTILINE)
            md = header_re.sub(lambda m: f"## {m.group(1)}", message, count=1)
            f.write(md.strip() + "\n\n")

    print(f"\n📁 Log saved to `{log_path}`")


def stream_callback(role, message):
    badge = {
        "Linguist": "👩‍💻",
        "Skeptic": "🤔",
        "Adjudicator": "👩‍⚖️",
        "Archivist": "📚",
        "Maestro": "🪄",
    }.get(role, "👤")

    is_first_token = role not in token_buffers or token_buffers[role] == ""

    # Distinguish prompts visually
    if message.strip().startswith(">>>"):
        # Make prompt italic and grayish
        formatted = f"\033[3;90m{message.replace('>>>', '')}\033[0m\n"
        print(f"{formatted}", end="", flush=True)
        return

    token_buffers[role] += message

    if is_first_token:
        print(f"\n{badge} {role}:\n", end="", flush=True)

    print(message, end="", flush=True)

    # Logging for markdown
    if not any(entry[0] == role for entry in log_entries):
        log_entries.append((role, ""))
    for i in range(len(log_entries)):
        if log_entries[i][0] == role:
            log_entries[i] = (role, token_buffers[role])

def main(max_words: Optional[int] = 1):
    print("🪄 Initializing semantic tribunal...\n")
    crew = RootExtractionCrew()
    crew.run_with_streaming(max_words=max_words, stream_callback=stream_callback)
    print("\n🎉 Crew has completed their assigned task.")
    save_log_to_md(log_entries)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run root extraction CLI.")
    parser.add_argument(
        "--max_words", type=int, default=3, help="Max words to process (0 for all)"
    )
    args = parser.parse_args()

    max_words = None if args.max_words == 0 else args.max_words
    main(max_words=max_words)
