import argparse
import os
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict
from enochian_lm.root_extraction.utils.local_env_refresher import refresh_local_env
from enochian_lm.root_extraction.pipeline.run_root_extraction import RootExtractionCrew


# Buffers for streaming
token_buffers = defaultdict(str)
log_entries = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Enochian root extraction pipeline with optional skip replay"
        )
    )
    parser.add_argument(
        "--only-skipped",
        nargs="?",
        const="",
        default=None,
        help=(
            "Process only n-grams previously recorded in the skips table. "
            "Optionally provide a specific reason_code to filter the replay."
        ),
    )
    return parser.parse_args()


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
    args = parse_args()
    GOLD = "\033[38;5;178m"
    RESET = "\033[0m"
    local_remote_mode = None
    remote = True
    while local_remote_mode not in ("1", "2"):
        local_remote_mode = input("Do you want to use a local LLM with LM Studio (1) or a remote LLM through OpenRouter (2)? ")
    if local_remote_mode == "1" or local_remote_mode == "2":
        if(refresh_local_env()):
            env_local = find_dotenv(".env_local")
            env_remote = find_dotenv(".env_remote")
            load_dotenv(env_local, override=True)
            load_dotenv(env_remote, override=True)
            if local_remote_mode == "1":
                remote = False
        else:
            print("[Error] Could not load environment file for local LLM connection. Exiting.")
            return
    else:
        load_dotenv(".env_remote", override=True)
        
    mode = None
    while mode not in ("1", "2"):
        mode = input("Do you want to eval a specific ngram (1) or evaluate a number of ngrams (2)? ")

    style = None
    while style not in ("1", "2"):
        style = input("Do you want each ngram to be debated (1) or analyzed in a single pass (2)? ")
        
    if style == "1":
        style = "debate"
    else:
        style = "solo"

    process_only_skipped = args.only_skipped is not None
    skipped_reason_code = args.only_skipped or None

    crew = RootExtractionCrew(style, remote)

    if mode == "1":
        ngram = input("Which ngram do you want to evaluate? ").strip().lower()
        print(f"🔍 Evaluating single ngram: {GOLD}{ngram.upper()}{RESET}\n")
        crew.process_ngrams(
            single_ngram=ngram,
            stream_callback=stream_callback,
            style=style,
        )

    else:
        max_words = None
        while max_words is None:
            try:
                max_words_input = input(
                    "How many root words should I process? (0 for all of them): "
                )
                max_words = int(max_words_input)
            except ValueError:
                print("Invalid number. Please use a digit.")
        print(f"🔍 Evaluating {GOLD}{max_words}{RESET} ngrams...")
        crew.process_ngrams(
            max_words=max_words,
            stream_callback=stream_callback,
            style=style,
            process_only_skipped=process_only_skipped,
            skipped_reason_code=skipped_reason_code,
        )

    if style == "solo":
        print("\n\n🎉 The researcher has completed their assigned task(s)!")        
    else:    
        print("\n\n🎉 The research team has completed their assigned task(s)!")



if __name__ == "__main__":
    main()
