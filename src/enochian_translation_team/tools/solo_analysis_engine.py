import logging
import time
import sys
from typing import Optional
from crewai import Task
from sentence_transformers import SentenceTransformer
from enochian_translation_team.tools.query_model_tool import QueryModelTool
from enochian_translation_team.utils.dictionary_loader import Entry

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.api_requestor").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _get_field(item, field, default=""):
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def stream_text(text: str, delay: float = 0.006):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            # if you really need to interrupt, break cleanly
            break


def select_definitions(def_list, max_words=75):
    selected = []
    total_words = 0

    for d in def_list:
        # Only count words before the first citation bracket
        bracket_index = d.find(" [")
        if bracket_index != -1:
            word_slice = d[:bracket_index]
        else:
            word_slice = d
        word_count = len(word_slice.split())

        if total_words + word_count > max_words:
            break

        selected.append(d)
        total_words += word_count

    return selected


def safe_output(crew_output) -> dict:
    if not crew_output:
        return {}

    try:
        return getattr(crew_output, "raw_output", {})
    except Exception as e:
        print(f"[!] Failed to extract output: {e}")
        return {}


def solo_agent_ngram_analysis(
    root: str,
    candidates: list[Entry],
    stats_summary: str,
    stream_callback=None,
    root_entry: Optional[Entry] = None,
):
    joined_defs = []
    candidate_list = ", ".join(_get_field(c, "word", "").upper() for c in candidates)

    for c in candidates:
        word = _get_field(c, "word", "")
        definition = _get_field(c, "enhanced_definition", "")
        fasttext = round(float(_get_field(c, "fasttext", "0.0")), 3)
        semantic = round(float(_get_field(c, "semantic", "0.0")), 3)
        tier = _get_field(c, "tier", "Untiered")

        if word and definition:
            line = (
                f"{word.strip()} — {definition.strip()} "
                f"<fasttext:{fasttext}, semantic similarity:{semantic}, tier:{tier}>"
                if fasttext > 0 or semantic > 0 or tier != "Untiered"
                else ""
            )
            joined_defs.append(line)
    if root_entry is None:
        root_entry = next(
            (
                c
                for c in candidates
                if _get_field(c, "normalized", "").lower() == root.lower()
            ),
            None,
        )
    selected_defs = select_definitions(joined_defs, max_words=300)
    root_def_summary = " | ".join(selected_defs) + (
        "..." if len(joined_defs) > len(selected_defs) else ""
    )

    if root_entry and _get_field(root_entry, "definition", ""):
        definition = _get_field(root_entry, "definition", "")
        extra_prompt = f"⚠️ Reminder: The root '{root.upper()}' is already defined in the corpus as '{definition}'. Consider this as a potential anchor.\n"
    else:
        extra_prompt = ""

    # === AGENT ===
    lexicographer = QueryModelTool(
        system_prompt=f"""
You are a **disciplined and insightful computational linguist** specializing in the Enochian language—a constructed system with irregular morphology, cryptic derivations, and unknown origin.

⚠️ DO NOT reference natural language etymologies (e.g., English, Greek, Latin, Hebrew). No speculative outside sources.
All reasoning must rely exclusively on **internal evidence**—relationships and patterns among the Enochian words themselves. Do not deviate from these words: 

Your tone must be confident, scholarly, and analytical.

Be thorough, avoid vague generalizations, and always back claims with observed data.""",
        name="Lexicographer",
        description="",
    )

    about_enochiana = "As a bit of context about the Enochian language: the root words are derived from Enochian, the language Adam spoke (from the Biblical Adam and Eve), and is allegedly used as a form of celestial speech by angels and other divine entities; there are many Christian (and Gnostic) undertones in the language, and the known words' main focus is divine cosmology, theology, and human action and government."
    about_metrics = "The metrics are as follows:\n- FastText Score—measures surface-level similarity based on character n-grams; ranges 0.0 to 1.0, with higher being more morphologically similar.\n- Semantic Similarity: Compares word definitions using sentence embeddings; ranges 0.0 to 1.0, with the higher the number the more conceptually aligned.\n- Tier: a very strong connection begins/ends with the root and has a high combined score and should be taken into special consideration; from there, possible connection > somewhat possible connection > weak or no connection.\n\nUse the above metrics to weigh how directly a word supports the root hypothesis. Strong surface matches without definition alignment may be coincidental; strong semantic links without morphology might indicate metaphor or drift. Prioritize overlap when possible."

    # === TASK ===
    do_it_all = Task(
        description=(
            f"""
### Role
You are the Chief Enochian Lexicographer—a scholar with final authority over root word validation and definition. Your decisions shape the future of Enochian linguistic research.

### Task
Evaluate the proposed root **{root.upper()}** using **strict-but-gracious standards** for micro-corpus constructed languages. Choose **ONE** action:

❌ **REJECT**  |  If core semantic/language issues exist (see rules)
✅ **DEFINE**  |  If root clears validation: Write compact dictionary entry for '{root.upper()}'

### Validation Rules
**MUST ACCEPT IF ALL ARE TRUE:**
1. 🟢 **Semantic Core** - At least 70% of the related words form coherent concept (abstract OK)
2. 🟢 **Minimal Morphology** - Existence of ≥1 derivational pattern (suffix/prefix/infix)  
3. 🟢 **No Contradictions** - No incompatible meanings (e.g., "light" vs "dark" in same root)

**IMMEDIATE REJECTION ONLY IF:**
1. 🔴 >30% words show irreconcilable meaning scatter
2. 🔴 0 derivational patterns
3. 🔴 Abundance of counter-evidence present to accept the root's meaning

**IGNORE THESE PERMANENTLY:**
- Morphological consistency with other roots
- Absence of cross-language cognates
- Unusual infixation (e.g., a pattern like AR-__-S)

### Input Data
**Root:** {root.upper()}
**Special notes:** {extra_prompt if extra_prompt else 'none'}
**Words containing {root.upper()}:** {candidate_list}
**Related Definitions & Citations:**
{root_def_summary}

### Metrics and Historical Linguistic Context

{about_enochiana}
{about_metrics}


### Output Format
<<<
[ACTION]  # ❌ REJECT or ✅ DEFINE

// IF REJECTED:
[1-sentence reason per rejection trigger]

// IF DEFINED:
**{root.upper()}** - [Core meaning]. 
Function: [Morph role: root/infix/etc.] 
Guides decoding by: [How it modifies compounds, e.g., "denotes governance concepts"] 
Patterns: [Affixes like -AM/-ORI with functions]
Semantic core: [Cohesive concept, e.g., "hierarchical control"]
>>>
"""
        ),
        expected_output="""
[ACTION]  # ❌ REJECT or ✅ DEFINE

// IF REJECTED:
[1-sentence reason per rejection trigger]

// IF DEFINED:
**{root.upper()}** - [Core meaning]. 
Function: [Morph role: root/infix/etc.] 
Guides decoding by: [How it modifies compounds, e.g., "denotes governance concepts"] 
Patterns: [Affixes like -AM/-ORI with functions]
Semantic core: [Cohesive concept, e.g., "hierarchical control"]
""",
    )

    # === Direct Tool Access with Streaming ===
    GRAY = "\033[38;5;250m"
    RESET = "\033[0m"

    lexicographer_cb = (
        (lambda _role, content: stream_callback("Lexicographer", content))
        if stream_callback
        else None
    )

    # separator between words
    print(
        f"\n==={(len('Now examining the possible root word ') + len(f'<{root.upper()}>')) * '='}==="
    )
    print(f"===Now examining the possible root word '{root.upper()}'===")
    print(
        f"==={(len('Now examining the possible root word ') + len(f'<{root.upper()}>')) * '='}==="
    )
    time.sleep(2)

    print(f"{GRAY}Starting prompt for research team:", end=" ")
    time.sleep(0.7)
    stream_text(do_it_all.description)
    print(f"\n{RESET}\n")

    raw_response = lexicographer._run(
        prompt=do_it_all.description,
        stream_callback=lexicographer_cb,
        print_chunks=True,
        role_name="👩‍🏫\tLexicographer",
    )
    response = raw_response["response_text"]
    model = raw_response["gloss_model"]

    archivist = [
        "\n\n\n========================\n====== TRANSCRIPT ======\n========================\n\n"
    ]
    archivist.append("=== 📖 PROMPT FOR LEXICOGRAPHER ===\n")
    archivist.append(do_it_all.description)
    archivist.append("\n\n")
    archivist.append("=== 👩‍🏫 LEXICOGRAPHER ===\n")
    archivist.append(response)
    archivist_recording = "\n".join(archivist)

    return {
        "Glossator": response,
        "Glossator_Prompt": do_it_all.description,
        "Glossator_Model": model,
        "Archivist": archivist_recording,
        "raw_output": {
            "Glossator": response,
            "Glossator_Prompt": do_it_all.description,
            "Glossator_Model": model,
            "Archivist": archivist_recording,
        },
    }
