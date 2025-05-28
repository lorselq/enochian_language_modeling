from itertools import product, combinations


def normalize_word(word, subst_map):
    normalized = ""
    for char in word:
        if char in subst_map and any(
            alt["direction"] in ["from", "both"]
            for alt in subst_map[char]["alternates"]
        ):
            normalized += subst_map[char]["canonical"]
        else:
            normalized += char
    return normalized


def apply_sequence_compressions(word, compression_rules):
    for rule in compression_rules:
        if rule["direction"] in ["from", "both"] and rule["from"] in word:
            word = word.replace(rule["from"], rule["to"])
    return word


def generate_variants(word, subst_map, max_subs=2):
    word = word.lower()
    variants = set()
    variants.add(word)

    # Build simple substitution lookup
    sub_dict = {
        k: [
            alt["value"]
            for alt in v["alternates"]
            if alt["direction"] in ["to", "both"]
        ]
        for k, v in subst_map.items()
    }

    for positions in combinations(range(len(word)), max_subs):
        replacement_options = []
        for i in positions:
            char = word[i]
            replacement_options.append(sub_dict.get(char, [char]))

        for replacements in product(*replacement_options):
            temp = list(word)
            for idx, sub in zip(positions, replacements):
                temp[idx] = sub
            variant = "".join(temp)
            variants.add(variant)
    return list(variants)
