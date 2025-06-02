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


def generate_variants(word, subst_map, max_subs=3, return_subst_meta=False):
    word = word.lower()
    variants = set()
    variants.add((word, 0, ())) if return_subst_meta else variants.add(word)

    sub_dict = {
        k: [
            (alt["value"], alt.get("type", "unknown"))
            for alt in v["alternates"]
            if alt["direction"] in ["to", "both"]
        ]
        for k, v in subst_map.items()
    }

    # Generate positions for possible substitutions
    for n_subs in range(1, max_subs + 1):
        for positions in combinations(range(len(word)), n_subs):
            replacement_sets = []
            valid = True
            for i in positions:
                char = word[i]
                if char not in sub_dict:
                    valid = False
                    break
                replacement_sets.append(sub_dict[char])
            if not valid:
                continue

            for replacements in product(*replacement_sets):
                temp = list(word)
                letter_names = []
                for idx, (sub, sub_type) in zip(positions, replacements):
                    temp[idx] = sub
                    if sub_type == "letter_name":
                        letter_names.append(sub.upper())
                variant = "".join(temp)
                if return_subst_meta:
                    variants.add((variant, n_subs, tuple(letter_names)))
                else:
                    variants.add(variant)

    return list(variants)
