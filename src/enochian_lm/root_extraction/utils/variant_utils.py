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
            (alt["value"], alt.get("type", "unknown"), alt.get("confidence", "low"))
            for alt in v["alternates"]
            if alt["direction"] in ["to", "both"]
        ]
        for k, v in subst_map.items()
    }

    def apply_subs(word, max_subs):
        results = set()
        queue = [(list(word), 0, 0, [])]  # (char_list, idx, subs_used, letter_names)

        while queue:
            chars, idx, subs_used, letter_names = queue.pop()

            if subs_used > max_subs:
                continue

            if idx >= len(chars):
                variant = "".join(chars)
                if return_subst_meta:
                    results.add((variant, subs_used, tuple(letter_names)))
                else:
                    results.add(variant)
                continue

            char = chars[idx]
            if char in sub_dict:
                for sub, sub_type, conf in sub_dict[char]:
                    is_letter_sub = sub_type == "letter_name" or conf == "low"
                    if is_letter_sub:
                        continue
                        # the below is to experiment with at a later date 6/3/2025
                        # if letter_names:
                        #     continue  # already used a letter_name, skip
                        # if subs_used + 1 > max_subs:
                        #     continue
                        # new_chars = chars[:idx] + [sub] + chars[idx + 1 :]
                        # queue.append(
                        #     (
                        #         new_chars,
                        #         idx + 1,
                        #         subs_used + 1,
                        #         letter_names + [sub.upper()],
                        #     )
                        # )
                    else:
                        if subs_used + 1 > max_subs:
                            continue
                        new_chars = chars[:idx] + [sub] + chars[idx + 1 :]
                        queue.append((new_chars, idx + 1, subs_used + 1, letter_names))

            # Always include the path with no substitution
            queue.append((chars, idx + 1, subs_used, letter_names))

        return results

    return list(apply_subs(word, max_subs))
