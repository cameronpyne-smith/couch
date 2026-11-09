import re

import torch


def marker_step(tokeniser, generated_ids, marker):
    for step in range(len(generated_ids)):
        if tokeniser.decode(generated_ids[: step + 1], skip_special_tokens=True).rstrip().endswith(marker):
            return step + 1
    return None


def choice_ids(tokeniser, word):
    return {tokeniser(form, add_special_tokens=False).input_ids[0] for form in (" " + word, word)}


def choice_probability(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    yes = sum(probs[i].item() for i in yes_ids)
    no = sum(probs[i].item() for i in no_ids)
    return yes / (yes + no) if yes + no > 0 else float("nan")


def parse_verdict(text, marker, choices):
    match = re.search(re.escape(marker) + r"\s*(" + "|".join(re.escape(c) for c in sorted(choices, key=len, reverse=True)) + ")", text)
    return match.group(1) if match else None


def verdict_probability(tokeniser, generated_ids, scores, marker, yes, no):
    step = marker_step(tokeniser, generated_ids, marker)
    if step is None or step >= len(scores):
        return None
    while step < len(scores) - 1 and not tokeniser.decode([generated_ids[step]]).strip():
        step += 1
    return choice_probability(scores[step][0], choice_ids(tokeniser, yes), choice_ids(tokeniser, no))
