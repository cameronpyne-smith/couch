import math

import torch

from finetune.verdict import choice_ids, choice_probability, marker_step, parse_verdict, verdict_probability

VOCAB = ["Ver", "dict", ":", " break", " no", "break", "no", " ", " something"]


class Tokeniser:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(VOCAB[i] for i in ids)

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            input_ids = [VOCAB.index(text)] if text in VOCAB else [VOCAB.index(text.strip())]

        return Encoded()


def logits_for(token_id, size=len(VOCAB), high=5.0):
    logits = torch.zeros(size)
    logits[token_id] = high
    return logits


def test_marker_step_points_at_the_token_after_the_marker():
    ids = [VOCAB.index(" something"), 0, 1, 2, 3]
    assert marker_step(Tokeniser(), ids, "Verdict:") == 4


def test_marker_step_none_when_marker_missing():
    assert marker_step(Tokeniser(), [VOCAB.index(" something")], "Verdict:") is None


def test_choice_probability_normalises_between_the_two_sides():
    probs = torch.log(torch.tensor([0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.05, 0.05]))
    yes = choice_ids(Tokeniser(), "break")
    no = choice_ids(Tokeniser(), "no")
    assert math.isclose(choice_probability(probs, yes, no), 0.4 / 0.6, rel_tol=1e-6)


def test_verdict_probability_reads_the_score_at_the_choice_step():
    ids = [0, 1, 2, VOCAB.index(" break")]
    scores = [logits_for(0)[None], logits_for(1)[None], logits_for(2)[None], logits_for(VOCAB.index(" break"))[None]]
    p = verdict_probability(Tokeniser(), ids, scores, "Verdict:", "break", "no")
    assert p > 0.95


def test_verdict_probability_skips_a_lone_space_token():
    ids = [0, 1, 2, VOCAB.index(" "), VOCAB.index("no")]
    scores = [logits_for(i)[None] for i in ids]
    p = verdict_probability(Tokeniser(), ids, scores, "Verdict:", "break", "no")
    assert p < 0.05


def test_parse_verdict_prefers_the_longer_choice():
    assert parse_verdict("reasoning\n\nVerdict: no break\n", "Verdict:", ("break", "no break")) == "no break"
    assert parse_verdict("Verdict: break\n\nfinding", "Verdict:", ("break", "no break")) == "break"
    assert parse_verdict("never concluded", "Verdict:", ("break", "no break")) is None
