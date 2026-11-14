from collections import Counter

from projects.public_bugs.pairs import balance, hold_out, pair_rows_by_content


def test_balance_keeps_the_smaller_count_of_each_kind_in_order():
    rows = [{"kind": "real", "number": n} for n in range(7)] + [{"kind": "clean", "number": n} for n in range(3)]
    kept = balance(rows)
    assert Counter(row["kind"] for row in kept) == {"real": 3, "clean": 3}
    assert kept == [row for row in rows if row in kept]
    assert balance(rows) == kept


def test_content_pairs_join_a_matching_answer_with_one_that_does_not():
    example = {"system": "s", "prompt": "p", "kind": "real", "repo": "r", "number": 1}
    answer = {"kind": "real", "repo": "r", "number": 1, "samples": [{"verdict": "break", "text": "a"}, {"verdict": "no break", "text": "b"}, {"verdict": "break", "text": "c"}, {"verdict": "break", "text": "d"}]}
    rows = pair_rows_by_content([answer, answer], [example, example], {0: {"0": "no", "2": "match", "3": "no"}})
    assert rows == [example | {"chosen": "c", "rejected": "a"}]


def test_hold_out_keeps_every_pair_of_a_function_on_one_side():
    rows = [{"kind": "real", "repo": "r", "number": n // 2} for n in range(40)]
    train, held = hold_out(rows, 0.1)
    assert len(held) == 4 and len(train) == 36
    assert not {row["number"] for row in train} & {row["number"] for row in held}
