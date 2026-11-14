from collections import Counter

from projects.public_bugs.pairs import balance


def test_balance_keeps_the_smaller_count_of_each_kind_in_order():
    rows = [{"kind": "real", "number": n} for n in range(7)] + [{"kind": "clean", "number": n} for n in range(3)]
    kept = balance(rows)
    assert Counter(row["kind"] for row in kept) == {"real": 3, "clean": 3}
    assert kept == [row for row in rows if row in kept]
    assert balance(rows) == kept
