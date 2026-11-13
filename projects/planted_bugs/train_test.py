from projects.planted_bugs.train import auc, named_line, pick_samples


def row(pair, repo, bug_line, mutation):
    return {"pair": pair, "repo": repo, "bug_line": bug_line, "mutation": mutation}


def test_named_line_reads_the_number_and_nothing_else():
    assert named_line("12") == 12
    assert named_line(" 7\n") == 7
    assert named_line("none") is None
    assert named_line("") is None


def test_auc_is_the_share_of_planted_scored_above_clean_with_ties_as_half():
    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auc([0.1, 0.9], [True, False]) == 0.0
    assert auc([0.5, 0.5], [True, False]) == 0.5


def test_pick_samples_returns_both_rows_of_each_chosen_pair():
    val = [row(0, "a", None, None), row(0, "a", 3, "eq-ne"), row(1, "b", None, None), row(1, "b", 5, "lt-le")]
    assert pick_samples(val, 1) == [0, 1]
    assert pick_samples(val, 2) == [0, 1, 2, 3]


def test_pick_samples_prefers_mutations_not_yet_shown():
    val = [row(0, "a", 3, "eq-ne"), row(1, "b", 4, "eq-ne"), row(2, "b", 5, "lt-le")]
    assert pick_samples(val, 2) == [0, 2]
