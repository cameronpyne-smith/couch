from projects.public_bugs.minimal_pairs import fix_lines, old_lines, parts, signature


def test_fix_lines_are_the_added_lines():
    assert fix_lines(["@@ -10,2 +10,3 @@\n keep\n+added\n keep"]) == [11]


def test_fix_lines_fall_back_to_the_hunk_start_when_the_fix_only_removes():
    assert fix_lines(["@@ -3,2 +7,1 @@\n keep\n-gone"]) == [7]


def test_signature_ignores_indentation():
    assert signature("    def f(a):\n        return a") == signature("def f(a):\n    return a + 1")


def test_parts_measures_function_and_diff_separately():
    assert parts("abcd\n\n### Changes to this file in the pull request\n```diff\nxy\n```") == (4, 14)


def test_old_lines_cover_the_old_side_of_each_hunk():
    assert old_lines(["@@ -3,2 +7,1 @@\n keep\n-gone", "@@ -20 +19,2 @@\n+new\n keep"]) == [3, 4, 20]
