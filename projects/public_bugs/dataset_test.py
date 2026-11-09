from projects.public_bugs.dataset import PORCELAIN_HEADER, changed_lines

HUNK = "@@ -10,4 +10,5 @@ def f():\n context\n-removed\n+added one\n+added two\n context\n context"


def test_changed_lines_are_new_side_numbers_of_added_lines():
    assert changed_lines([HUNK]) == [11, 12]


def test_changed_lines_skips_hunks_with_only_removals():
    assert changed_lines(["@@ -3,2 +3,1 @@\n keep\n-gone"]) == []


def test_porcelain_header_reads_sha_and_original_line():
    match = PORCELAIN_HEADER.match("a" * 40 + " 17 42 3")
    assert match.group(1) == "a" * 40
    assert int(match.group(2)) == 17
