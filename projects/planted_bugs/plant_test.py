from projects.planted_bugs.plant import plant, render, sites


def names(function):
    return [(i, name) for i, name, _, _ in sites(function)]


def test_comparisons_and_booleans_are_sites():
    function = [(1, "if (a < b && done == true)")]
    assert names(function) == [(0, "lt-le"), (0, "eq-ne"), (0, "and-or"), (0, "true-false")]


def test_strings_and_comments_are_not_sites():
    function = [(1, 'log("a < b"); // x == y'), (2, "// if (a < b)")]
    assert names(function) == []


def test_generics_are_not_comparisons():
    assert names([(1, "List<int> items = new List<int>();")]) == []


def test_plant_changes_one_line_and_keeps_its_number():
    function = [(1, "if (count > 0)"), (2, "    return x != null;")]
    site = next(s for s in sites(function) if s[1] == "ne-eq")
    assert plant(function, site) == [(1, "if (count > 0)"), (2, "    return x == null;")]


def test_drop_plus_one_removes_the_increment():
    function = [(1, "var last = index + 1;")]
    site = next(s for s in sites(function) if s[1] == "drop-plus-one")
    assert plant(function, site)[0][1] == "var last = index;"


def test_render_right_aligns_line_numbers():
    function = [(k, "x") for k in range(9, 11)]
    assert render(function) == " 9  x\n10  x"
