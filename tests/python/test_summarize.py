"""ci/summarize.py: counting warnings, reading ctest results, annotating changed lines."""

from conftest import import_script

summarize = import_script("ci/summarize.py")

ROOT = "/work/repo"
BUILD = "/work/repo/build"

LOG = """\
[ 10%] Building CXX object src/CMakeFiles/src.dir/eos.cpp.o
/work/repo/src/eos.cpp:12:7: warning: unused variable 'x' [-Wunused-variable]
/work/repo/include/kernel.h:40:3: warning: comparison of integer expressions of different signedness: 'int' and 'unsigned int' [-Wsign-compare]
/work/repo/include/kernel.h:40:3: warning: comparison of integer expressions of different signedness: 'int' and 'unsigned int' [-Wsign-compare]
../unit_tests/vector_test.cpp:5:1: warning: no flag on this one
/opt/deps/include/Cabana_Utils.hpp:35:1: warning: 'x' is deprecated [-Wdeprecated-declarations]
/work/repo/build/_deps/googletest-src/googletest/src/gtest.cc:9:9: warning: something [-Wextra]
/work/repo/src/eos.cpp:20:1: error: this is not a warning
/work/repo/src/eos.cpp:30:5: note: declared here
"""

JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Linux" tests="4" failures="1" disabled="0" skipped="1" time="3">
  <testcase name="KernelTest.A" classname="KernelTest.A" time="0.5" status="run"/>
  <testcase name="KernelTest.B" classname="KernelTest.B" time="1.5" status="fail">
    <failure message=""/>
    <system-out>first line
Expected 1, got 2</system-out>
  </testcase>
  <testcase name="smoke" classname="smoke" time="1.0" status="run"/>
  <testcase name="python_tests" classname="python_tests" time="0.0" status="notrun">
    <skipped message=""/>
  </testcase>
</testsuite>
"""

DIFF = """\
diff --git a/src/eos.cpp b/src/eos.cpp
--- a/src/eos.cpp
+++ b/src/eos.cpp
@@ -10,0 +11,3 @@ void f()
+int x;
+int y;
+int z;
@@ -40 +44 @@ void g()
-old
+new
diff --git a/old.txt b/old.txt
--- a/old.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-gone
-gone
"""


def parsed():
    return summarize.parse_warnings(LOG, ROOT, BUILD)


def test_warnings_are_unique_and_classified():
    warnings = parsed()
    assert len(warnings) == 5  # the repeated one counts once
    own = [w for w in warnings if w.own]
    assert [(w.path, w.line, w.flag) for w in own] == [
        ("src/eos.cpp", 12, "-Wunused-variable"),
        ("include/kernel.h", 40, "-Wsign-compare"),
        ("unit_tests/vector_test.cpp", 5, "(no flag)"),  # relative to the build directory
    ]
    # dependency headers and downloaded sources in the build directory are not CCAKE's
    assert sorted(w.path for w in warnings if not w.own) == [
        "/opt/deps/include/Cabana_Utils.hpp",
        "/work/repo/build/_deps/googletest-src/googletest/src/gtest.cc",
    ]


def test_errors_and_notes_are_not_warnings():
    assert all("not a warning" not in w.message for w in parsed())
    assert all(w.line != 30 for w in parsed())


def test_junit(tmp_path):
    path = tmp_path / "result.xml"
    path.write_text(JUNIT)
    run = summarize.parse_junit(path, "all threads")
    assert (run.total, run.failed, run.skipped) == (4, 1, 1)
    assert run.seconds == 3.0
    assert run.failures == [("KernelTest.B", "first line\nExpected 1, got 2")]


def test_changed_lines():
    changed = summarize.changed_lines(DIFF)
    assert changed["src/eos.cpp"] == {11, 12, 13, 44}
    assert "old.txt" not in changed  # a deleted file has no new lines


def test_annotations_only_on_changed_lines():
    commands, count = summarize.annotations(parsed(), summarize.changed_lines(DIFF), limit=30)
    assert count == 1
    assert commands == [
        "::warning file=src/eos.cpp,line=12,col=7,title=-Wunused-variable::unused variable 'x'"
    ]


def test_annotations_are_limited():
    warnings = [summarize.Warning("src/a.cpp", n, 1, "-Wx", "m", True) for n in range(1, 6)]
    commands, count = summarize.annotations(warnings, {"src/a.cpp": set(range(1, 6))}, limit=2)
    assert count == 5
    assert len(commands) == 3 and commands[-1].startswith("::notice::3 more")


def test_escape():
    assert summarize.escape("50% done\nnext") == "50%25 done%0Anext"
    assert summarize.escape("a:b,c", is_property=True) == "a%3Ab%2Cc"


def test_render(tmp_path):
    path = tmp_path / "result.xml"
    path.write_text(JUNIT)
    text = summarize.render("gcc-13", parsed(), [summarize.parse_junit(path, "all threads")], 1)
    assert "## gcc-13" in text
    assert "| all threads | FAIL | 4 | 2 | 1 | 1 | 3.0 s |" in text
    assert "failed: <code>KernelTest.B</code>" in text
    assert "| on lines changed by this pull request | 1 |" in text
    assert "| in CCAKE files | 3 |" in text
    assert "| in dependency headers | 2 |" in text


def test_main_writes_a_summary(tmp_path):
    log = tmp_path / "build.log"
    log.write_text(LOG)
    junit = tmp_path / "result.xml"
    junit.write_text(
        JUNIT.replace('status="fail"', 'status="run"').replace('<failure message=""/>', "")
    )
    output = tmp_path / "summary.md"
    code = summarize.main(
        [
            "--title",
            "t",
            "--build-log",
            str(log),
            "--repo-root",
            ROOT,
            "--build-dir",
            BUILD,
            "--junit",
            "%s=one thread" % junit,
            "--junit",
            "%s=missing" % (tmp_path / "nothing.xml"),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    text = output.read_text()
    assert "| one thread | pass | 4 | 3 | 0 | 1 | 3.0 s |" in text
    assert "| missing | FAIL | 0 |" in text  # a missing result file is not a pass


KNOWN = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Linux" tests="2" failures="1" disabled="0" skipped="0" time="1">
  <testcase name="KnownIssue.still_broken" classname="x" time="0.5" status="fail">
    <failure message=""/>
  </testcase>
  <testcase name="KnownIssue.fixed_meanwhile" classname="x" time="0.5" status="run"/>
</testsuite>
"""


def test_known_issues_are_listed_apart(tmp_path):
    known = tmp_path / "known.xml"
    known.write_text(KNOWN)
    output = tmp_path / "summary.md"
    summarize.main(["--title", "t", "--known-issues", str(known), "--output", str(output)])
    text = output.read_text()
    assert "### Known issues (not blocking)" in text
    assert "| KnownIssue.still_broken | failing, as documented |" in text
    assert "KnownIssue.fixed_meanwhile | **passes now" in text
    # an expected failure is not reported as a failed test run
    assert "FAIL" not in text
