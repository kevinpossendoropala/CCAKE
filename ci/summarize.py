#!/usr/bin/env python3
"""Summarise a CI build: compiler warnings and test results, as Markdown.

    ci/summarize.py --title "gcc-13" --build-log build.log --build-dir build \\
        --junit build/threads.xml="all threads" \\
        --junit build/one.xml="OpenMP, 1 thread" \\
        --output "$GITHUB_STEP_SUMMARY" --annotate-diff HEAD^1

Warnings are counted once each (a header that is included by many files reports
the same warning many times) and split into those in CCAKE's own files and
those in the headers of its dependencies.

With --annotate-diff <base>, warnings that sit on lines changed since <base>
are also printed as GitHub annotations, so that a pull request shows the
warnings it introduces and not the existing ones.

Only the Python standard library is used. The script reports; it never fails
the build.
"""

import argparse
import collections
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ElementTree

# Directories that hold CCAKE's own code, relative to the repository root.
OWN_DIRECTORIES = ("src/", "include/", "unit_tests/", "utilities/", "tests/")

WARNING = re.compile(
    r"^(?P<path>[^\s:][^:]*):(?P<line>\d+):(?:(?P<column>\d+):)? warning: "
    r"(?P<message>.*?)(?: \[(?P<flag>-W[^\]\s]+)\])?\s*$"
)
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")

Warning = collections.namedtuple("Warning", "path line column flag message own")
TestRun = collections.namedtuple("TestRun", "label total failed skipped seconds failures passed")


def relative_to_repo(path, repo_root, build_dir):
    """Path relative to the repository root, or None when it lies outside."""
    if not os.path.isabs(path):
        path = os.path.join(build_dir, path)
    path = os.path.normpath(path)
    root = os.path.normpath(repo_root)
    if path == root or not path.startswith(root + os.sep):
        return None
    return os.path.relpath(path, root).replace(os.sep, "/")


def parse_warnings(text, repo_root, build_dir):
    """Unique compiler warnings found in a build log."""
    seen = set()
    warnings = []
    for line in text.splitlines():
        match = WARNING.match(line)
        if not match:
            continue
        relative = relative_to_repo(match.group("path"), repo_root, build_dir)
        own = relative is not None and relative.startswith(OWN_DIRECTORIES)
        warning = Warning(
            path=relative if own else match.group("path"),
            line=int(match.group("line")),
            column=int(match.group("column") or 0),
            flag=match.group("flag") or "(no flag)",
            message=match.group("message"),
            own=own,
        )
        if warning not in seen:
            seen.add(warning)
            warnings.append(warning)
    return warnings


def parse_junit(path, label):
    """Totals and failures of one JUnit file written by ctest --output-junit."""
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = failed = skipped = 0
    seconds = 0.0
    failures = []
    passed = []
    for suite in suites:
        for case in suite.iter("testcase"):
            total += 1
            seconds += float(case.get("time") or 0)
            if case.find("failure") is not None or case.find("error") is not None:
                failed += 1
                output = case.findtext("system-out") or ""
                failures.append((case.get("name"), output.strip()))
            elif case.find("skipped") is not None or case.get("status") in ("notrun", "disabled"):
                skipped += 1
            else:
                passed.append(case.get("name"))
    return TestRun(label, total, failed, skipped, seconds, failures, passed)


def changed_lines(diff_text):
    """{path: set of line numbers} added or modified, from 'git diff -U0' output."""
    changed = collections.defaultdict(set)
    current = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = target[2:] if target.startswith("b/") else None
        elif current:
            match = HUNK.match(line)
            if match:
                start = int(match.group("start"))
                count = int(match.group("count") or 1)
                changed[current].update(range(start, start + count))
    return changed


def escape(text, is_property=False):
    """Escape a value for a GitHub workflow command."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if is_property:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def annotations(warnings, changed, limit):
    """Workflow commands for the warnings on changed lines, and how many there are."""
    lines = []
    for warning in warnings:
        if warning.own and warning.line in changed.get(warning.path, ()):
            lines.append(
                "::warning file=%s,line=%d,col=%d,title=%s::%s"
                % (
                    escape(warning.path, True),
                    warning.line,
                    max(warning.column, 1),
                    escape(warning.flag, True),
                    escape(warning.message),
                )
            )
    count = len(lines)
    if count > limit:
        lines = lines[:limit] + [
            "::notice::%d more warnings on changed lines are not shown" % (count - limit)
        ]
    return lines, count


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return out


def render_known_issues(run):
    """Known issues: tests that document reported problems and are expected to fail."""
    out = ["### Known issues (not blocking)", ""]
    if run.total == 0:
        return out + ["No known-issue tests ran.", ""]
    out.append(
        "These tests document problems that were reported and are not fixed yet; "
        "they are expected to fail."
    )
    out.append("")
    rows = [(name, "failing, as documented") for name, _ in run.failures]
    rows += [(name, "**passes now: move it out of the KnownIssue suite**") for name in run.passed]
    out += table(("test", "status"), rows)
    return out + [""]


def render(title, warnings, runs, new_warnings=None, known=None):
    """The Markdown summary."""
    out = ["## %s" % title, ""]
    if runs:
        out.append("### Tests")
        out.append("")
        rows = []
        for run in runs:
            passed = run.total - run.failed - run.skipped
            verdict = "pass" if run.failed == 0 and run.total > 0 else "FAIL"
            rows.append(
                (
                    run.label,
                    verdict,
                    run.total,
                    passed,
                    run.failed,
                    run.skipped,
                    "%.1f s" % run.seconds,
                )
            )
        out += table(("run", "result", "tests", "passed", "failed", "skipped", "time"), rows)
        out.append("")
        for run in runs:
            for name, output in run.failures:
                out.append(
                    "<details><summary>failed: <code>%s</code> (%s)</summary>" % (name, run.label)
                )
                out += ["", "```"] + output.splitlines()[-40:] + ["```", "", "</details>", ""]
    if known is not None:
        out += render_known_issues(known)
    if warnings is not None:
        own = [w for w in warnings if w.own]
        other = [w for w in warnings if not w.own]
        out.append("### Compiler warnings")
        out.append("")
        rows = [("in CCAKE files", len(own)), ("in dependency headers", len(other))]
        if new_warnings is not None:
            rows.insert(0, ("on lines changed by this pull request", new_warnings))
        out += table(("where", "unique warnings"), rows)
        out.append("")
        if own:
            by_flag = collections.Counter(w.flag for w in own).most_common(12)
            by_file = collections.Counter(w.path for w in own).most_common(12)
            out.append("<details><summary>CCAKE warnings by kind and by file</summary>")
            out.append("")
            out += table(("kind", "count"), by_flag)
            out.append("")
            out += table(("file", "count"), by_file)
            out += ["", "</details>", ""]
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--title", default="Build")
    parser.add_argument("--build-log", help="output of the build, to count warnings")
    parser.add_argument("--build-dir", default="build", help="directory the build ran in")
    parser.add_argument("--repo-root", default=".", help="root of the repository")
    parser.add_argument(
        "--junit",
        action="append",
        default=[],
        metavar="FILE=LABEL",
        help="a JUnit file from ctest and a name for that run",
    )
    parser.add_argument(
        "--known-issues",
        metavar="FILE",
        help="JUnit file of the known-issue tests (ctest -L quarantine), listed separately",
    )
    parser.add_argument(
        "--annotate-diff",
        metavar="BASE",
        help="annotate warnings on lines changed since this commit",
    )
    parser.add_argument("--max-annotations", type=int, default=30)
    parser.add_argument("--output", help="append the Markdown here (default: print it)")
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(args.repo_root)
    build_dir = os.path.abspath(args.build_dir)

    warnings = None
    if args.build_log and os.path.isfile(args.build_log):
        with open(args.build_log, errors="replace") as handle:
            warnings = parse_warnings(handle.read(), repo_root, build_dir)

    runs = []
    for item in args.junit:
        path, _, label = item.partition("=")
        if os.path.isfile(path):
            runs.append(parse_junit(path, label or os.path.basename(path)))
        else:
            runs.append(TestRun(label or path, 0, 0, 0, 0.0, [("(no result file)", path)], []))

    new_warnings = None
    if args.annotate_diff and warnings is not None:
        diff = subprocess.run(
            ["git", "-C", repo_root, "diff", "-U0", "--no-color", args.annotate_diff, "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff.returncode == 0:
            commands, new_warnings = annotations(
                warnings, changed_lines(diff.stdout), args.max_annotations
            )
            for command in commands:
                print(command)
        else:
            print(
                "::notice::could not compare with %s, no warning annotations" % args.annotate_diff
            )

    known = None
    if args.known_issues and os.path.isfile(args.known_issues):
        known = parse_junit(args.known_issues, "known issues")
    markdown = render(args.title, warnings, runs, new_warnings, known)
    if args.output:
        with open(args.output, "a") as handle:
            handle.write(markdown)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
