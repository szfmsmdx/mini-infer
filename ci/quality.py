"""Check the lines introduced by this change. Run from the repository root."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


CPP = {".h", ".hpp", ".cuh", ".c", ".cc", ".cpp", ".cxx", ".cu"}
PRODUCT = {"src", "include", "examples"}
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def git(*args):
    return subprocess.check_output(["git", "--literal-pathspecs", *args]).decode("utf-8")


def documents_only(names):
    return bool(names) and all(Path(name).suffix.lower() in {".md", ".rst", ".drawio"} for name in names)


def changes(base):
    # Both line numbers and coverage refer to the checked-out head, even if main moved.
    base = git("merge-base", base, "HEAD").strip()
    paths = git("diff", "--name-only", "--no-renames", "-z", base, "HEAD").split("\0")
    result = {}
    for name in filter(None, paths):
        path = Path(name)
        if path.parts[0] == "3rdparty" or not path.is_file() or path.suffix not in CPP:
            continue
        ranges = []
        for line in git("diff", "--no-ext-diff", "--no-renames", "--unified=0", base, "HEAD", "--", name).splitlines():
            match = HUNK.match(line)
            if match:
                start, count = int(match[1]), int(match[2] or 1)
                if count:
                    ranges.append([start, start + count - 1])
        if ranges:
            result[name] = ranges
    return result


def lint(changed, build):
    for name, ranges in changed.items():
        args = [f"--lines={start}:{end}" for start, end in ranges]
        subprocess.run(["clang-format-18", "--dry-run", "--Werror", *args, "--", name], check=True)

    host = {name: ranges for name, ranges in changed.items() if Path(name).suffix not in {".cu", ".cuh"}}
    if not host:
        return
    filters = [{"name": "/" + name, "lines": ranges} for name, ranges in host.items()]
    # A header can affect any translation unit. Restrict reported diagnostics to the diff.
    headers = any(Path(name).suffix in {".h", ".hpp"} for name in host)
    units = json.loads((build / "compile_commands.json").read_text())
    checked = set()
    for unit in units:
        source = (Path(unit["directory"]) / unit["file"]).resolve()
        if not source.is_relative_to(Path.cwd()) or source in checked:
            continue
        if headers or source.relative_to(Path.cwd()).as_posix() in host:
            subprocess.run(["clang-tidy-18", "-p", str(build), "--quiet",
                            "--line-filter=" + json.dumps(filters), str(source)], check=True)
            checked.add(source)


def coverage_lines(report):
    # A header can appear in multiple translation units; a hit in any one counts.
    files = {}
    for item in ET.parse(report).iter("class"):
        path = Path(item.attrib["filename"])
        if path.is_absolute():
            path = path.relative_to(Path.cwd())
        if not path.parts or path.parts[0] not in PRODUCT:
            continue
        lines = files.setdefault(path.as_posix(), {})
        for line in item.findall("lines/line"):
            number = int(line.attrib["number"])
            lines[number] = lines.get(number, False) or int(line.attrib["hits"]) > 0
    if not any(files.values()):
        raise ValueError("Coverage report contains no measured production lines")
    return files


def assess(files, changed):
    total = sum(len(lines) for lines in files.values())
    hits = sum(sum(lines.values()) for lines in files.values())
    patch = patch_hits = 0
    unmeasured = []
    for name, ranges in changed.items():
        path = Path(name)
        if path.parts[0] not in PRODUCT:
            continue
        if path.suffix in {".cu", ".cuh"} or not files.get(name):
            unmeasured.append(name)
            continue
        for number, hit in files.get(name, {}).items():
            if any(start <= number <= end for start, end in ranges):
                patch += 1
                patch_hits += hit
    needs_waiver = hits * 100 < total * 60 or patch_hits * 100 < patch * 70 or bool(unmeasured)
    return hits, total, patch_hits, patch, unmeasured, needs_waiver


def ratio(hit, total):
    return f"{hit}/{total} ({hit / total:.2%})" if total else "N/A (no changed executable lines)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["lint", "coverage"])
    parser.add_argument("base")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    changed = changes(args.base)
    if args.command == "lint":
        lint(changed, args.input)
        return
    hits, total, ph, pn, missing, waiver = assess(coverage_lines(args.input), changed)
    names = git("diff", "--name-only", "--no-renames", "-z", args.base + "...HEAD").split("\0")[:-1]
    docs_waiver = os.environ.get("GITHUB_EVENT_NAME") == "pull_request" and documents_only(names)
    summary = (f"### CPU line coverage\n\n| Scope | Hits | Minimum |\n|---|---:|---:|\n"
               f"| Production | {ratio(hits, total)} | 60% |\n| PR additions | {ratio(ph, pn)} | 70% |\n")
    if missing:
        summary += "\nChanged production files without CPU measurement: " + ", ".join(f"`{p}`" for p in missing) + "\n"
    if docs_waiver:
        summary += "\nDocumentation-only PR: the owner can approve the N/A patch measurement in Review deployments.\n"
        waiver = True
    print(summary)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(summary)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            print("waiver=" + str(waiver).lower(), file=stream)
    elif waiver:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
