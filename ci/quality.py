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
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
CUDA_MACRO = re.compile(r"\b(?:ENABLE_CUDA|__CUDACC__|__CUDA_ARCH__)\b")


def git(*args):
    return subprocess.check_output(["git", "--literal-pathspecs", *args]).decode("utf-8")


def documents_only(names):
    return bool(names) and all(Path(name).suffix.lower() in {".md", ".rst", ".drawio"} for name in names)


def cuda_lines(source):
    """Track CUDA guards on each revision, including the directives themselves."""
    if not CUDA_MACRO.search(source):
        return set()
    # Preserve line numbers and quoted literals while dropping comments.
    source = re.sub(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|//[^\n]*|/\*[\s\S]*?\*/''',
                    lambda m: " " + "\n" * m[0].count("\n") if m[0].startswith("/") else m[0], source)
    guarded = set()
    stack = []
    lines = iter(enumerate(source.splitlines(), 1))
    for start, line in lines:
        end = start
        while line.endswith("\\"):
            end, continuation = next(lines, (end, ""))
            line = line[:-1] + continuation
        directive = re.match(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", line)
        guard_line = False
        if directive:
            op, expr = directive[1], directive[2].strip()
            cuda = bool(CUDA_MACRO.search(expr))
            if op in {"ifdef", "ifndef", "if"}:
                # (current arm, alternative arm). For compound expressions check both.
                stack.append((cuda and op != "ifndef", cuda and op != "ifdef"))
                guard_line = cuda
            elif stack:
                active, alternative = stack[-1]
                guard_line = active or alternative
                if op == "else":
                    stack[-1] = (alternative, active)
                elif op == "elif":
                    stack[-1] = (alternative or cuda, guard_line or cuda)
                    guard_line |= cuda
                elif op == "endif":
                    stack.pop()
        if line.strip() and (guard_line or any(active for active, _ in stack)):
            guarded.update(range(start, end + 1))
    return guarded


def changes(base):
    base = git("merge-base", base, "HEAD").strip()
    entries = iter(git("diff", "--name-status", "--find-renames", "-z", base, "HEAD").split("\0")[:-1])
    names, changed, cuda = set(), {}, set()
    for status in entries:
        old = new = next(entries)
        if status.startswith("R"):
            new = next(entries)
        names.update((old, new))
        for name in (old, new):
            path = Path(name)
            if path.parts[0] in PRODUCT and path.suffix in {".cu", ".cuh"}:
                cuda.add(name)
        if not any(Path(p).suffix in CPP and Path(p).parts[0] != "3rdparty" for p in (old, new)):
            continue
        if status == "R100":
            continue
        old_blob, new_blob = f"{base}:{old}", f"HEAD:{new}"
        # Compare the two actual blobs for a rename, including names with spaces.
        revisions = [base, "HEAD", "--", new] if status in {"A", "D"} else [old_blob, new_blob]
        patch = git("diff", "--no-ext-diff", "--no-textconv", "--unified=0", *revisions)
        before, after = [], []
        for line in patch.splitlines():
            match = HUNK.match(line)
            if match:
                for ranges, start, count in ((before, match[1], match[2]), (after, match[3], match[4])):
                    if int(count or 1):
                        ranges.append([int(start), int(start) + int(count or 1) - 1])
        if after and Path(new).suffix in CPP and Path(new).parts[0] != "3rdparty":
            changed[new] = after
        for name, blob, ranges in ((old, old_blob, before), (new, new_blob, after)):
            path = Path(name)
            if ranges and name not in cuda and path.parts[0] in PRODUCT and path.suffix in CPP:
                lines = cuda_lines(git("show", blob))
                if any(first <= number <= last for number in lines for first, last in ranges):
                    cuda.add(name)
    return names, changed, cuda


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


def assess(files, changed, cuda=()):
    total = sum(len(lines) for lines in files.values())
    hits = sum(sum(lines.values()) for lines in files.values())
    patch = patch_hits = 0
    unmeasured = set(cuda)
    for name, ranges in changed.items():
        path = Path(name)
        if path.parts[0] not in PRODUCT:
            continue
        if path.suffix in {".cu", ".cuh"} or not files.get(name):
            unmeasured.add(name)
            continue
        for number, hit in files.get(name, {}).items():
            if any(start <= number <= end for start, end in ranges):
                patch += 1
                patch_hits += hit
    needs_waiver = hits * 100 < total * 60 or patch_hits * 100 < patch * 70 or bool(unmeasured)
    return hits, total, patch_hits, patch, sorted(unmeasured), needs_waiver


def ratio(hit, total):
    return f"{hit}/{total} ({hit / total:.2%})" if total else "N/A (no changed executable lines)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["lint", "coverage"])
    parser.add_argument("base")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    names, changed, cuda = changes(args.base)
    if args.command == "lint":
        lint(changed, args.input)
        return
    hits, total, ph, pn, missing, waiver = assess(coverage_lines(args.input), changed, cuda)
    docs_waiver = os.environ.get("GITHUB_EVENT_NAME") == "pull_request" and documents_only(names)
    summary = (f"### CPU line coverage\n\n| Scope | Hits | Minimum |\n|---|---:|---:|\n"
               f"| Production | {ratio(hits, total)} | 60% |\n| PR additions | {ratio(ph, pn)} | 70% |\n")
    if missing:
        summary += "\nCUDA-dependent changes or production files without CPU measurement: " + ", ".join(f"`{p}`" for p in missing) + "\n"
    if docs_waiver:
        summary += "\nDocumentation-only PR: the owner can approve the N/A patch measurement in Review deployments.\n"
        waiver = True
    print(summary)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(summary)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            print("waiver=" + str(waiver and os.environ.get("GITHUB_EVENT_NAME") == "pull_request").lower(), file=stream)
    elif waiver:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
