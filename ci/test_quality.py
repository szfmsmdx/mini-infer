from pathlib import Path
import os
import subprocess
import tempfile
import unittest

import quality


class CoveragePolicy(unittest.TestCase):
    def test_thresholds_are_inclusive_and_independent(self):
        changed = {"src/a.cpp": [[1, 10]]}
        lines = {i: i <= 7 or 11 <= i <= 15 for i in range(1, 21)}
        self.assertEqual(quality.assess({"src/a.cpp": lines}, changed), (12, 20, 7, 10, [], False))
        lines[15] = False  # 55% overall, 70% patch
        self.assertTrue(quality.assess({"src/a.cpp": lines}, changed)[-1])
        lines[7], lines[15], lines[16] = False, True, True  # 60% overall, 60% patch
        self.assertTrue(quality.assess({"src/a.cpp": lines}, changed)[-1])

    def test_docs_and_unmeasured_sources(self):
        self.assertTrue(quality.documents_only(["README.md", "note/design.drawio"]))
        self.assertFalse(quality.documents_only(["README.md", "src/module.cpp"]))
        self.assertFalse(quality.documents_only([]))
        files = {"src/a.cpp": {5: True}}
        self.assertEqual(quality.assess(files, {}), (1, 1, 0, 0, [], False))
        for name in ("src/unused.cpp", "include/new.h", "src/backend/cuda/kernel.cu"):
            with self.subTest(name=name):
                self.assertEqual(quality.assess(files, {name: [[1, 3]]})[-2:], ([name], True))

    def test_report_merges_hits_and_rejects_empty_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.xml"
            classes = ''.join(f'<class filename="include/x.h"><lines><line number="3" hits="{n}"/>'
                              '</lines></class>' for n in (0, 1, 0))
            report.write_text(f"<coverage>{classes}</coverage>")
            self.assertEqual(quality.coverage_lines(report), {"include/x.h": {3: True}})
            report.write_text("<coverage/>")
            with self.assertRaises(ValueError):
                quality.coverage_lines(report)

    def test_git_hunks_count_head_lines_only(self):
        with tempfile.TemporaryDirectory() as directory:
            # Use a real Git diff, including a deletion and a filename with a space.
            previous = Path.cwd()
            try:
                os.chdir(directory)
                def git(*args):
                    subprocess.run(["git", *args], check=True, capture_output=True)
                git("init", "-q")
                file = Path("a b.cpp")
                file.write_text("remove\nunchanged\n")
                git("add", ".")
                def commit():
                    git("-c", "user.name=CI", "-c", "user.email=ci@example.invalid", "commit", "-qm", "test")
                commit()
                file.write_text("unchanged\nnew\n")
                git("add", ".")
                commit()
                self.assertEqual(quality.changes("HEAD~1"), {"a b.cpp": [[2, 2]]})
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
