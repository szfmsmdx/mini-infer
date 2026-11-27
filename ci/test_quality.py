from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

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

    def test_cuda_guards_keep_cpu_only_arms_out_of_approval(self):
        for op, gpu, cpu in (("ifdef", 2, 4), ("ifndef", 4, 2)):
            text = f"#{op} ENABLE_CUDA\nfirst();\n#else\nsecond();\n#endif"
            lines = quality.cuda_lines(text)
            self.assertIn(gpu, lines)
            self.assertNotIn(cpu, lines)
            self.assertTrue({1, 3, 5} <= lines)
        lines = quality.cuda_lines("#if OTHER\na();\n#elif defined(ENABLE_CUDA)\nb();\n#endif")
        self.assertEqual(lines, {3, 4, 5})

class GitChanges(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(directory.name)
        self.git("init", "-q")
        Path("src").mkdir()

    def git(self, *args):
        subprocess.run(["git", "-c", "user.name=CI", "-c", "user.email=ci@example.invalid", *args],
                       check=True, capture_output=True)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "test")

    def test_rename_only_and_rename_with_one_edit(self):
        old = Path("src/a b.cpp")
        old.write_text("".join(f"int variable_{i};\n" for i in range(10)))
        self.commit()
        new = old.rename("src/renamed.cpp")
        self.commit()
        self.assertEqual(quality.changes("HEAD~1"), ({"src/a b.cpp", "src/renamed.cpp"}, {}, set()))
        new.write_text(new.read_text().replace("variable_4", "replacement"))
        self.commit()
        self.assertEqual(quality.changes("HEAD~2")[1], {"src/renamed.cpp": [[5, 5]]})

    def test_cuda_deletions_and_hidden_lines_in_measured_file(self):
        kernel, header, mixed = map(Path, ("src/kernel.cu", "src/helper.cuh", "src/runtime.cpp"))
        kernel.write_text("keep();\nremoved();\n")
        header.write_text("helper();\n")
        src = "#ifdef ENABLE_CUDA\n#if OTHER\nold_gpu();\n#endif\n#endif\ncpu();\n"
        mixed.write_text(src)
        self.commit()
        kernel.write_text("keep();\n")
        header.unlink()
        mixed.write_text(src.replace("old_gpu", "new_gpu"))
        self.commit()
        _, changed, cuda = quality.changes("HEAD~1")
        self.assertEqual(changed, {"src/runtime.cpp": [[3, 3]]})
        self.assertEqual(cuda, {"src/kernel.cu", "src/helper.cuh", "src/runtime.cpp"})
        self.assertTrue(quality.assess({"src/runtime.cpp": {6: True}}, changed, cuda)[-1])
        mixed.write_text(mixed.read_text().replace("new_gpu();\n", ""))
        self.commit()
        self.assertEqual(quality.changes("HEAD~1")[1:], ({}, {"src/runtime.cpp"}))
        mixed.write_text(mixed.read_text().replace("#if OTHER", "// note\n#if OTHER").replace("cpu();", "cpu2();"))
        self.commit()
        self.assertFalse(quality.changes("HEAD~1")[2])

    def test_push_reports_low_coverage_without_requesting_approval(self):
        source = Path("src/a.cpp")
        source.write_text("void f() {}\n")
        self.commit()
        source.write_text("void g() {}\n")
        self.commit()
        Path("report.xml").write_text('<coverage><class filename="src/a.cpp"><lines>'
                                     '<line number="1" hits="0"/></lines></class></coverage>')
        for event, expected in (("pull_request", "true"), ("push", "false")):
            with self.subTest(event=event), patch.dict(os.environ, GITHUB_EVENT_NAME=event,
                    GITHUB_OUTPUT="out", GITHUB_STEP_SUMMARY="summary"), \
                    patch("sys.argv", ["quality.py", "coverage", "HEAD~1", "report.xml"]):
                Path("out").write_text("")
                quality.main()
                self.assertEqual(Path("out").read_text().strip(), f"waiver={expected}")


if __name__ == "__main__":
    unittest.main()
