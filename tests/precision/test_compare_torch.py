"""Checks for the oracle bridge itself, including deliberately wrong outputs."""

import copy
import math
import unittest

import torch

from compare_torch import (
    BINARY, TOLERANCES, UNARY, compare, compare_dual, decode, math_reference,
    reference, validate_inventory, cohort_ranges, accept_cohorts,
)


class ComparatorTest(unittest.TestCase):
    @staticmethod
    def record(name, a, g=None, b=None, dtype="float32"):
        record = {"name": name, "dtype": dtype, "device": "cpu"}
        for key, values in (("a", a), ("g", g), ("b", b)):
            if values is None:
                continue
            tensor = torch.tensor(values, dtype=torch.float64).to(getattr(torch, dtype))
            bits = tensor.view(torch.int32 if dtype == "float32" else torch.int16)
            mask = 0xffffffff if dtype == "float32" else 0xffff
            record[key] = {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                           "offset": 0, "bits": [v & mask for v in bits.tolist()]}
        return record

    def test_dual_acceptance_is_per_element(self):
        for dtype in TOLERANCES:
            kw = {"dtype": getattr(torch, dtype)}
            actual = torch.tensor([1., 2., 9., float("inf"), float("nan"), 0.], **kw)
            mathematical = torch.tensor([1., 3., 4., 10., 10., .4], **kw)
            expected = torch.tensor([2., 2., 5., float("inf"), float("nan"), 0.], **kw)
            result = compare_dual(actual, mathematical, expected, dtype)
            self.assertEqual(result["math_pass"], 1)
            self.assertEqual(result["torch_only_pass"], 4)
            self.assertEqual(result["failed_indices"], [2])
            self.assertEqual(result["count"], result["math_pass"] +
                             result["torch_only_pass"] + result["failed"])

    def test_dual_rejects_invalid_special_values_and_metadata(self):
        actual = torch.tensor([float("inf"), float("nan")])
        mathematical = torch.tensor([-float("inf"), 0.])
        expected = mathematical.clone()
        self.assertEqual(compare_dual(actual, mathematical, expected, "float32")
                         ["failed_indices"], [0, 1])
        with self.assertRaises(ValueError):
            compare_dual(torch.zeros(2), torch.zeros(1), torch.zeros(2), "float32")
        with self.assertRaises(ValueError):
            compare_dual(torch.zeros(1), torch.zeros(1), torch.zeros(1).half(), "float32")

    def test_analytic_formulas_and_both_binary_gradients(self):
        s = 1 / (1 + math.exp(-4.))
        ds = math.exp(-4.) / (1 + math.exp(-4.)) ** 2
        unary = {"exp": (math.exp(4.), 3 * math.exp(4.)), "sigmoid": (s, 3 * ds),
                 "silu": (4 * s, 3 * (s + 4 * ds)), "sqrt": (2., .75),
                 "rsqrt": (.5, -.1875), "neg": (-4., -3.)}
        binary = {"add": (6., 3., 3.), "sub": (2., 3., -3.),
                  "mul": (8., 6., 12.), "div": (2., 1.5, -3.)}
        for dtype in TOLERANCES:
            for op, values in {**unary, **binary}.items():
                stages = ("forward", "backward") if op in UNARY else ("forward", "grad_a", "grad_b")
                for stage, value in zip(stages, values):
                    record = self.record(f"{op}/{stage}", [4.], g=[3.],
                                         b=[2.] if op in BINARY else None, dtype=dtype)
                    actual = math_reference(record)
                    expected = torch.tensor([value], dtype=torch.float64).to(getattr(torch, dtype))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_math_reference_survives_low_precision_intermediate_overflow(self):
        record = self.record("sigmoid/backward", [12.], g=[65504.], dtype="float16")
        mathematical = math_reference(record)
        self.assertEqual(mathematical.item(), torch.tensor(.40246554030972825).half().item())
        zero = torch.zeros_like(mathematical)
        result = compare_dual(zero, mathematical, reference(record), "float16")
        self.assertEqual(result["torch_only_pass"], 1)
        self.assertEqual(result["failed"], 0)
        record = self.record("sigmoid/backward", [8.], g=[1.], dtype="bfloat16")
        mathematical = math_reference(record)
        result = compare_dual(mathematical, mathematical, reference(record), "bfloat16")
        self.assertEqual(result["math_pass"], 1)
        self.assertGreater(result["torch_comparison"]["failed"], 0)

    def test_math_special_values_and_expected_output_overflow(self):
        x = [0., -0., float("inf"), -float("inf"), float("nan")]
        expected = {"exp": [1., 1., math.inf, 0., math.nan],
                    "sigmoid": [.5, .5, 1., 0., math.nan],
                    "silu": [0., -0., math.inf, math.nan, math.nan],
                    "sqrt": [0., -0., math.inf, math.nan, math.nan],
                    "rsqrt": [math.inf, -math.inf, 0., math.nan, math.nan],
                    "neg": [-0., 0., -math.inf, math.inf, math.nan]}
        for op, values in expected.items():
            torch.testing.assert_close(math_reference(self.record(f"{op}/special", x)),
                                       torch.tensor(values), rtol=0, atol=0, equal_nan=True)
        self.assertTrue(math.isinf(math_reference(
            self.record("exp/forward", [12.], dtype="float16")).item()))

    def test_analytic_extreme_gradients_stay_in_fp64_until_final_rounding(self):
        tiny = 2. ** -149
        cases = [
            (self.record("div/grad_b", [tiny], g=[tiny], b=[tiny]), -1.),
            (self.record("rsqrt/backward", [tiny], g=[tiny]), -.5 / math.sqrt(tiny)),
            (self.record("exp/backward", [90.], g=[2. ** -100]), math.exp(90.) * 2. ** -100),
        ]
        for record, value in cases:
            expected = torch.tensor([value], dtype=torch.float64).float()
            torch.testing.assert_close(math_reference(record), expected, rtol=0, atol=0)
        record = self.record("div/special", [1., -1., 0.], b=[0., 0., 0.])
        torch.testing.assert_close(math_reference(record),
                                   torch.tensor([math.inf, -math.inf, math.nan]),
                                   rtol=0, atol=0, equal_nan=True)

    def test_bits_and_strided_offset(self):
        for dtype, bits in {
            "float32": [0x80000000, 1, 0x7f800000, 0x7fc00000],
            "float16": [0x8000, 1, 0x7c00, 0x7e00],
            "bfloat16": [0x8000, 1, 0x7f80, 0x7fc0],
        }.items():
            with self.subTest(dtype=dtype):
                x = decode({"shape": [4], "stride": [3], "offset": 2, "bits": bits}, dtype)
                self.assertEqual(x.stride(), (3,))
                self.assertEqual(x.storage_offset(), 2)
                self.assertTrue(torch.signbit(x[0]))
                self.assertGreater(x[1].item(), 0)
                self.assertTrue(math.isinf(x[2].item()))
                self.assertTrue(math.isnan(x[3].item()))
                raw_type = torch.int32 if dtype == "float32" else torch.int16
                mask = 0xffffffff if dtype == "float32" else 0xffff
                self.assertEqual([v & mask for v in x.contiguous().view(raw_type).tolist()], bits)

    def test_wrong_values_fail_all_dtypes(self):
        for dtype in TOLERANCES:
            a = torch.tensor([1., 2., float("inf"), float("nan")], dtype=getattr(torch, dtype))
            b = torch.tensor([2., 2., -float("inf"), 1.], dtype=a.dtype)
            result = compare(a, b, dtype)
            self.assertEqual(result["failed_indices"], [0, 2, 3])
            self.assertEqual(result["finite_failed"], 1)
            self.assertEqual(result["nonfinite_failed"], 2)

    def test_matching_nan_inf_and_small_zero_difference(self):
        a = torch.tensor([float("nan"), float("inf"), -float("inf"), 0., -0.])
        b = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e-10, 0.])
        result = compare(a, b, "float32")
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["zero_mismatch"], 1)
        self.assertEqual(result["zero_sign_mismatch"], 1)

    def test_shape_and_dtype_mismatch(self):
        with self.assertRaises(ValueError):
            compare(torch.zeros(2), torch.zeros(1), "float32")
        with self.assertRaises(ValueError):
            compare(torch.zeros(1), torch.zeros(1).half(), "float32")

    def test_real_autograd_uses_supplied_upstream(self):
        a = {"shape": [2], "stride": [2], "offset": 0, "bits": [0, 0]}
        g = {"shape": [2], "stride": [2], "offset": 1, "bits": [0x40000000, 0xc0400000]}
        result = reference({"name": "exp/backward", "dtype": "float32",
                            "device": "cpu", "a": a, "g": g})
        self.assertEqual(result.tolist(), [2., -3.])

    def test_inventory_cannot_silently_drop_or_duplicate_cases(self):
        records = [{"name": f"{op}/{stage}", "dtype": dtype, "device": "cpu"}
                   for dtype in TOLERANCES for op in UNARY + BINARY
                   for stage in (("special", "forward", "backward") if op in UNARY
                                 else ("special", "forward", "grad_a", "grad_b"))]
        for record in records:
            special = record["name"].endswith("/special")
            record["cohorts"] = ({"special": 5} if special else
                                 {"basic": 513, "boundary": 1, "random": 4097})
            record["a"] = {"bits": [0] * sum(record["cohorts"].values())}
        validate_inventory(records, "cpu")
        for bad in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                validate_inventory(bad, "cpu")
        with self.assertRaises(ValueError):
            validate_inventory(records, "cuda")
        changed = copy.deepcopy(records)
        changed[2]["a"] = {"bits": [1]}
        with self.assertRaises(ValueError):
            validate_inventory(changed, "cpu")

    def test_random_threshold_is_20_not_21(self):
        # Exercise real dual comparisons, not just fabricated failure counts.
        for dtype in TOLERANCES:
            record = self.record("silu/backward", [1.] * 5330, dtype=dtype)
            record["cohorts"] = {"basic": 513, "boundary": 720, "random": 4097}
            expected = torch.ones(5330, dtype=getattr(torch, dtype))
            for failed in (0, 20, 21):
                actual = expected.clone()
                actual[1233:1233 + failed] = 2
                result = compare_dual(actual, expected, expected, dtype)
                verdict = accept_cohorts(record, result)
                random = verdict["cohorts"]["random"]
                self.assertEqual(verdict["accepted"], failed <= 20)
                self.assertEqual(random["allowed_failures"], 20)
                self.assertEqual(random["failed"], failed)
                self.assertEqual(random["failed_indices"], list(range(1233, 1233 + failed)))
                self.assertEqual(result["failed"], failed)  # Never erase tolerated failures.
                self.assertAlmostEqual(random["pass_rate"], (4097 - failed) / 4097)
                self.assertEqual(verdict["cohorts"]["basic"]["count"], 513)
                self.assertEqual(verdict["cohorts"]["boundary"]["count"], 720)

    def test_strict_cohorts_cannot_borrow_random_budget(self):
        record = self.record("silu/backward", [1.] * 5330)
        record["cohorts"] = {"basic": 513, "boundary": 720, "random": 4097}
        expected = torch.ones(5330)
        for index, cohort in ((0, "basic"), (512, "basic"),
                              (513, "boundary"), (1232, "boundary")):
            actual = expected.clone()
            actual[index] = 2
            result = accept_cohorts(record, compare_dual(actual, expected, expected, "float32"))
            self.assertFalse(result["accepted"])
            self.assertEqual(result["cohorts"][cohort]["allowed_failures"], 0)
            self.assertTrue(result["cohorts"]["random"]["accepted"])
        special = self.record("exp/special", [0., 1., 2., 3., 4.])
        special["cohorts"] = {"special": 5}
        result = accept_cohorts(special, compare_dual(torch.ones(5), torch.zeros(5),
                                                     torch.zeros(5), "float32"))
        self.assertFalse(result["accepted"])

    def test_cohort_metadata_must_cover_all_samples(self):
        record = self.record("silu/forward", [1.] * 5330)
        valid = {"basic": 513, "boundary": 720, "random": 4097}
        for counts in ({}, {"special": 5330}, {**valid, "extra": 1},
                       {**valid, "basic": 512}, {**valid, "random": 4096},
                       {**valid, "boundary": 719}, {**valid, "boundary": 0},
                       {**valid, "boundary": -1}, {**valid, "boundary": 720.0}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                cohort_ranges({**record, "cohorts": counts})
        record["cohorts"] = valid
        self.assertEqual(cohort_ranges(record),
                         {"basic": (0, 513), "boundary": (513, 1233), "random": (1233, 5330)})


if __name__ == "__main__":
    unittest.main()
