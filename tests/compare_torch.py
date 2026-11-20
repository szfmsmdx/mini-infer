"""Accept each element if it matches the mathematical reference or PyTorch."""

import argparse
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import struct
import subprocess
import sys

import torch


TOLERANCES = {
    "float32": (1.3e-6, 1e-5),
    "float16": (1e-3, 1e-5),
    "bfloat16": (1.6e-2, 1e-5),
}
UNARY = ("exp", "sigmoid", "silu", "sqrt", "rsqrt", "neg")
BINARY = ("add", "sub", "mul", "div")
# 0.5% = 1/200. Use integer arithmetic at the acceptance boundary.
RANDOM_FAILURE_DENOMINATOR = 200


def decode(record, dtype, device="cpu"):
    """Recreate logical values, stride and storage offset without float conversion."""
    shape, stride, offset = record["shape"], record["stride"], record["offset"]
    if len(shape) != len(stride) or math.prod(shape) != len(record["bits"]):
        raise ValueError("invalid tensor metadata")
    if not record["bits"] or offset < 0 or any(s < 0 for s in stride):
        raise ValueError("empty samples or invalid layout")
    fmt = "I" if dtype == "float32" else "H"
    data = bytearray(struct.pack("<" + fmt * len(record["bits"]), *record["bits"]))
    values = torch.frombuffer(data, dtype=getattr(torch, dtype)).reshape(shape)
    size = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    storage = torch.empty(size, dtype=values.dtype, device=device)
    view = storage.as_strided(shape, stride, offset)
    view.copy_(values)
    return view


def reference(record):
    op, stage = record["name"].split("/")
    backward = stage in ("backward", "grad_a", "grad_b")
    a = decode(record["a"], record["dtype"], record["device"])
    a.requires_grad_(backward)
    if op in BINARY:
        b = decode(record["b"], record["dtype"], record["device"])
        b.requires_grad_(backward)
        y = getattr(torch, op)(a, b)
    else:
        y = torch.nn.functional.silu(a) if op == "silu" else getattr(torch, op)(a)
    if backward:
        g = decode(record["g"], record["dtype"], record["device"])
        y.backward(g)
        return (b.grad if stage == "grad_b" else a.grad).detach().cpu()
    return y.detach().cpu()


def math_reference(record):
    """Independent analytic formulas in FP64, rounded once to the output dtype.

    Current finite samples (FP32/FP16/BF16 inputs, exp inputs within [-200, 200])
    keep these intermediate calculations within FP64 range. No autograd is used.
    """
    op, stage = record["name"].split("/")
    a = decode(record["a"], record["dtype"]).double()
    backward = stage in ("backward", "grad_a", "grad_b")
    if backward:
        g = decode(record["g"], record["dtype"]).double()
    if op in BINARY:
        b = decode(record["b"], record["dtype"]).double()
        if not backward:
            if op == "add": result = a + b
            elif op == "sub": result = a - b
            elif op == "mul": result = a * b
            else: result = a / b
        elif op == "add": result = g
        elif op == "sub": result = g if stage == "grad_a" else -g
        elif op == "mul": result = g * (b if stage == "grad_a" else a)
        else: result = g / b if stage == "grad_a" else -(g * a) / (b * b)
    elif op == "exp":
        result = a.exp()
        if backward: result = g * result
    elif op in ("sigmoid", "silu"):
        t = (-a.abs()).exp()
        s = torch.where(a >= 0, 1 / (1 + t), t / (1 + t))
        if backward:
            ds = t / ((1 + t) * (1 + t))
            result = g * ds if op == "sigmoid" else g * (s + a * ds)
        else:
            result = s if op == "sigmoid" else a * s
    elif op == "sqrt":
        result = g / (2 * a.sqrt()) if backward else a.sqrt()
    elif op == "rsqrt":
        result = (-0.5 * g) / (a * a.sqrt()) if backward else 1 / a.sqrt()
    elif op == "neg":
        result = -g if backward else -a
    else:
        raise ValueError(f"missing mathematical reference for {op}")
    return result.to(getattr(torch, record["dtype"]))


def compare(actual, expected, dtype):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("output shape/dtype differs from reference")
    rtol, atol = TOLERANCES[dtype]
    a, b = actual.double().flatten(), expected.double().flatten()
    # Compute error in double; comparing two FP16 values must not itself overflow.
    close = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol,
                                   equal_nan=True, check_device=False)
        assertion_passed = True
    except AssertionError:
        assertion_passed = False
    if assertion_passed != bool(close.all()):
        raise RuntimeError("diagnostic mask disagrees with torch.testing.assert_close")
    finite = torch.isfinite(a) & torch.isfinite(b)
    failed = (~close).nonzero().flatten().tolist()
    return {
        "count": a.numel(),
        "failed": len(failed),
        "finite_failed": int((~close & finite).sum()),
        "nonfinite_failed": int((~close & ~finite).sum()),
        # These are diagnostics, not additional pass/fail thresholds.
        "zero_mismatch": int((finite & ((a == 0) != (b == 0))).sum()),
        "zero_sign_mismatch": int(((a == 0) & (b == 0) &
                                   (torch.signbit(a) != torch.signbit(b))).sum()),
        "both_nan": int((torch.isnan(a) & torch.isnan(b)).sum()),
        "same_inf": int((torch.isinf(a) & (a == b)).sum()),
        "max_finite_abs_error": float((a[finite] - b[finite]).abs().max())
        if finite.any() else None,
        "failed_indices": failed,
    }


def compare_dual(actual, mathematical, expected, dtype):
    math_result = compare(actual, mathematical, dtype)
    torch_result = compare(actual, expected, dtype)
    math_failed = set(math_result["failed_indices"])
    failed = sorted(math_failed.intersection(torch_result["failed_indices"]))
    # Classify the final failures, not all disagreements with either reference.
    finite = (torch.isfinite(actual.flatten()) & torch.isfinite(mathematical.flatten()) &
              torch.isfinite(expected.flatten()))
    finite_failed = sum(bool(finite[i]) for i in failed)
    return {
        "count": actual.numel(),
        "math_pass": actual.numel() - len(math_failed),
        "torch_only_pass": len(math_failed) - len(failed),
        "failed": len(failed), "failed_indices": failed,
        "finite_failed": finite_failed, "nonfinite_failed": len(failed) - finite_failed,
        "math_comparison": math_result, "torch_comparison": torch_result,
    }


def cohort_ranges(record):
    """Validate the producer's contiguous cohorts without slicing input tensors."""
    counts = record.get("cohorts", {})
    special = record["name"].endswith("/special")
    names = ("special",) if special else ("basic", "boundary", "random")
    if set(counts) != set(names) or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("missing or invalid sample cohorts")
    if not special and (counts["basic"] != 513 or counts["random"] != 4097):
        raise ValueError("expected 513 basic and 4097 random samples")
    if sum(counts.values()) != len(record["a"]["bits"]):
        raise ValueError("cohorts do not cover the complete input")
    start = 0
    ranges = {}
    for name in names:
        ranges[name] = (start, start + counts[name])
        start += counts[name]
    return ranges


def accept_cohorts(record, result):
    """Keep every element failure; decide acceptance independently per cohort."""
    ranges = cohort_ranges(record)
    if result["count"] != len(record["a"]["bits"]):
        raise ValueError("comparison count differs from cohort inventory")
    math_failed = set(result["math_comparison"]["failed_indices"])
    cohorts = {}
    for name, (start, end) in ranges.items():
        indices = [i for i in result["failed_indices"] if start <= i < end]
        count = end - start
        math_pass = count - sum(start <= i < end for i in math_failed)
        allowed = count // RANDOM_FAILURE_DENOMINATOR if name == "random" else 0
        cohorts[name] = {
            "start": start, "count": count, "math_pass": math_pass,
            "torch_only_pass": count - math_pass - len(indices),
            "failed": len(indices), "failed_indices": indices,
            "pass_rate": (count - len(indices)) / count,
            "allowed_failures": allowed, "accepted": len(indices) <= allowed,
        }
    return {"cohorts": cohorts, "accepted": all(c["accepted"] for c in cohorts.values())}


def validate_inventory(records, device):
    expected = {(dtype, f"{op}/{stage}") for dtype in TOLERANCES
                for op in UNARY + BINARY
                for stage in (("special", "forward", "backward") if op in UNARY
                              else ("special", "forward", "grad_a", "grad_b"))}
    seen = set()
    inputs = {}
    for record in records:
        key = record["dtype"], record["name"]
        if key not in expected or key in seen or record["device"] != device:
            raise ValueError(f"unexpected/duplicate case or wrong device: {key}")
        seen.add(key)
        cohort_ranges(record)
        op, stage = record["name"].split("/")
        group = record["dtype"], op, stage == "special"
        pair = record["a"], record.get("b"), record["cohorts"]
        if inputs.setdefault(group, pair) != pair:
            raise ValueError(f"forward/backward inputs differ: {key}")
    if seen != expected:
        raise ValueError(f"missing cases: {sorted(expected - seen)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("SKIP: PyTorch CUDA device unavailable", flush=True)
        return 77
    torch.set_num_threads(1)
    denormals_preserved = torch.set_flush_denormal(False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.report_dir / "samples.jsonl"
    report_path = args.report_dir / "report.json"
    # Remove stale results before starting; never present an earlier run as current.
    report_path.unlink(missing_ok=True)
    sample_path.unlink(missing_ok=True)
    print(f"PyTorch {torch.__version__} | {args.device} | exporting mini-infer samples",
          flush=True)
    process = subprocess.run([str(args.executable.resolve()), str(sample_path.resolve())])
    if process.returncode:
        return process.returncode
    with sample_path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    validate_inventory(records, args.device)
    report = {
        "acceptance": "math_or_torch_per_element",
        "group_acceptance": "basic_boundary_special_all_pass_random_99.5_percent",
        "random_failure_denominator": RANDOM_FAILURE_DENOMINATOR,
        "math_reference": "FP64 analytic formulas rounded once to output dtype",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "python": platform.python_version(), "platform": platform.platform(),
        "device": args.device, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "cpu_flush_denormal_disabled": denormals_preserved,
        "torch_build": torch.__config__.show(),
        "threads": torch.get_num_threads(), "tolerances": TOLERANCES,
        "equal_nan": True, "samples": str(sample_path.resolve()),
        "executable": str(args.executable.resolve()), "groups": [],
    }
    for record in records:
        actual = decode(record["actual"], record["dtype"])
        mathematical = math_reference(record)
        expected = reference(record)
        result = compare_dual(actual, mathematical, expected, record["dtype"])
        result.update(accept_cohorts(record, result))
        result.update(dtype=record["dtype"], name=record["name"])
        result["examples"] = []
        for i in result["failed_indices"][:6]:
            cohort = next(name for name, c in result["cohorts"].items()
                          if c["start"] <= i < c["start"] + c["count"])
            example = {"index": i, "cohort": cohort,
                       "actual": str(actual.flatten()[i].item()),
                       "math_rounded": str(mathematical.flatten()[i].item()),
                       "torch": str(expected.flatten()[i].item())}
            for field in ("a", "b", "g"):
                if field in record:
                    example[field] = str(decode(record[field], record["dtype"])
                                         .flatten()[i].item())
            result["examples"].append(example)
        report["groups"].append(result)
        summary = " ".join(f"{name}={c['count'] - c['failed']}/{c['count']}"
                           for name, c in result["cohorts"].items())
        print(f"{record['dtype']:8} {record['name']:18} "
              f"{'PASS' if result['accepted'] else 'FAIL'} {summary}",
              flush=True)
    report["comparisons"] = sum(g["count"] for g in report["groups"])
    report["failed"] = sum(g["failed"] for g in report["groups"])
    report["math_pass"] = sum(g["math_pass"] for g in report["groups"])
    report["torch_only_pass"] = sum(g["torch_only_pass"] for g in report["groups"])
    report["failed_groups"] = sum(not g["accepted"] for g in report["groups"])
    report["accepted"] = report["failed_groups"] == 0
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"TOTAL: math_pass={report['math_pass']} torch_only_pass={report['torch_only_pass']} "
          f"failed={report['failed']} / {report['comparisons']} comparisons; "
          f"failed_groups={report['failed_groups']} / {len(report['groups'])}")
    print(f"Report: {report_path.resolve()}")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
