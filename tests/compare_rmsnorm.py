"""Check fixed RMSNorm cases against FP64 formulas or same-device PyTorch."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from compare_torch import TOLERANCES, compare_dual, decode


CASES = ("vector", "matrix", "tensor3d", "strided", "single_feature", "zeros",
         "large_eps", "small_input", "large_input", "wide")
STAGES = ("forward", "grad_x", "grad_weight")


def analytic(x, weight, g, eps):
    """FP64 independent formulas; sum weight gradients over every leading axis."""
    x, weight, g = x.double(), weight.double(), g.double()
    h = x.shape[-1]
    r = 1 / ((x * x).mean(-1, keepdim=True) + eps).sqrt()
    y = x * r * weight
    dot = (g * weight * x).sum(-1, keepdim=True)
    dx = g * weight * r - x * r * r * r * dot / h
    dw = (g * x * r).reshape(-1, h).sum(0)
    return dict(zip(STAGES, (y, dx, dw)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("SKIP: PyTorch CUDA device unavailable")
        return 77
    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    samples = args.report_dir / "samples.jsonl"
    output = args.report_dir / "report.json"
    output.unlink(missing_ok=True)
    samples.unlink(missing_ok=True)
    process = subprocess.run([str(args.executable.resolve()), str(samples.resolve())])
    if process.returncode:
        return process.returncode
    records = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines()]
    expected = {(dtype, case) for dtype in TOLERANCES for case in CASES}
    seen = set()
    for record in records:
        key = record["dtype"], record["case"]
        if key not in expected or key in seen or record["device"] != args.device:
            raise ValueError(f"unexpected/duplicate RMSNorm case or wrong device: {key}")
        seen.add(key)
    if seen != expected:
        raise ValueError(f"missing RMSNorm cases: {sorted(expected - seen)}")
    report = {"torch": torch.__version__, "torch_git": torch.version.git_version,
              "device": args.device, "tolerances": TOLERANCES,
              "acceptance": "math_or_torch_per_element_all_fixed_cases_required",
              "groups": []}
    for record in records:
        dtype, eps = record["dtype"], record["eps"]
        x = decode(record["x"], dtype, args.device).requires_grad_(True)
        w = decode(record["weight"], dtype, args.device).requires_grad_(True)
        g = decode(record["g"], dtype, args.device)
        y = torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)
        y.backward(g)
        reference = dict(zip(STAGES, (y.detach().cpu(), x.grad.cpu(), w.grad.cpu())))
        mathematical = analytic(x.detach().cpu(), w.detach().cpu(), g.cpu(), eps)
        for stage in STAGES:
            actual = decode(record[stage], dtype)
            math_rounded = mathematical[stage].to(getattr(torch, dtype))
            result = compare_dual(actual, math_rounded, reference[stage], dtype)
            result.update(dtype=dtype, case=record["case"], stage=stage)
            result["examples"] = [
                {"index": i, "actual": str(actual.flatten()[i].item()),
                 "math_rounded": str(math_rounded.flatten()[i].item()),
                 "torch": str(reference[stage].flatten()[i].item())}
                for i in result["failed_indices"][:6]]
            report["groups"].append(result)
            print(f"{dtype:8} {record['case']:15} {stage:11} "
                  f"failed={result['failed']}/{result['count']}")
    report["comparisons"] = sum(g["count"] for g in report["groups"])
    report["failed"] = sum(g["failed"] for g in report["groups"])
    report["accepted"] = report["failed"] == 0
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"TOTAL: failed={report['failed']}/{report['comparisons']}; report={output}")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
