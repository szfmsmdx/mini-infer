"""Compare FP32 optimizer update traces with PyTorch and FP64 formulas."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from compare_torch import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    traces = args.report_dir / "traces.jsonl"
    output = args.report_dir / "report.json"
    traces.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("SKIP: PyTorch CUDA device unavailable")
        return 77
    torch.set_num_threads(1)
    process = subprocess.run([str(args.executable.resolve()), str(traces.resolve())])
    if process.returncode:
        return process.returncode
    records = [json.loads(line) for line in traces.read_text(encoding="utf-8").splitlines()]
    expected_names = {"sgd", "sgd_momentum", "adam", "adam_custom", "adam_beta_zero"}
    if len(records) != len(expected_names) or {r["name"] for r in records} != expected_names:
        raise ValueError("optimizer trace inventory mismatch")

    report = {"torch": torch.__version__, "device": args.device,
              "rtol": 1e-5, "atol": 1e-6, "groups": []}
    for record in records:
        if record["device"] != args.device or len(record["steps"]) != 10:
            raise ValueError("optimizer trace device or step count mismatch")
        if len(record["initial"]) != 2:
            raise ValueError("optimizer trace must contain two parameters")
        # Match the C++ float hyperparameters exactly.
        lr, momentum, beta1, beta2, eps = [
            torch.tensor(record[key], dtype=torch.float32).item()
            for key in ("lr", "momentum", "beta1", "beta2", "eps")
        ]
        parameters = [decode(p, "float32", args.device).requires_grad_(True)
                      for p in record["initial"]]
        is_adam = record["name"].startswith("adam")
        if is_adam:
            optimizer = torch.optim.Adam(parameters, lr=lr, betas=(beta1, beta2),
                                         eps=eps, foreach=False, fused=False)
        else:
            optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                                        foreach=False, fused=False)
        mathematical = [p.detach().cpu().double().clone() for p in parameters]
        first = [torch.zeros_like(p) for p in mathematical]
        second = [torch.zeros_like(p) for p in mathematical]
        steps = [0, 0]
        for iteration, record_step in enumerate(record["steps"], start=1):
            gradients = record_step["gradients"]
            actual_parameters = record_step["parameters"]
            if len(gradients) != 2 or len(actual_parameters) != 2:
                raise ValueError("optimizer trace parameter/gradient count mismatch")
            if (gradients[1] is None) != (iteration <= 3) or gradients[0] is None:
                raise ValueError("optimizer missing-gradient schedule mismatch")
            optimizer.zero_grad(set_to_none=False)
            for i, gradient in enumerate(gradients):
                if gradient is None:
                    if parameters[i].grad is not None:
                        raise ValueError("a previously allocated gradient cannot become absent in this trace")
                    continue
                parameters[i].grad = decode(gradient, "float32", args.device)
                g = parameters[i].grad.cpu().double()
                if is_adam:
                    steps[i] += 1
                    first[i] = beta1 * first[i] + (1 - beta1) * g
                    second[i] = beta2 * second[i] + (1 - beta2) * g.square()
                    direction = (first[i] / (1 - beta1 ** steps[i])) / (
                        (second[i] / (1 - beta2 ** steps[i])).sqrt() + eps)
                elif momentum:
                    first[i] = momentum * first[i] + g
                    direction = first[i]
                else:
                    direction = g
                mathematical[i] -= lr * direction
            optimizer.step()

            for i, actual_record in enumerate(actual_parameters):
                actual = decode(actual_record, "float32")
                reference = parameters[i].detach().cpu()
                math_reference = mathematical[i].float()
                if actual.shape != reference.shape:
                    raise ValueError("optimizer output shape mismatch")
                torch_ok = torch.isclose(actual, reference, rtol=report["rtol"], atol=report["atol"])
                math_ok = torch.isclose(actual, math_reference, rtol=report["rtol"], atol=report["atol"])
                passed = torch_ok & math_ok & torch.isfinite(actual)
                failed = (~passed).flatten().nonzero().flatten().tolist()
                report["groups"].append({
                    "name": record["name"], "step": iteration, "parameter": i,
                    "count": actual.numel(), "failed": len(failed), "failed_indices": failed,
                    "max_torch_error": float((actual - reference).abs().max()),
                    "max_math_error": float((actual - math_reference).abs().max()),
                })
        failed_count = sum(g["failed"] for g in report["groups"] if g["name"] == record["name"])
        print(f"{record['name']}: failed={failed_count}/50")
    report["comparisons"] = sum(g["count"] for g in report["groups"])
    report["failed"] = sum(g["failed"] for g in report["groups"])
    report["accepted"] = report["failed"] == 0
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"TOTAL: failed={report['failed']}/{report['comparisons']}; report={output}")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
