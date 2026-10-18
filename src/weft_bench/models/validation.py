"""Small, untimed model-logit comparison using the normal and selected llama libraries."""
from array import array
from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import sys

from ..adapters.native import NATIVE, flags_for
from ..execution import Executor


def _run(name, target, implementation, parameters, directory, token_reference=None):
    settings = target.settings["llama"]
    config = settings["implementations"][implementation]
    deployment = {**target.settings.get("deployment", {}), "cpus": settings.get("cpus", [])}
    executor = Executor(replace(target, settings={**target.settings, "deployment": deployment}), directory)
    source = config.get("source", target.settings.get("ggml", {}).get("source"))
    if not source:
        raise ValueError("model validation needs a configured llama.cpp source/include directory")
    library = str(Path(config["binary"]).parent)
    helper = executor.upload(NATIVE / "llama_validation.cpp", "llama_validation.cpp")
    tools, flags = flags_for(target)
    if config.get("package") and config.get("weft_include"):
        flags += ["-DWB_WEFT_CHECK", "-I" + config["weft_include"],
                  "-I" + target.settings["triton_runtime"]["include"]]
    executable = executor.workspace + "/validate-model"
    command = [*tools["cxx"], *flags, "-std=c++17", "-I" + source + "/include",
               "-I" + source + "/ggml/include", helper, "-L" + library, "-Wl,-rpath," + library,
               "-lllama", "-lggml", "-lggml-cpu", "-lggml-base", *tools.get("ldflags", []), "-o", executable]
    done = executor.command(command, stage="model_validation_build", check=False)
    (directory / "build.log").write_text(done.stdout + done.stderr)
    (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2))
    if done.returncode:
        raise RuntimeError("model_validation_build: " + done.stderr[-6000:])
    tokens = executor.upload(token_reference, "reference.tokens.bin") if token_reference else "-"
    prefix = executor.workspace + "/output"
    environment = {**settings.get("env", {}), **config.get("env", {})}
    if config.get("package"):
        environment["WEFT_GGML_PACKAGE"] = config["package"]
    command = ["env", "-u", "WEFT_GGML_PACKAGE", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH",
               "-u", "SPACEMIT_PERFER_CORE_ID", *[str(k) + "=" + str(v) for k, v in environment.items()],
               executable, settings["models"][name], prefix, tokens, str(parameters["threads"]),
               str(parameters["prompt_tokens"]), str(parameters["decode_steps"])]
    done = executor.command(command, affinity=True, stage="model_validation_run", check=False)
    (directory / "stdout.log").write_text(done.stdout)
    (directory / "stderr.log").write_text(done.stderr)
    (directory / "commands.json").write_text(json.dumps(executor.commands, indent=2))
    if done.returncode:
        raise RuntimeError("model_validation_run: " + done.stderr[-6000:])
    records = [json.loads(line[len("WB_MODEL_CHECK "):]) for line in done.stdout.splitlines() if line.startswith("WB_MODEL_CHECK ")]
    if len(records) != 1:
        raise ValueError("missing model validation output")
    for suffix in ("logits.bin", "tokens.bin"):
        executor.download(prefix + "." + suffix, directory / suffix)
    records[0]["replacement_snapshots"] = [json.loads(line[len("WB_MODEL_HITS "):]) for line in done.stdout.splitlines() if line.startswith("WB_MODEL_HITS ")]
    return records[0]


def _logits(path):
    with Path(path).open("rb") as stream:
        rows, columns = struct.unpack("<II", stream.read(8))
        values = array("f")
        values.frombytes(stream.read())
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != rows * columns:
        raise ValueError("model logit output length mismatch")
    return rows, columns, values


def validate_model(name, target, implementation="weft", *, output, reference_output=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference = Path(reference_output) if reference_output else output / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    settings = target.settings["llama"]
    selected = settings.get("validation", {})
    parameters = {"prompt_tokens": selected.get("prompt_tokens", 16), "decode_steps": selected.get("decode_steps", 2),
                  "threads": settings.get("threads", 1)}
    tolerance = {"rtol": selected.get("rtol", 1e-3), "atol": selected.get("atol", 1e-3)}
    if any(not math.isfinite(v) or v < 0 for v in tolerance.values()):
        raise ValueError("model validation tolerances must be finite and nonnegative")
    contract = {"model": settings["models"][name], "target": target.name, "parameters": parameters,
                "normal_binary": settings["implementations"]["normal"]["binary"], "protocol": "fixed-prompt-teacher-forced-logits-v1"}
    report = {"contract": contract, "tolerance": tolerance, "reference": str(reference.resolve()),
              "candidate_package": settings["implementations"][implementation].get("package"),
              "scope": "full-vocabulary last-prefill logits and teacher-forced decode logits; separate from phase timing"}
    try:
        reference_record = reference / "reference.json"
        if reference_record.exists():
            old = json.loads(reference_record.read_text())
            if old["contract"] != contract:
                raise ValueError("model numerical reference input/execution differs; choose a corresponding reference directory")
            report["reference_reused"] = True
        else:
            details = _run(name, target, "normal", parameters, reference)
            reference_record.write_text(json.dumps({"contract": contract, "details": details}, indent=2))
            report["reference_reused"] = False
        candidate = _run(name, target, implementation, parameters, output, reference / "tokens.bin")
        report["candidate_execution"] = candidate
        if report["candidate_package"] and not candidate["replacement_observed"]:
            raise ValueError("configured package had no observed replacement in the numerical check; configure weft_include and inspect its selected library/path")
        before_rows, before_columns, before = _logits(reference / "logits.bin")
        rows, columns, after = _logits(output / "logits.bin")
        if (rows, columns) != (before_rows, before_columns):
            raise ValueError("model logit shapes differ")
        metrics = []
        for row in range(rows):
            a, b = before[row * columns:(row + 1) * columns], after[row * columns:(row + 1) * columns]
            if any(not math.isfinite(v) for v in (*a, *b)):
                raise ValueError("model logits contain nonfinite values")
            errors = [abs(x - y) for x, y in zip(a, b)]
            outside = sum(error > tolerance["atol"] + tolerance["rtol"] * abs(x) for error, x in zip(errors, a))
            expected_token = max(range(columns), key=a.__getitem__)
            actual_token = max(range(columns), key=b.__getitem__)
            metrics.append({"stage": "prefill" if row == 0 else "decode", "step": row,
                            "max_abs_error": max(errors), "rms_error": math.sqrt(sum(e * e for e in errors) / columns),
                            "outside_tolerance": outside, "reference_top1": expected_token,
                            "candidate_top1": actual_token, "top1_matches": expected_token == actual_token})
        ok = all(m["outside_tolerance"] == 0 and m["top1_matches"] for m in metrics)
        report.update(status="ok" if ok else "numeric_mismatch", correctness="sampled-logits-within-tolerance" if ok else "failed",
                      vocabulary=columns, metrics=metrics)
    except Exception as error:
        report.update(status="validation_failed", correctness="not_checked", error=str(error))
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    return report


def attach_validation(rows, report, implementation="weft"):
    for row in rows:
        if row.implementation != implementation:
            continue
        row.details["model_validation"] = report
        row.correctness = report["correctness"]
        if report["status"] != "ok" and row.status == "ok":
            row.status = "correctness_failed" if report["status"] == "numeric_mismatch" else "validation_failed"
            row.error = report.get("error", "sampled model logits differ from the normal implementation")
    return rows
