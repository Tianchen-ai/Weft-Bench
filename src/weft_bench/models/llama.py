from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from ..execution import Executor
from ..results import save
from ..types import Result


def replacement_counts(text):
    snapshots = re.findall(
        r"WB_COUNTER sample=(\d+) point=(begin|end) type=(\S+) dots=(\d+) quantizes=(\d+) ime=(\d+)",
        text,
    )
    if snapshots:
        samples = {}
        for sample, point, kind, dots, quant, ime in snapshots:
            samples.setdefault(int(sample), {}).setdefault(point, {})[kind] = (
                int(dots),
                int(quant),
                int(ime),
            )
        deltas = []
        for sample, points in sorted(samples.items()):
            if "begin" not in points or "end" not in points:
                continue
            by_type = {
                kind: tuple(
                    end[i] - points["begin"].get(kind, (0, 0, 0))[i] for i in range(3)
                )
                for kind, end in points["end"].items()
            }
            values = list(by_type.values())
            if any(v < 0 for triple in values for v in triple):
                raise ValueError("counter reset inside measurement window")
            deltas.append(
                {
                    "sample": sample,
                    "vec_dot": sum(v[0] for v in values),
                    "quantize": sum(v[1] for v in values),
                    "ime": sum(v[2] for v in values),
                    "by_type": {
                        kind: dict(zip(("vec_dot", "quantize", "ime"), value))
                        for kind, value in by_type.items()
                    },
                }
            )
        per_type = {}
        for sample in deltas:
            for kind, counts in sample["by_type"].items():
                total = per_type.setdefault(kind, dict.fromkeys(("vec_dot", "quantize", "ime"), 0))
                for key, value in counts.items():
                    total[key] += value
        return {
            key: sum(d[key] for d in deltas) for key in ("vec_dot", "quantize", "ime")
        } | {"observed": bool(deltas), "samples": deltas, "scope": "measured-windows", "by_type": per_type}
    dots = re.findall(r"WEFT_CALLS type=(\S+) dots=(\d+) quantizes=(\d+)", text)
    ime = [int(v) for v in re.findall(r"WEFT_IME_CALLS (\d+)", text)]
    groups = defaultdict(list)
    for weight, d, q in dots:
        groups[weight].append((int(d), int(q)))
    # Older logs only record sample ends: exclude the warmup-containing first snapshot.
    return {
        "vec_dot": sum(v[-1][0] - v[0][0] for v in groups.values() if len(v) > 1),
        "quantize": sum(v[-1][1] - v[0][1] for v in groups.values() if len(v) > 1),
        "ime": ime[-1] - ime[0] if len(ime) > 1 else 0,
        "observed": any(len(v) > 1 for v in groups.values()) or len(ime) > 1,
        "scope": "historical-sample-deltas-first-window-unavailable",
    }


def summarize_samples(text, windows):
    weights = defaultdict(int)
    for line in text.splitlines():
        match = re.match(
            r"\s*([\d.]+):\s+(\d+)\s+[0-9a-f]+\s+(.*)\s+\(([^()]*)\)\s*$", line
        )
        if match:
            time, period, symbol, dso = match.groups()
            if any(start <= float(time) <= end for start, end in windows):
                weights[(symbol, dso)] += int(period)
    total = sum(weights.values())
    return [
        {"symbol": key[0], "dso": key[1], "percent": value * 100 / total}
        for key, value in sorted(weights.items(), key=lambda item: -item[1])[:3]
    ]


def profile_hotspots(executor, perf, path, log):
    windows = [
        tuple(map(float, pair))
        for pair in re.findall(r"PROFILE_WINDOW ([\d.]+) ([\d.]+)", log)
    ]
    if not windows:
        return (
            [],
            "llama-bench has no PROFILE_WINDOW markers; phase-only hotspots unavailable",
        )
    result = executor.command(
        [perf, "script", "-i", path, "-F", "time,period,ip,sym,dso"], check=False
    )
    if result.returncode:
        return [], result.stderr
    return summarize_samples(result.stdout, windows), None


def benchmark_model(
    name,
    implementations,
    target,
    *,
    phases="both",
    repetitions=3,
    profile=False,
    output,
    build=False,
    validate=False,
    validation_reference=None,
):
    settings = target.settings.get("llama", {})
    models = settings.get("models", {})
    if name not in models:
        raise ValueError(f"model {name!r} is not in target.llama.models")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rows = []
    if build:
        executor = Executor(target, Path(output) / "build")
        for implementation in implementations:
            config = settings["implementations"][implementation]
            commands = config.get("build_commands")
            if not commands:
                raise ValueError(f"{implementation} has no configured build_commands")
            for command in commands:
                if not isinstance(command, list):
                    raise ValueError("build_commands must contain argv lists")
                executor.command(
                    command,
                    env={**settings.get("env", {}), **config.get("env", {})},
                    stage="compile",
                )
    phase_names = ("prefill", "decode") if phases == "both" else (phases,)
    for phase in phase_names:
        phase_options = {
            "prefill": {"prompt": 128, "generate": 0, "depth": 0},
            "decode": {"prompt": 0, "generate": 32, "depth": 512},
        }[phase]
        phase_options.update(settings.get("phases", {}).get(phase, {}))
        for implementation in implementations:
            if implementation not in settings.get("implementations", {}):
                raise ValueError(
                    f"model implementation {implementation!r} needs binary/env/package configuration"
                )
            config = settings["implementations"][implementation]
            directory = Path(output) / "artifacts" / name / phase / implementation
            deployment = {
                **target.settings.get("deployment", {}),
                "cpus": settings.get("cpus", []),
            }
            executor = Executor(
                replace(target, settings={**target.settings, "deployment": deployment}),
                directory,
            )
            threads = settings.get("threads", target.settings.get("threads", 1))
            params = {
                **phase_options,
                "threads": threads,
                "batch": settings.get("batch", 2048),
                "ubatch": settings.get("ubatch", 512),
                "kv_k": settings.get("kv_k", "f16"),
                "kv_v": settings.get("kv_v", "f16"),
                "extra_args": settings.get("args", []),
            }
            row = Result(
                name + "/" + phase,
                implementation,
                target.name,
                dict(phase_options),
                "llama-" + phase,
                "pending",
                correctness="not_checked",
            )
            row.details = {
                "input_policy": models[name],
                "dtype": "GGUF",
                "phase": phase,
                "measurement": {"repetitions": repetitions, "profile": profile},
                "execution": {**params, "cpus": deployment["cpus"]},
                "model": models[name],
                "binary": config["binary"],
            }
            command = [
                config["binary"],
                "-m",
                models[name],
                "-p",
                str(params["prompt"]),
                "-n",
                str(params["generate"]),
                "-d",
                str(params["depth"]),
                "-t",
                str(threads),
                "-b",
                str(params["batch"]),
                "-ub",
                str(params["ubatch"]),
                "-ctk",
                params["kv_k"],
                "-ctv",
                params["kv_v"],
                "-r",
                str(repetitions),
                "-o",
                "json",
                *params["extra_args"],
            ]
            environment = {**settings.get("env", {}), **config.get("env", {})}
            if config.get("package"):
                environment["WEFT_GGML_PACKAGE"] = config["package"]
            # Remove inherited integration settings before applying this implementation.
            command = [
                "env",
                "-u",
                "WEFT_GGML_PACKAGE",
                "-u",
                "LD_PRELOAD",
                "-u",
                "SPACEMIT_PERFER_CORE_ID",
                *[f"{key}={value}" for key, value in environment.items()],
                *command,
            ]
            try:
                if profile:
                    perf = settings.get("perf", "perf")
                    executor.command(["mkdir", "-p", executor.workspace])
                    trace = executor.workspace + "/profile.data"
                    command = [
                        perf,
                        "record",
                        "--no-buildid",
                        "--clockid",
                        "mono",
                        "-e",
                        "cpu-clock:u",
                        "-F",
                        "199",
                        "-o",
                        trace,
                        "--",
                        *command,
                    ]
                print(
                    f"Running {name}/{phase}/{implementation} on {target.name}",
                    flush=True,
                )
                result = executor.command(command, affinity=True, check=False)
                (directory / "llama.json").write_text(result.stdout)
                (directory / "stderr.log").write_text(result.stderr)
                if result.returncode:
                    row.status, row.error = "run_failed", result.stderr[-6000:]
                else:
                    native = json.loads(result.stdout)
                    if len(native) != 1:
                        raise ValueError("one llama-bench phase result required")
                    native = native[0]
                    expected_fields = {
                        "n_prompt": params["prompt"],
                        "n_gen": params["generate"],
                        "n_depth": params["depth"],
                        "n_threads": threads,
                        "n_batch": params["batch"],
                        "n_ubatch": params["ubatch"],
                        "type_k": params["kv_k"],
                        "type_v": params["kv_v"],
                    }
                    for key, value in expected_fields.items():
                        if native.get(key) != value:
                            raise ValueError(
                                f"llama-bench returned {key}={native.get(key)!r}, expected {value!r}"
                            )
                    row.samples_ms = [v / 1e6 for v in native["samples_ns"]]
                    if not row.samples_ms or any(
                        not math.isfinite(v) or v <= 0 for v in row.samples_ms
                    ):
                        raise ValueError("invalid llama phase timing")
                    row.median_ms = statistics.median(row.samples_ms)
                    tokens = params["prompt"] or params["generate"]
                    row.details["tokens_per_second"] = tokens * 1000 / row.median_ms
                    row.details["native"] = native
                    counts = replacement_counts(result.stderr)
                    row.details["replacement_counts"] = counts
                    row.details["replacement"] = (
                        "hit"
                        if sum(counts[k] for k in ("vec_dot", "quantize", "ime"))
                        else ("not_hit" if counts["observed"] else "not_observed")
                    )
                    row.status = "ok"
                    if profile:
                        hotspots, error = profile_hotspots(
                            executor, perf, trace, result.stderr
                        )
                        from .hotspots import link_hotspots
                        from ..results import read

                        operator_rows = (
                            read(settings["operator_results"])
                            if settings.get("operator_results")
                            else []
                        )
                        info_path = settings.get("model_info", {}).get(name)
                        model_info = (
                            json.loads(Path(info_path).read_text())
                            if info_path
                            else None
                        )
                        hotspots = link_hotspots(
                            hotspots,
                            target=target.name,
                            phase=phase,
                            call_sites=settings.get("call_sites", {})
                            .get(name, {})
                            .get(phase, {}),
                            operator_rows=operator_rows,
                            model_info=model_info,
                        )
                        row.details.update(hotspots=hotspots, profile_path=trace)
                        if error:
                            row.details["profile_error"] = error
            except Exception as error:
                row.status, row.error = "run_failed", str(error)
            (directory / "commands.json").write_text(
                json.dumps(executor.commands, indent=2)
            )
            rows.append(row)
            save(rows, output)
    if validate:
        from .validation import validate_model, attach_validation

        for implementation in implementations:
            if implementation == "normal":
                continue
            report = validate_model(name, target, implementation,
                                    output=Path(output) / "validation" / implementation,
                                    reference_output=validation_reference or Path(output) / "validation" / "reference")
            attach_validation(rows, report, implementation)
        save(rows, output)
    return rows
