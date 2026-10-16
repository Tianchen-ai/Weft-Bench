"""Outer build/deployment orchestration for the transport-free native bench core."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import tempfile

from .core import worker_source
from .diagnostics import save_request, failure_stage
from .execution import Executor, ExecutionError
from .inputs import Inputs, native_inputs
from .libraries import ggml_call, weft_call
from .types import Measurement, Result
from .workloads import native_workload, ports
from .adapters.native import NATIVE, compile_artifact, flags_for, ggml_flags
from .operators.implementations import resolve_binding

BUILTINS = {
    "weft_bench.adapters.native:WeftAdapter": "weft",
    "weft_bench.adapters.native:ArtifactAdapter": "artifact",
    "weft_bench.adapters.native:GGMLAdapter": "ggml",
}


def mode(name, target):
    factory = target.implementations.get(name, {}).get("factory")
    return (
        BUILTINS.get(factory)
        if factory
        else {"weft": "weft", "ggml": "ggml", "artifact": "artifact"}.get(name)
    )


def _failed(row, error, stage="prepare"):
    stage = error.stage if isinstance(error, ExecutionError) else stage
    row.status = (
        "unsupported" if isinstance(error, NotImplementedError) else stage + "_failed"
    )
    row.error = str(error)
    row.details["failure_stage"] = stage
    row.details["compiler_stage"] = failure_stage(row.error)


def _compile_details(row, directory):
    if (directory / "compile.json").exists():
        row.details["build"] = json.loads((directory / "compile.json").read_text())
    if (directory / "canonical.mlir").exists():
        row.details.setdefault("artifacts", {})["canonical"] = str(
            (directory / "canonical.mlir").resolve()
        )
    if (directory / "effective-options.json").exists():
        row.details.setdefault("build", {})["effective_options"] = json.loads(
            (directory / "effective-options.json").read_text()
        )


def run_operator(
    case,
    implementations,
    target,
    measurement=None,
    *,
    output=None,
    inputs=None,
    save_outputs=False,
):
    measurement = measurement or Measurement()
    root = Path(output or tempfile.mkdtemp(prefix="weft-bench-"))
    # Multiple compiler candidates are separate comparisons; each can reuse GGML observations.
    if sum(mode(n, target) != "ggml" for n in implementations) > 1:
        return [
            r
            for n in implementations
            for r in run_operator(
                case,
                [n],
                target,
                measurement,
                output=root,
                inputs=inputs,
                save_outputs=save_outputs,
            )
        ]
    if len(set(implementations)) != len(implementations):
        raise ValueError("implementation names must be distinct")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in implementations):
        raise ValueError("implementation names must be identifiers")
    selected = inputs or Inputs()
    case = selected.resolve(case)
    worker = Executor(target, root / case.name / "worker")
    files = selected.materialize(worker.local / "inputs", ports(case))
    input_record = selected.describe(files)
    # Input content describes the scenario, not a build-hash approval gate.
    identity = {
        "generator": "ggml-workload-v3",
        "seed": selected.seed,
        "parameters": (
            {"epsilon": selected.parameters.get("epsilon", 1e-5)}
            if case.name in ("rms_norm", "layer_norm")
            else {}
        ),
    }
    identity["buffers"] = {
        k: {p: v for p, v in value.items() if p != "path"} for k, value in files.items()
    }
    case = replace(case, data={**case.data, "inputs": input_record})
    rows = []
    calls = []
    objects = []
    for name in implementations:
        directory = root / case.name / name
        directory.mkdir(parents=True, exist_ok=True)
        row = Result(case.name, name, target.name, case.shape, case.scope, "pending")
        row.details = {
            "measurement": {**asdict(measurement), "protocol": "native-workload-v3"},
            "input_policy": json.dumps(identity, sort_keys=True),
            "inputs": input_record,
            "ports": [asdict(p) for p in ports(case)],
            "dtype": case.dtype,
            "phase": case.phase,
            "execution": {
                "cpus": worker.cpus,
                "threads": target.settings.get("threads", 1),
                "isa": target.settings.get("target", {}),
                "env": worker.env,
            },
            "diagnostics": str(directory.resolve()),
            "worker": str(worker.local.resolve()),
            "native_build": target.settings.get("toolchain", {}),
        }
        row.details["replay"] = save_request(case, name, target, measurement, directory)
        rows.append(row)
    try:
        if target.settings.get("threads", 1) != 1:
            raise ValueError(
                "operator workers use one thread; model threads are separate"
            )
        if case.family == "ime" and (
            "spacemit-ime1" not in target.settings.get("capabilities", [])
            or len(worker.cpus) != 1
        ):
            raise NotImplementedError(
                "packed IME requires spacemit-ime1 and one eligible CPU"
            )
        workload = native_workload(case)
        tools, flags = flags_for(target)
        includes, libraries = ggml_flags(target)
        for header in NATIVE.glob("*.h"):
            worker.upload(header, header.name)
        for role, info in files.items():
            info["remote"] = worker.upload(info["path"], "inputs/" + role + ".bin")
    except Exception as error:
        for row in rows:
            _failed(row, error)
        return rows
    for row in rows:
        name = row.implementation
        options = dict(target.implementations.get(name, {}))
        directory = root / case.name / name
        try:
            if mode(name, target) == "ggml":
                if case.name == "rope_split":
                    from .adapters.reference import prepare_dense_reference

                    extra, _ = prepare_dense_reference(case, target, worker)
                    objects.extend(extra)
                call = ggml_call(name, case)
                row.details["reference_build"] = target.settings["ggml"]
            else:
                if mode(name, target) == "artifact":
                    configured = options.get("artifacts", {}).get(
                        case.name, options.get("path")
                    )
                    if not configured:
                        raise ValueError("Weft artifact path is required")
                    artifact = Path(configured).expanduser()
                    if not (artifact / "manifest.json").is_file():
                        raise ValueError(f"{artifact} is not a Weft artifact directory")
                    snapshot = directory / "artifact"
                    if artifact.resolve() != snapshot.resolve():
                        shutil.copytree(artifact, snapshot)
                        artifact = snapshot
                else:
                    local = Executor(target, directory)
                    artifact = compile_artifact(case, target, options, local)
                    (directory / "compile-commands.json").write_text(
                        json.dumps(local.commands, indent=2)
                    )
                manifest = json.loads((artifact / "manifest.json").read_text())
                if any(
                    manifest["target"].get(k) != v
                    for k, v in target.settings["target"].items()
                ):
                    raise ValueError("artifact target differs from execution target")
                metadata = manifest["kernel"]
                if case.family == "projection" and any(
                    a["name"] == "W" and "I[" in a.get("encoding", "")
                    for a in metadata["arguments"]
                ):
                    raise NotImplementedError(
                        "projection expects ordinary GGML weight blocks; select a matching source binding"
                    )
                source = worker.upload(artifact / "kernel.c", name + "/kernel.c")
                header = worker.upload(artifact / "kernel.h", name + "-kernel.h")
                worker.upload(artifact / "kernel.h", name + "/kernel.h")
                obj = worker.workspace + "/" + name + ".o"
                worker.command(
                    [*tools["cc"], *flags, "-std=c11", "-c", source, "-o", obj],
                    stage="native_compile",
                )
                objects.append(obj)
                binding = resolve_binding(case, target, options)
                call = weft_call(
                    name, case, workload, metadata, binding.get("arguments", {})
                )
                row.details["artifacts"] = {"weft": str(artifact.resolve())}
            calls.append(call)
        except Exception as error:
            _failed(row, error)
        finally:
            _compile_details(row, directory)
    if calls:
        try:
            source = worker.write(
                "worker.cpp",
                worker_source(
                    workload,
                    calls,
                    native_inputs(selected, files),
                    save_outputs=save_outputs,
                ),
            )
            binary = worker.workspace + "/worker"
            worker.command(
                [
                    *tools["cxx"],
                    *flags,
                    "-std=c++17",
                    *workload.defines,
                    *includes,
                    "-I" + worker.workspace,
                    source,
                    *objects,
                    *tools.get("ldflags", []),
                    *libraries,
                    "-lm",
                    "-ldl",
                    "-pthread",
                    "-o",
                    binary,
                ],
                stage="native_compile",
            )
            done = worker.command(
                [binary],
                cwd=worker.workspace,
                env=measurement.env(),
                affinity=True,
                check=False,
            )
            (worker.local / "stdout.log").write_text(done.stdout)
            (worker.local / "stderr.log").write_text(done.stderr)
            if done.returncode:
                raise ExecutionError("run", [binary], done)
            records = {}
            for line in done.stdout.splitlines():
                if line.startswith("WB_RESULT "):
                    record = json.loads(line[10:])
                    name = record["implementation"]
                    if name in records:
                        raise ValueError("duplicate native result " + name)
                    records[name] = record
            for row in rows:
                if row.status != "pending":
                    continue
                record = records[row.implementation]
                row.details["native"] = record
                if record.get("status") == "check_failed":
                    row.status = "check_failed"
                    row.error = record["error"]
                    if save_outputs and record.get("outputs_available"):
                        path = root / case.name / row.implementation / "Y.bin"
                        worker.download(
                            worker.workspace + "/" + row.implementation + "-Y.bin", path
                        )
                        row.details["outputs"] = {"Y": str(path.resolve())}
                    continue
                samples = record["samples_ms"]
                if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
                    raise ValueError("invalid native timing")
                row.samples_ms = samples
                row.median_ms = statistics.median(samples)
                row.correctness = record["correctness"]
                row.status = "ok"
                row.details.setdefault("artifacts", {})["executable"] = binary
                if save_outputs:
                    path = root / case.name / row.implementation / "Y.bin"
                    worker.download(
                        worker.workspace + "/" + row.implementation + "-Y.bin", path
                    )
                    row.details["outputs"] = {"Y": str(path.resolve())}
        except Exception as error:
            for row in rows:
                if row.status == "pending":
                    _failed(row, error, "run")
    (worker.local / "commands.json").write_text(json.dumps(worker.commands, indent=2))
    return rows
