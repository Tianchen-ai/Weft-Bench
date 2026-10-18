"""Visible file/version identity for resumable verification, without content hashes."""
from pathlib import Path
import json

from ..execution import Executor


def observe_runtime(target, output):
    configuration = target.settings.get("verification", {}).get("runtime", {})
    if not configuration.get("python"):
        return {"status": "not_configured"}
    executor = Executor(target, Path(output) / "environment")
    script = """import json, pathlib, sys, triton
root = pathlib.Path(triton.__file__).parent
files = {}
for path in root.rglob('*'):
    if path.is_file() and path.suffix in ('.py', '.so') and '__pycache__' not in path.parts:
        stat = path.stat()
        files[str(path.resolve())] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
print(json.dumps({'status': 'observed', 'python': sys.executable, 'triton_version': triton.__version__,
                  'triton_package': triton.__file__, 'files': files}))
"""
    completed = executor.command([configuration["python"], "-c", script],
                                 env={**configuration.get("environment", {}), "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    try:
        identity = json.loads(completed.stdout) if not completed.returncode else {
            "status": "unavailable", "error": completed.stderr, "exit_code": completed.returncode}
    except ValueError:
        identity = {"status": "unavailable", "error": completed.stdout + completed.stderr}
    (executor.local / "observed-runtime.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def snapshot(entry, target):
    paths = []
    if entry.get("kind") == "program":
        paths.extend((Path(__file__).with_name("program_process.py"),
                      Path(__file__).parent.parent / "operators/program_oracles.py",
                      Path(__file__).parent.parent / "operators/dense_oracles.py",
                      Path(__file__).parent.parent / "operators/encoded_oracles.py",
                      Path(__file__).parent.parent / "operators/example_oracles.py"))
        paths.extend(entry.get("imports", {}).values())
    if entry.get("kind") in ("framework", "runtime"):
        probes = Path(__file__).with_name("probes")
        paths.append(probes / (entry["kind"] + "_process.py"))
        paths.append(probes / ("framework_kernels" if entry["kind"] == "framework" else "runtime_kernels.py"))
        if entry["kind"] == "runtime":
            paths.append(probes / "riscv_kernels.py")
        paths.append(Path(__file__).parent.parent / "adapters/compile_triton.py")
    if entry.get("kind") == "usage":
        paths.append(Path(__file__).with_name("usage_process.py"))
        paths.extend(entry.get("imports", {}).values())
    options = entry.get("options", {})
    if entry.get("artifact"):
        paths.append(entry["artifact"])
    else:
        paths.extend(target.settings.get("verification", {}).get("provenance_paths", []))
        paths.extend(options.get("pythonpath", []))
        if options.get("python"):
            paths.append(options["python"])
    source = entry.get("source", options.get("source"))
    if source and not entry.get("artifact"):
        paths.append(source)
    records = {}
    for item in paths:
        path = Path(item).expanduser().resolve()
        files = path.rglob("*") if path.is_dir() else (path,)
        for file in files:
            if "__pycache__" in file.parts or ".git" in file.parts or file.is_dir():
                continue
            if file.suffix in (".pyc", ".pyo"):
                continue
            try:
                stat = file.stat()
                records[str(file)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            except FileNotFoundError:
                records[str(file)] = {"missing": True}
    settings = target.settings.get("verification", {})
    return {"entry": entry, "compiler_environment": settings.get("compiler_environment"),
            "observed_runtime": target.settings.get("_verification_runtime_identity"),
            "runtime": settings.get("runtime", {}), "files": records}
