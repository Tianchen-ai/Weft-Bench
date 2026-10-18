from __future__ import annotations

import csv
import json
from pathlib import Path

from .types import Result


def contract(row):
    """Readable comparison key; compiler build may change between candidates."""
    d = row.details
    measurement = d.get("measurement", {})
    timing = {
        k: measurement[k]
        for k in ("cache", "eviction_bytes", "profile", "protocol")
        if k in measurement
    }
    if timing.get("cache") == "reuse":
        timing.pop("eviction_bytes", None)
    return (
        row.case,
        row.target,
        json.dumps(row.shape, sort_keys=True),
        row.scope,
        json.dumps(timing, sort_keys=True),
        d.get("input_policy"),
        d.get("dtype"),
        d.get("phase"),
        json.dumps(d.get("execution", {}), sort_keys=True),
    )


def apply_baselines(rows, baselines, name=None):
    for row in rows:
        row.speedup = None
        row.baseline = None
        row.details.pop("comparison_note", None)
        row.details.pop("comparison_differences", None)
        matches = [
            b
            for b in baselines
            if contract(b) == contract(row)
            and b.status == "ok"
            and b.median_ms is not None
            and (name is None or b.implementation == name)
        ]
        if len(matches) == 1 and row.status == "ok" and row.median_ms:
            row.speedup = matches[0].median_ms / row.median_ms
            row.baseline = matches[0].implementation
        elif len(matches) > 1:
            row.details["comparison_note"] = (
                "multiple matching baselines; select one implementation"
            )
        else:
            row.details["comparison_note"] = (
                "no baseline with matching execution/timing scope"
            )
            fields = ("case", "target", "shape", "scope", "measurement", "input_policy", "dtype", "phase", "execution")
            row.details["comparison_differences"] = [
                {"implementation": b.implementation, "differences": {
                    key: {"current": current, "reference": reference}
                    for key, current, reference in zip(fields, contract(row), contract(b))
                    if current != reference
                }}
                for b in baselines if b.case == row.case and b.target == row.target
                and (name is None or b.implementation == name)
            ]
    return rows


def save(rows, destination):
    path = Path(destination)
    if path.suffix == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
        path = path / "results.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "results": [r.to_dict() for r in rows]}, indent=2
        )
        + "\n"
    )
    fields = [
        "case",
        "shape",
        "implementation",
        "target",
        "scope",
        "median_ms",
        "speedup",
        "status",
        "correctness",
        "tokens_per_second",
        "error",
    ]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            data = row.to_dict()
            data["shape"] = ";".join(f"{k}={v}" for k, v in row.shape.items())
            data["tokens_per_second"] = row.details.get("tokens_per_second", "")
            writer.writerow({key: data.get(key) for key in fields})
    return path


def read(path):
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    rows = [Result(**row) for row in json.loads(path.read_text())["results"]]
    # A copied run can replay its own captured canonical IR on a different host.
    for row in rows:
        local = path.parent / "artifacts" / row.case / row.implementation
        if (local / "replay.json").exists():
            row.details["replay"] = str((local / "replay.json").resolve())
            row.details["diagnostics"] = str(local.resolve())
            for key, relative in (
                ("canonical", "canonical.mlir"),
                ("weft", "artifact"),
            ):
                if (local / relative).exists():
                    row.details.setdefault("artifacts", {})[key] = str(
                        (local / relative).resolve()
                    )
    return rows


def table(rows):
    header = ["case", "shape", "implementation", "target", "ms", "speedup", "status"]
    models = any("tokens_per_second" in r.details for r in rows)
    if models:
        header += ["tokens/s", "replacement"]
    cells = [header]
    for r in rows:
        line = [
            r.case,
            ",".join(f"{k}={v}" for k, v in r.shape.items()),
            r.implementation,
            r.target,
            f"{r.median_ms:.6f}" if r.median_ms is not None else "—",
            f"{r.speedup:.3f}x" if r.speedup is not None else "—",
            r.status,
        ]
        if models:
            tokens = r.details.get("tokens_per_second")
            line += [
                f"{tokens:.3f}" if tokens else "—",
                r.details.get("replacement", "—"),
            ]
        cells.append(line)
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    return "\n".join(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        for row in cells
    )
