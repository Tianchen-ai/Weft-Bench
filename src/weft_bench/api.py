"""Public GGML/Weft comparisons; deployment is delegated to the outer runner."""

from dataclasses import replace
from pathlib import Path

from .inputs import Inputs
from .libraries import GGML, WeftKernel
from .types import Measurement


def benchmark(
    case,
    implementations,
    target,
    measurement=None,
    *,
    output=None,
    inputs=None,
    save_outputs=False,
):
    """Compatibility entry for check/history clients using implementation names."""
    from .runner import mode, run_operator

    if any(mode(name, target) is None for name in implementations):
        from .compat import benchmark as command_benchmark

        rows = []
        for name in implementations:
            if mode(name, target) is None:
                rows.extend(
                    command_benchmark(case, [name], target, measurement, output=output)
                )
            else:
                rows.extend(
                    run_operator(
                        case,
                        [name],
                        target,
                        measurement,
                        output=output,
                        inputs=inputs,
                        save_outputs=save_outputs,
                    )
                )
        return rows
    if inputs is None and case.data.get("inputs"):
        saved = case.data["inputs"]
        inputs = Inputs(
            preset=saved["preset"],
            shape=saved["shape"],
            seed=saved["seed"],
            parameters=saved["parameters"],
            buffers={k: v["path"] for k, v in saved["buffers"].items()},
        )
    return run_operator(
        case,
        implementations,
        target,
        measurement,
        output=output,
        inputs=inputs,
        save_outputs=save_outputs,
    )


def compare(
    operation,
    *,
    target,
    candidate=None,
    baseline=None,
    inputs=None,
    measurement=None,
    output=None,
    save_outputs=False,
):
    """Compare one Weft source/artifact with GGML using one shared input instance.

    candidate is a WeftKernel or file.py:symbol/artifact path. baseline selects a
    GGML build; omit it to use the configured library. Target supplies external
    compilation/deployment parameters, not workload semantics.
    """
    from .operators import get
    from .results import apply_baselines

    if isinstance(candidate, str):
        candidate = WeftKernel.parse(candidate)
    if candidate is not None and not isinstance(candidate, WeftKernel):
        raise TypeError("candidate must be a WeftKernel")
    if isinstance(inputs, str):
        inputs = (
            Inputs.load(inputs) if Path(inputs).is_file() else Inputs(preset=inputs)
        )
    selected = inputs or Inputs()
    case = selected.resolve(get(operation) if isinstance(operation, str) else operation)
    settings = dict(target.settings)
    if baseline is not None:
        if not isinstance(baseline, GGML):
            raise TypeError("baseline must be a GGML library")
        settings["ggml"] = baseline.configuration()
    implementations = dict(target.implementations)
    implementations["weft"] = {
        **implementations.get("weft", {}),
        **(candidate.configuration() if candidate else {}),
    }
    target = replace(target, settings=settings, implementations=implementations)
    rows = benchmark(
        case,
        ["ggml", "weft"],
        target,
        measurement,
        output=output,
        inputs=selected,
        save_outputs=save_outputs,
    )
    return apply_baselines(
        rows,
        [
            replace(r, details=dict(r.details))
            for r in rows
            if r.implementation == "ggml"
        ],
        "ggml",
    )
