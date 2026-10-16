"""Bind native ABI from typed artifact metadata; never parse generated C text."""

import math

DENSE_KINDS = {
    "rms_norm": 0,
    "layer_norm": 1,
    "softmax": 2,
    "rope_split": 3,
    "gemm_f32": 4,
    "rope_full": 5,
}


def dense_defines(case):
    return [f"-DWB_KIND={DENSE_KINDS[case.name]}"] + [
        f"-DWB_{k}={case.shape.get(k,1)}" for k in ("M", "N", "K", "T", "H", "P")
    ]


def bind_metadata(metadata, pointers, dimensions, scalars=None, aliases=None):
    """Pure metadata to direct native call binding, reusable by custom workloads."""
    scalars, aliases = scalars or {}, aliases or {}
    args, strides = [], []
    dims = {**metadata.get("bindings", {}), **dimensions}
    for parameter in metadata["arguments"]:
        name = aliases.get(parameter["name"], parameter["name"])
        if parameter["kind"] == "view":
            if name not in pointers:
                raise ValueError(
                    f"missing workload buffer {name!r}; configure implementation arguments mapping"
                )
            args.append(pointers[name])
            if parameter["strided"]:
                shape = [
                    int(x) if str(x).isdecimal() else dims[x]
                    for x in parameter["shape"]
                ]
                strides.extend(
                    str(math.prod(shape[i + 1 :])) for i in range(len(shape))
                )
        elif parameter["kind"] == "scalar":
            args.append(str(scalars[name]))
        else:
            raise ValueError(f"unsupported ABI kind {parameter['kind']}")
    args.extend(str(dims[d]) for d in metadata["shape_parameters"])
    args.extend(strides)
    return metadata["symbol"] + "(" + ", ".join(args) + ");"
