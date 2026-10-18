from dataclasses import replace

from ..loading import load_object
from ..types import Case

FORMATS = (
    "q1_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_s",
    "iq2_xs",
    "iq2_xxs",
    "iq3_s",
    "iq3_xxs",
    "iq4_nl",
    "iq4_xs",
    "tq1_0",
    "tq2_0",
    "mxfp4",
    "nvfp4",
)
PARTNERS = (
    "q8_0",
    "q8_0",
    "q8_1",
    "q8_0",
    "q8_1",
    "q8_0",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_0",
    "q8_k",
    "q8_k",
    "q8_k",
    "q8_0",
    "q8_0",
)
DECODE_DEFINES = dict(
    zip(
        (
            "q1_0",
            "q4_0",
            "q4_1",
            "q5_0",
            "q5_1",
            "q8_0",
            "q2_k",
            "q3_k",
            "q4_k",
            "q5_k",
            "q6_k",
            "iq1_s",
            "iq1_m",
            "iq2_xxs",
            "iq4_nl",
            "iq4_xs",
            "tq1_0",
            "tq2_0",
            "mxfp4",
            "nvfp4",
        ),
        (
            "Q10",
            "Q40",
            "Q41",
            "Q50",
            "Q51",
            "Q80",
            "Q2K",
            "Q3K",
            "Q4K",
            "Q5K",
            "Q6K",
            "IQ1S",
            "IQ1M",
            "IQ2_XXS",
            "IQ4_NL",
            "IQ4_XS",
            "TQ10",
            "TQ20",
            "MXFP4",
            "NVFP4",
        ),
    )
)


def catalog():
    cases = {}
    for index, (fmt, partner) in enumerate(zip(FORMATS, PARTNERS)):
        name = fmt + "_dequantize"
        cases[name] = Case(
            name,
            "dequantize",
            {"rows": 1024, "K": 4096},
            "row-dequantization",
            input_policy="finite-random-record-replicated",
            dtype=f"{fmt}->f32",
        )
        for suffix, rows in (("dot", 1), ("dot_rows", 4096)):
            name = f"{fmt}_{suffix}"
            cases[name] = Case(
                name,
                "vec_dot",
                {"rows": rows, "K": 4096},
                "vec-dot-leaf" if rows == 1 else "independent-dots-shared-activation",
                input_policy="independent-quantized-rows-seed410",
                dtype=f"{fmt}*{partner}",
            )
    for kind, fmt in enumerate(("q8_0", "q8_1", "q8_k")):
        name = fmt + "_quantize"
        cases[name] = Case(
            name,
            "quantize",
            {"rows": 1, "K": 4096},
            "activation-quantization",
            input_policy="fixed-float-mod31",
            dtype=f"f32->{fmt}",
        )
    cases["gemm_f32"] = Case(
        "gemm_f32",
        "dense",
        {"M": 128, "N": 4096, "K": 4096},
        "preallocated-gemm",
    )
    cases["affine"] = Case("affine", "dense", {"N": 4096}, "elementwise-scale-bias")
    cases["causal_attention"] = Case(
        "causal_attention", "attention", {"B": 1, "T": 4, "S": 16, "H": 4, "G": 2, "D": 32},
        "causal-grouped-query-attention", input_policy="finite-random-seed410",
    )
    for name in ("rms_norm", "layer_norm"):
        cases[name] = Case(name, "dense", {"M": 128, "N": 4096}, name + "-affine")
    cases["rope_split"] = Case(
        "rope_split",
        "dense",
        {"T": 128, "H": 32, "P": 64},
        "rope-precomputed-cos-sin",
    )
    cases["rope_full"] = replace(
        cases["rope_split"], name="rope_full", scope="rope-neox-includes-coefficients"
    )
    cases["softmax"] = Case("softmax", "dense", {"M": 4096, "N": 128}, "row-softmax")
    for name, affine in (("ime_centered", False), ("ime_affine", True)):
        cases[name] = Case(
            name,
            "ime",
            {"M": 4, "N": 16, "K": 4096},
            "packed-ime-leaf",
            input_policy="distinct-columns-mod61-activation-mod17",
            dtype="packed-i4*i8",
            data={"affine": affine},
        )
        cases[name + "_decode"] = replace(
            cases[name], name=name + "_decode", shape={"M": 1, "N": 16, "K": 4096},
            phase="decode", data={"affine": affine, "tile_rows": 1},
        )
    for index, fmt in enumerate(("f16", *FORMATS)):
        for phase, m in (("decode", 1), ("prefill", 128)):
            name = fmt + "_projection_" + phase
            cases[name] = Case(
                name,
                "projection",
                {"M": m, "N": 4096, "K": 4096},
                "projection-includes-activation-quantization",
                phase=phase,
                dtype=fmt + "*f32",
                input_policy="valid-quantized-record-replicated",
                data={"format": fmt},
            )
    from .shared_semantics import catalog as shared_catalog
    from .pointer_semantics import catalog as pointer_catalog

    cases.update(shared_catalog())
    cases.update(pointer_catalog())
    return cases


SUITES = {
    "pointer-semantics": tuple("pointer_" + name for name in ("add", "tail", "strided", "sum", "softmax", "dot", "control", "alias", "mask_guard", "carry", "select", "bool")),
    "shared-semantics": tuple("shared_" + name for name in ("arange", "sum", "max", "softmax", "dot_acc", "batch_dot_acc", "for_while", "nested_level")),
    "quick": (
        "q4_0_dot",
        "q4_k_dot",
        "q8_k_quantize",
        "gemm_f32",
        "rms_norm",
        "layer_norm",
        "rope_split",
        "softmax",
    ),
    "llama-hot": (
        "q4_0_dot_rows",
        "q4_k_dot_rows",
        "q6_k_dot_rows",
        "q4_k_projection_decode",
        "q4_k_projection_prefill",
        "ime_centered",
        "ime_affine",
    ),
}


def get(name, shape=None):
    if ":" in name:
        case = load_object(name)
        if callable(case):
            case = case()
    else:
        try:
            case = catalog()[name]
        except KeyError:
            raise ValueError(f"unknown case {name!r}; use weft-bench list") from None
    if not isinstance(case, Case):
        raise TypeError("custom case must be a Case or a factory returning one")
    if shape:
        unknown = shape.keys() - case.shape.keys()
        if unknown or any(type(v) is not int or v <= 0 for v in shape.values()):
            raise ValueError(f"invalid shape override: {shape}")
        case = replace(case, shape={**case.shape, **shape})
    return case
