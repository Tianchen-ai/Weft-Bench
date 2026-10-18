"""The shipped GGML workloads: semantics/IO, never compiler or machine selection."""

from dataclasses import dataclass
from .core import NativeWorkload
from .adapters.dense import dense_defines


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    access: str = "input"
    layout: str = "contiguous"


def ggml_type(fmt):
    return "GGML_TYPE_" + fmt.upper()


def parameter_defaults(case):
    if case.name == "pointer_select":
        return {"select_x": 1.0}
    if case.name in ("rms_norm", "layer_norm"):
        return {"epsilon": 1e-5}
    if case.name == "affine":
        return {"scale": 1.25, "bias": -0.5}
    if case.family == "attention":
        return {"scale": case.shape["D"] ** -0.5, "start": case.shape["S"] - case.shape["T"]}
    return {}


def ports(case):
    if case.family == "pointer_semantics":
        from .operators.pointer_semantics import ports as pointer_ports

        return pointer_ports(case)
    if case.family == "shared_semantics":
        from .operators.shared_semantics import ports as shared_ports

        return shared_ports(case)
    s = case.shape
    family = case.family
    if family == "attention":
        if s["H"] % s["G"] or s["T"] > s["S"]:
            raise ValueError("attention requires H divisible by G and T <= S")
        return [TensorSpec("Q", "f32", (s["B"], s["T"], s["H"], s["D"])),
                TensorSpec("K", "f32", (s["B"], s["S"], s["G"], s["D"])),
                TensorSpec("V", "f32", (s["B"], s["S"], s["G"], s["D"])),
                TensorSpec("Y", "f32", (s["B"], s["T"], s["H"], s["D"]), "output")]
    if family == "dense":
        if case.name == "affine":
            return [TensorSpec("X", "f32", (s["N"],)),
                    TensorSpec("Y", "f32", (s["N"],), "output")]
        if case.name.startswith("rope"):
            result = [TensorSpec("X", "f32", (s["T"], s["H"], 2, s["P"]))]
            if case.name == "rope_split":
                result += [
                    TensorSpec(k, "f32", (s["T"], s["P"])) for k in ("Cos", "Sin")
                ]
            return result + [TensorSpec("Y", "f32", result[0].shape, "output")]
        result = [TensorSpec("X", "f32", (s["M"], s.get("K", s["N"])))]
        if case.name == "gemm_f32":
            result.append(TensorSpec("W", "f32", (s["N"], s["K"])))
        if case.name in ("rms_norm", "layer_norm"):
            result.append(TensorSpec("Weight", "f32", (s["N"],)))
        if case.name == "layer_norm":
            result.append(TensorSpec("Bias", "f32", (s["N"],)))
        return result + [TensorSpec("Y", "f32", (s["M"], s["N"]), "output")]
    if family == "vec_dot":
        w, x = case.dtype.split("*")
        return [
            TensorSpec("W", w, (s["rows"], s["K"]), layout="ggml-blocks"),
            TensorSpec("X", x, (s["K"],), layout="ggml-blocks"),
            TensorSpec("Y", "f32", (s["rows"],), "output"),
        ]
    if family in ("quantize", "dequantize"):
        before, after = case.dtype.split("->")
        return [
            TensorSpec(
                "X" if family == "quantize" else "W",
                before,
                (s["rows"], s["K"]),
                layout="contiguous" if before == "f32" else "ggml-blocks",
            ),
            TensorSpec(
                "Y",
                after,
                (s["rows"], s["K"]),
                "output",
                "contiguous" if after == "f32" else "ggml-blocks",
            ),
        ]
    if family == "projection":
        return [
            TensorSpec(
                "W", case.data["format"], (s["N"], s["K"]), layout="ggml-blocks"
            ),
            TensorSpec("X", "f32", (s["M"], s["K"])),
            TensorSpec("Y", "f32", (s["M"], s["N"]), "output"),
        ]
    if family == "ime":
        return [
            TensorSpec("W_f32", "f32", (s["N"], s["K"]), layout="packed-before-timing"),
            TensorSpec("X_f32", "f32", (s["M"], s["K"]), layout="packed-before-timing"),
            TensorSpec("Y", "f32", (s["M"], s["N"]), "output"),
        ]
    raise ValueError(f"no GGML workload for {case.name}")


def native_workload(case):
    if case.family == "pointer_semantics":
        from .operators.pointer_semantics import workload

        return workload(case)
    if case.family == "shared_semantics":
        from .operators.shared_semantics import workload

        return workload(case)
    s = case.shape
    family = case.family
    if family == "attention":
        return NativeWorkload(
            "attention_inputs.h", "AttentionInputs data(inputs);",
            {"Q": "data.query.data()", "K": "data.keys.data()",
             "V": "data.values.data()", "Y": "output.data()"},
            defines=tuple(f"-DWB_{key}={value}" for key, value in s.items()),
            data_type="AttentionInputs", scalars={"start": "data.start", "scale": "data.scale"},
        )
    if family == "dense":
        return NativeWorkload(
            "dense_inputs.h",
            "DenseInputs data(inputs);",
            {
                "X": "data.input.data()",
                "W": "data.weight.data()",
                "Weight": "data.weight.data()",
                "Bias": "data.bias.data()",
                "Cos": "data.cosine.data()",
                "Sin": "data.sine.data()",
                "Y": "output.data()",
            },
            defines=tuple(dense_defines(case)),
            data_type="DenseInputs",
            scalars={"epsilon": "data.epsilon", "scale": "data.scale", "bias": "data.scalar_bias"},
            correctness=(
                "sampled-64-within-tolerance"
                if case.name == "gemm_f32"
                else "within-tolerance"
            ),
        )
    tables = {
        k: "data." + k + ".data()"
        for k in ("grid", "signs", "codebook", "f16_bits", "scale")
    }
    tables.update(
        powers="data.powers",
        e8m0_scale="data.scale.data()",
        ue4m3_scale="data.scale.data()",
    )
    if family in ("vec_dot", "dequantize", "quantize"):
        kind = {"vec_dot": 0, "quantize": 1, "dequantize": 2}[family]
        if family == "vec_dot":
            fmt, partner = case.dtype.split("*")
        else:
            fmt = case.dtype.split("->")[1 if family == "quantize" else 0]
            partner = "q8_0"
        buffers = {
            **tables,
            "W": "data.weights.data()+row*data.weight_bytes",
            "Y": "output.data()+row",
        }
        if family == "vec_dot":
            buffers["X"] = "data.activation.data()"
        elif family == "quantize":
            buffers.update(
                X="data.source.data()+row*data.K",
                Y="output.data()+row*data.weight_bytes",
            )
        else:
            buffers["Y"] = "output.data()+row*data.K"
        defines = (
            f"-DWB_ROWS={s['rows']}",
            f"-DWB_K={s['K']}",
            f"-DWB_QUANT_KIND={kind}",
            "-DWB_WEIGHT_TYPE=" + ggml_type(fmt),
            "-DWB_ACTIVATION_TYPE=" + ggml_type(partner),
        )
        return NativeWorkload(
            "quantized_inputs.h",
            "QuantizedInputs data(inputs);",
            buffers,
            output_type="uint8_t" if family == "quantize" else "float",
            defines=defines,
            before_call="for(size_t row=0;row<data.rows;++row) {",
            after_call="}",
            correctness=("bit-exact" if family == "quantize" else
                         "within-tolerance-packed-affine-dot" if family == "vec_dot" and fmt in ("q4_1", "q5_1") else
                         "within-tolerance"),
        )
    if family == "projection":
        from .operators import FORMATS, PARTNERS

        fmt = case.data["format"]
        partner = "f16" if fmt == "f16" else PARTNERS[FORMATS.index(fmt)]
        buffers = {k: v.replace("data.", "data.tables.") for k, v in tables.items()}
        buffers.update(
            W="data.weights.data()",
            X="data.activation.data()",
            Xq="workspace.data()",
            Xh="reinterpret_cast<_Float16*>(workspace.data())",
            Y="output.data()",
        )
        if fmt == "f16":
            buffers["W"] = "reinterpret_cast<const _Float16*>(data.weights.data())"
        defines = tuple(f"-DWB_{k}={v}" for k, v in s.items()) + (
            f"-DWB_ROWS={s['N']}",
            "-DWB_QUANT_KIND=0",
            "-DWB_WEIGHT_TYPE=" + ggml_type(fmt),
            "-DWB_ACTIVATION_TYPE=" + ggml_type(partner),
        )
        return NativeWorkload(
            "projection_inputs.h",
            "ProjectionInputs data(inputs);",
            buffers,
            defines=defines,
            snapshots={"X": "data.activation", "W": "data.weights", "reference-Xq": "data.quantized",
                       "reference-Y": "data.expected", "reference-Y-indices": "data.checked_indices",
                       "table-grid": "data.tables.grid", "table-signs": "data.tables.signs"},
            correctness="within-tolerance-full-decode" if s["M"]==1 else "within-tolerance-sampled-64-prefill-and-finite-outputs",
        )
    if family == "ime":
        return NativeWorkload(
            "ime_inputs.h",
            "IMEInputs data(inputs);",
            {
                "W": "data.weights.data()",
                "Xq": "data.activation.data()+m*(data.K/32)*36",
                "Y": "output.data()+m*data.N",
            },
            defines=tuple(f"-DWB_{k}={v}" for k, v in s.items())
            + (f"-DWB_AFFINE={int(case.data['affine'])}",
               f"-DWB_TILE_ROWS={case.data.get('tile_rows', 4)}"),
            before_call="for(size_t m=0;m<data.M;m+=WB_TILE_ROWS) {",
            after_call="}",
        )
    raise ValueError(f"no native workload for {case.name}")
