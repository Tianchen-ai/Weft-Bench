"""Weft candidates and the fixed GGML library, independent of input selection."""

from dataclasses import dataclass, field
from pathlib import Path
from .core import NativeCall
from .adapters.dense import bind_metadata


@dataclass(frozen=True)
class WeftKernel:
    source: str = ""
    symbol: str = ""
    artifact: str = ""
    compiler: str = ""
    options: dict = field(default_factory=dict)
    arguments: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, value):
        if ":" in value:
            source, symbol = value.rsplit(":", 1)
            return cls(source=str(Path(source).expanduser().resolve()), symbol=symbol)
        return cls(artifact=str(Path(value).expanduser().resolve()))

    def configuration(self):
        result = {}
        if self.source:
            result.update(source=self.source, symbol=self.symbol)
        if self.artifact:
            result.update(
                path=self.artifact, factory="weft_bench.adapters.native:ArtifactAdapter"
            )
        if self.compiler:
            result["compiler"] = self.compiler
        if self.options:
            result["compile_options"] = self.options
        if self.arguments:
            result["arguments"] = self.arguments
        return result


@dataclass(frozen=True)
class GGML:
    source: str
    libraries: str
    compile_commands: str = ""

    def configuration(self):
        return {k: v for k, v in self.__dict__.items() if v}


def weft_call(name, case, workload, metadata, aliases=None):
    dimensions = {**case.shape}
    if case.family == "ime":
        dimensions["M"] = case.data.get("tile_rows", 4)
    validate_binding(case, metadata, aliases or {}, dimensions)
    call = bind_metadata(
        metadata, workload.buffers, dimensions, {"epsilon": "data.epsilon"}, aliases
    )
    if case.name == "rope_full":
        call = "data.prepare_rope(); " + call
    setup = (
        "std::vector<uint8_t> workspace(data.M*data.activation_bytes);"
        if case.family == "projection"
        else ""
    )
    return NativeCall(name, setup, call, includes=(name + "-kernel.h",))


def validate_binding(case, metadata, aliases, dimensions):
    from .workloads import ports

    expected = {
        p.name: (p.shape, "dense.f32" if p.dtype == "f32" else p.dtype.upper())
        for p in ports(case)
    }
    k = case.shape.get("K")
    if case.family in ("vec_dot", "quantize", "dequantize"):
        expected = {
            name: ((1,) if name == "Y" and case.family == "vec_dot" else (k,), encoding)
            for name, (_, encoding) in expected.items()
        }
    if case.family == "projection":
        from .operators import FORMATS, PARTNERS

        fmt = case.data["format"]
        activation = (
            "dense.f16" if fmt == "f16" else PARTNERS[FORMATS.index(fmt)].upper()
        )
        expected["Xh" if fmt == "f16" else "Xq"] = ((dimensions["M"], k), activation)
        if fmt == "f16":
            expected["W"] = (expected["W"][0], "dense.f16")
    if case.family == "ime":
        rows = dimensions["M"]
        expected = {
            "W": (
                (dimensions["N"], k),
                "IMEQ4Affine_I[16]" if case.data["affine"] else "IMEQ4Centered_I[16]",
            ),
            "Xq": ((rows, k), f"IMEQ8_I[{rows}]"),
            "Y": ((rows, dimensions["N"]), "dense.f32"),
        }
    dims = {**metadata.get("bindings", {}), **dimensions}
    for arg in metadata["arguments"]:
        role = aliases.get(arg["name"], arg["name"])
        if arg["kind"] != "view" or role not in expected:
            continue
        shape = tuple(int(x) if str(x).isdecimal() else dims[x] for x in arg["shape"])
        wanted, encoding = expected[role]
        if shape != wanted or arg["encoding"] != encoding:
            raise ValueError(
                f"kernel buffer {arg['name']} is {arg['encoding']}{shape}; workload {role} requires {encoding}{wanted}"
            )
        access = (
            "readwrite"
            if role in ("Xq", "Xh") and case.family == "projection"
            else "write" if role == "Y" else "read"
        )
        if arg.get("access") != access:
            raise ValueError(
                f"kernel access for {role} does not match the workload input/output contract"
            )


def ggml_call(name, case):
    family = case.family
    if family == "dense":
        return NativeCall(
            name,
            "DenseReference reference(data.input,data.weight,data.bias,data.cosine,data.sine,output,data.epsilon);",
            "reference.run(data.input,output);",
            includes=("dense_reference.h",),
        )
    if family == "vec_dot":
        return NativeCall(
            name,
            'if(!data.traits||!data.traits->vec_dot||data.traits->nrows!=1)throw std::runtime_error("GGML single-row dot unavailable");',
            "data.traits->vec_dot(data.K,output.data()+row,0,data.weights.data()+row*data.weight_bytes,0,data.activation.data(),0,1);",
        )
    if family == "quantize":
        fmt = case.dtype.split("->")[1]
        symbol = {
            "q8_k": "quantize_row_q8_K",
            "q8_0": "quantize_row_q8_0",
            "q8_1": "quantize_row_q8_1",
        }[fmt]
        return NativeCall(
            name,
            "",
            symbol
            + "(data.source.data()+row*data.K,output.data()+row*data.weight_bytes,data.K);",
        )
    if family == "dequantize":
        return NativeCall(
            name,
            "",
            "ggml_get_type_traits(data.weight_type)->to_float(data.weights.data()+row*data.weight_bytes,output.data()+row*data.K,data.K);",
        )
    if family == "projection":
        return NativeCall(
            name,
            "ProjectionReference reference(data.weight_type,data.M,data.N,data.K,data.weights.data(),data.activation.data(),output);",
            "reference.run();",
            "reference.collect();",
            includes=("projection_reference.h",),
        )
    if family == "ime":
        return NativeCall(
            name,
            "",
            'if(spacemit_kernels::ime1::gemm_kernel_i8i4(32,data.activation.data()+m*(data.K/32)*36,data.weights.data(),WB_AFFINE?data.weights.data():nullptr,output.data()+m*data.N,WB_TILE_ROWS,data.N,data.K/32,data.N)!=WB_TILE_ROWS)throw std::runtime_error("IME row tile unhandled");',
        )
    raise ValueError(f"GGML does not implement {case.name}")
