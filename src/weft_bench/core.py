"""Transport-free descriptions and native worker composition.

An implementation contributes preparation and a native call. Workloads own their
inputs, outputs and numerical oracle. Neither contributes a benchmark program.
"""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class NativeWorkload:
    header: str
    setup: str
    buffers: dict[str, str]
    output_type: str = "float"
    output_size: str = "data.output_size"
    check: str = "data.check(output)"
    includes: tuple[str, ...] = ()
    defines: tuple[str, ...] = ()
    before_call: str = ""
    after_call: str = ""
    correctness: str = "within-tolerance"


@dataclass(frozen=True)
class NativeCall:
    name: str
    prepare: str
    invoke: str
    collect: str = ""
    reset: str = ""
    includes: tuple[str, ...] = ()


def worker_source(
    workload: NativeWorkload,
    calls: list[NativeCall],
    input_source: str,
    *,
    save_outputs=False,
):
    includes = {workload.header, "worker.h", *workload.includes}
    for call in calls:
        includes.update(call.includes)
    lines = [
        *("#include " + json.dumps(h) for h in sorted(includes)),
        "int main() { try {",
        input_source,
        workload.setup,
    ]
    for call in calls:
        label = json.dumps(call.name)
        lines += [
            "{",
            f"std::vector<{workload.output_type}> output({workload.output_size});",
            call.prepare,
            "auto invoke = [&] { "
            + workload.before_call
            + call.invoke
            + workload.after_call
            + " };",
            "auto collect = [&] { " + call.collect + " };",
            "auto reset = [&] { " + call.reset + " };",
            "auto check = [&] { " + workload.check + "; };",
            f"wb::execute({label}, invoke, collect, reset, check, output, "
            + ("true" if call.reset else "false")
            + ", "
            + ("true" if save_outputs else "false")
            + ", "
            + json.dumps(workload.correctness)
            + ");",
            "}",
        ]
    lines += [
        'return 0; } catch(const std::exception &e) { std::fprintf(stderr,"prepare: %s\\n",e.what()); return 2; } }'
    ]
    return "\n".join(lines) + "\n"
