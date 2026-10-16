"""Thin bridges to the selected GGML build; reference math stays upstream."""

import json
from pathlib import Path
import shlex

from .dense import dense_defines


def prepare_dense_reference(case, target, executor):
    from .native import NATIVE

    executor.upload(NATIVE / "dense_reference.h", "dense_reference.h")
    objects = []
    if case.name == "rope_split":
        source = str(
            Path(target.settings["ggml"]["source"]) / "ggml/src/ggml-cpu/ops.cpp"
        )
        database = target.settings["ggml"].get(
            "compile_commands",
            str(
                Path(target.settings["ggml"]["libraries"]).parent
                / "compile_commands.json"
            ),
        )
        commands = json.loads(
            executor.command(["cat", database], stage="prepare").stdout
        )
        entries = [x for x in commands if x["file"].endswith("/ggml-cpu/ops.cpp")]
        if len(entries) != 1:
            raise ValueError("reference needs one GGML ops.cpp native compile command")
        entry = entries[0]
        original = entry.get("arguments") or shlex.split(entry["command"])
        flags = []
        skip = False
        for value in original[1:]:
            if skip:
                skip = False
                continue
            if value in ("-o", "-MF", "-MT", "-MQ"):
                skip = True
            elif value not in ("-c", "-MD", "-MMD", entry["file"]):
                flags.append(value)
        code = "#include " + json.dumps(source) + "\n" + """
extern "C" void wb_ggml_rotate(size_t tokens, size_t heads, size_t pairs,
    const float *cache, const float *x, float *y) {
  for(size_t t=0;t<tokens;++t) for(size_t h=0;h<heads;++h)
    rotate_pairs<float>(pairs*2, pairs, cache+t*pairs*2,
                       x+(t*heads+h)*pairs*2, y+(t*heads+h)*pairs*2);
}
"""
        bridge = executor.write("rope_reference.cpp", code)
        obj = executor.workspace + "/rope_reference.o"
        executor.command(
            [
                original[0],
                *flags,
                "-ffunction-sections",
                "-fdata-sections",
                "-c",
                bridge,
                "-o",
                obj,
            ],
            cwd=entry["directory"],
            stage="native_compile",
        )
        executor.write("reference-build.json", json.dumps(entry, indent=2))
        objects += [obj, "-Wl,--gc-sections"]
    return objects, dense_defines(case)
