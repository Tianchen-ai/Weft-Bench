# Environment setup

The benchmark core has no SSH or machine-name branches. CLI/Python convenience
calls use a selected external target for building and deploying a native worker.
Run `op` and `e2e` locally or select a configured SSH target without changing
operation, input or replacement definitions.

`bench.local.json` contains `defaults` and `targets`; see `configs/example.json`.

| Field | Meaning |
|---|---|
| `weft_root`, `compiler`, `weft_python` | Host-side Weft source/compiler/Python |
| `target` | Weft target march, ABI and VLEN |
| `toolchain` | Execution-side C/C++ commands, flags and libraries |
| `deployment` | Optional SSH alias, workspace, CPU affinity and target Python |
| `ggml` | Target-side llama.cpp source, GGML libraries and optional compile database |
| `bindings` / `bindings_file` | Optional preset Weft source/symbol/physical options |
| `capabilities` | Explicit extension availability, such as spacemit-ime1 |
| `llama` | Model paths, normal/integrated builds, threads and optional profiles |
| `check` | Optional historical reference/checkpoint convenience |

Source paths for Weft kernels are host-side. GGML libraries, model weights,
llama.cpp binaries and toolchain commands are execution-side. Credentials stay in
the existing SSH environment. Configuration files and models are not bundled.

Operator workloads use one CPU thread. Model thread counts are separately selectable.
The generated C from a Weft artifact is compiled by the configured native toolchain;
remote operator measurement does not require a remote Weft Python installation.

For precomputed RoPE, provide the normal GGML build's `compile_commands.json`
(`CMAKE_EXPORT_COMPILE_COMMANDS=ON`). Its original `ops.cpp` flags compile a thin
bridge to `rotate_pairs<float>`. RMSNorm calls the existing fused multiply path;
LayerNorm times norm+mul+add; Softmax uses its native execution function; projection
uses a prepared normal GGML graph. Numerical oracles are separate from timed refs.

## llama.cpp setup

```json
{
  "llama": {
    "threads": 8,
    "models": {"llama2-7b-q4_0": "/models/llama-2-7b.Q4_0.gguf"},
    "implementations": {
      "normal": {"binary": "/llama-normal/build/bin/llama-bench"},
      "weft": {
        "binary": "/llama-weft/build/bin/llama-bench",
        "source": "/llama-weft",
        "build_directory": "/llama-weft/build",
        "weft_include": "/sdk/share/weft/llama.cpp/include",
        "env": {"LD_LIBRARY_PATH": "/sdk/lib"}
      }
    }
  }
}
```

The normal and integrated builds use matching normal backend settings. The candidate
tree is prepared with Weft's public llama.cpp integration once. Its existing hooks
accept parameter-selected packages; changing a kernel or replacement mapping does
not require patching the hooks again.

When source/build/include are provided, the tool installs measurement-window
observers into that candidate benchmark harness and invokes an incremental
`cmake --build ... --target llama-bench`. This affects benchmark instrumentation,
not GGML kernel arithmetic. Supply a separate candidate source tree.

For replacement package linking, `deployment.python` must provide Weft's public
runtime, or `deployment.weft_root` can select an existing execution-side checkout.
The tool calls `Package.create` on the host and `Package.link` on the target;
it does not invent a second package format or linker.

Preset replacement groups:
- `ggml.vec_dot`: q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K, q5_K, q6_K.
- `ggml.from_float`: q8_0, q8_K.
- `ggml.ime1`: q4_centered_prefill, q4_affine_prefill.

IME requires an eligible target and explicit compiler extension options. It remains
a distinct packed call boundary. Replacing an ordinary dot does not automatically
replace an IME model path.

Saved profiles and optional `call_sites` mappings remain supported. They associate
model functions with workloads; they do not choose replacements automatically.
