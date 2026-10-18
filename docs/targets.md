# Environment setup

The benchmark core has no SSH or machine-name branches. CLI/Python convenience
calls use a selected external target for building and deploying a native worker.
Run `op` and `e2e` locally or select a configured SSH target without changing
operation, input or replacement definitions.

`bench.local.json` contains `defaults` and `targets`; see `configs/example.json`.

| Field | Meaning |
|---|---|
| `weft_root`, `compiler`, `weft_python` | Host-side legacy Weft source/compiler/Python |
| `implementations` | Selected adapters, source/artifact bindings and compiler options |
| `triton_runtime` | Execution-side exported Triton–Weft native SDK include/library paths |
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

Single-instance operator workloads use one CPU thread; explicit grid/JIT calls and
models record their own thread counts. The legacy adapter builds generated C with
the configured native toolchain. Triton–Weft AOT consumes the exported native
library directly. Native timing needs the SDK; Python JIT and framework verification
also need a target-side Python installation.

For precomputed RoPE, provide the normal GGML build's `compile_commands.json`
(`CMAKE_EXPORT_COMPILE_COMMANDS=ON`). Its original `ops.cpp` flags compile a thin
bridge to `rotate_pairs<float>`. RMSNorm calls the existing fused multiply path;
LayerNorm times norm+mul+add; Softmax uses its native execution function; projection
uses a prepared normal GGML graph. Numerical oracles are separate from timed refs.

## Triton–Weft candidates

Select `weft_bench.adapters.triton_native:TritonAdapter` as the implementation
`factory`. The usual implementation name `weft` may select either this adapter or
the legacy one; it is a configured label, not automatic compiler detection.

```json
{
  "implementations": {
    "weft": {
      "factory": "weft_bench.adapters.triton_native:TritonAdapter",
      "bindings": {"q4_k_dot": {"artifact": "/work/artifacts/q4_k_dot"}}
    }
  },
  "triton_runtime": {
    "include": "/opt/triton-weft/native/include",
    "library": "/opt/triton-weft/native/lib/libweft_runtime.so"
  }
}
```

This fragment belongs inside a target with its `target`, worker `toolchain`, GGML
and deployment settings. Artifacts are host-side; SDK paths are execution-side.
For source compilation, replace `artifact` with `source` and `symbol`, and provide
the selected host `python`, optional `pythonpath`, `signature`, `constexprs`,
`backend_options` and compiler `toolchain`. An empty `signature` uses typed source
annotations. Bench calls Triton compilation/export and loads the native artifact;
it does not compile the exported candidate C again.

`entry: "single"` uses the single-instance ABI. `entry: "grid"` additionally
requires `grid` and `num_cpu_threads`, and is recorded as a different timing scope.
`weft_bench.adapters.triton_jit:TritonJitAdapter` selects target-side Python JIT;
its call overhead is reported separately from native operator timing.

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

For legacy replacement package linking, `deployment.python` must provide Weft's public
runtime, or `deployment.weft_root` can select an existing execution-side checkout.
The tool calls `Package.create` on the host and `Package.link` on the target;
it does not invent a second package format or linker.

For Triton–Weft, set `llama.package_backend` to `triton` and configure
`llama.triton.python` plus the source role bindings if compiling replacements.
The existing role mappings call `triton.Package.create` with native artifacts;
this route does not use the legacy target-side linker. Both providers require a
compatible integrated llama.cpp build and report measured-window replacement hits.

Preset replacement groups:
- `ggml.vec_dot`: q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K, q5_K, q6_K.
- `ggml.from_float`: q8_0, q8_K.
- `ggml.ime1`: q4_centered_prefill, q4_affine_prefill.

IME requires an eligible target and explicit compiler extension options. It remains
a distinct packed call boundary. Replacing an ordinary dot does not automatically
replace an IME model path.

Saved profiles and optional `call_sites` mappings remain supported. They associate
model functions with workloads; they do not choose replacements automatically.
