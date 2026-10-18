# Triton–Weft native candidates

Use the task's explicitly configured Bench environment. In the local plugin settings,
an environment has the same `command` argument array and `config` path as the default.
This selects an installed Bench build/configuration; compilation still belongs to Triton.
The plugin name remains Weft Bench. An explicit environment selects the actual
Triton–Weft installation; neither the plugin name nor a `weft` implementation label
establishes which compiler is configured. Inspect `--plugin-info` and the selected
implementation before changing compiler environments.

```sh
python3 "$BENCH_SKILL/scripts/bench.py" --plugin-info
python3 "$BENCH_SKILL/scripts/bench.py" --environment triton-weft --plugin-info
python3 "$BENCH_SKILL/scripts/bench.py" --environment triton-weft op affine \
  --target k1 --impl triton_source --baseline /path/to/affine-ref --output /path/to/runs/affine
```

Environment and implementation names come from local configuration, not from the
runner. The examples use `triton-weft`, `triton_source` and `triton_aot`; inspect
the selected config to find the actual names and source/artifact bindings.

The implementation factory is `weft_bench.adapters.triton_native:TritonAdapter`.
Source configuration supplies the selected Triton `python`, optional `pythonpath`
list, `source`, `symbol`, ASTSource `signature`, `constexprs`, `backend_options` and
native `toolchain`. The public compile route is `triton.compile(ASTSource(...),
target=CPUTarget(...), options=...)`, followed by `CompiledKernel.export`.
An explicit empty `signature: {}` lets the Weft frontend use the kernel's typed
annotations. `--weft SOURCE:SYMBOL` overrides the chosen workload binding, and
`--weft ARTIFACT` preserves the configured Triton native provider. These overrides
also work through the Python comparison API; an artifact does not fall back to
the legacy artifact compiler.
AOT configuration supplies `artifact` instead of `source`. Both consume the actual
exported shared library; the worker never recompiles candidate C.

A configured `program` names a product catalog Program. Its source is prepared
through `Program.signature`, `Config` and `Program.compile`; ordinary kernels keep
the ASTSource route. For configured numerical and installed-runtime validation,
use [verify](verification.md). This is separate from native timing below.

The external target config supplies `triton_runtime.include` and
`triton_runtime.library` on the execution machine. They must expose the real native
SDK's `weft_kernel_open_native`, `weft_kernel_get_entry` and `weft_kernel_check_worker`.
Missing runtime support is a setup/unsupported result, not permission to bypass checks.

`entry: "single"` is the default. It uses the single-instance ABI in manifest.json,
checks the bounded measurement thread, and directly calls the native pointer without
creating an OpenMP team. Existing compatible workloads and references remain usable.

`entry: "grid"` also requires explicit `grid: [x, y, z]` and `num_cpu_threads`.
Use the exported `manifest.grid` and launcher declaration, not the `*_instance`
pointer. The generated native launcher calls the SDK guard on every worker before
writes. The result is a separate `grid-launch` scope with grid/thread/CPU parameters;
do not divide that time into a dot-leaf speedup or compare it automatically with
a single-instance baseline. The author kernel must partition its work by program id.
Current grid support requires whole-workload buffers; row/tile-loop workloads report
unsupported instead of wrapping a second dispatch loop around a grid.

`replay` uses the saved native artifact. For a source-origin run, `--source FILE`
recompiles through its recorded Triton environment/signature/options. The old
`--compiler /path/to/weft-compile` override does not apply to Triton candidates.
When fixed reference protocols or inputs differ, use `explain` to read the specific
comparison differences. Preserve the existing reference rather than changing its labels.

## Target-native Python JIT

The explicit factory `weft_bench.adapters.triton_jit:TritonJitAdapter` runs
`kernel[grid]` on the RISC-V machine with CPU Tensors backed by the existing workload
fixture. Configure the target-side `python`, `environment`, native `toolchain`, author
`source`/`symbol`, `constexprs`, `grid` and `num_cpu_threads`. Compilation belongs to
Triton; Bench supplies inputs, the numerical oracle and outer execution.

The result has a separate `python-jit-grid` scope. It records cold compile+launch,
first cache-hit launch, the warm Python-call median, and child CPU time. The warm
time includes Python argument binding, capability guards and native grid launch;
it must not be compared with native-only GGML as an operator speedup. Source/AOT
and native-grid measurements remain available through the native adapter above.
The runtime record includes the actual native toolchain identity and loaded
Triton/Weft/OpenMP libraries, which distinguishes installed wheel execution from
a development overlay without another measurement protocol.

Set `check_python_aot: true` to export and reload the just-compiled artifact through
`triton.load(...)[grid]`. Set `check_torch_compile: true` for an additional
`torch.compile(fullgraph=True)` numerical check when the selected private Torch
environment has the compiler's supported source integration. The latter is a
framework check, not the reported warm timing. Both options are off by default.
Current shared Tensor fixtures support float32, a single output, whole-workload
calls and reused inputs. They do not imply coverage of arbitrary alias layouts or
every PyTorch graph. Use the task's saved framework evidence for those contracts.

JIT replay uses the saved source and recorded target environment, with the same
input snapshot. `--source FILE` selects a replacement source; the legacy compiler
override is rejected. Do not run saved fullgraph requests against shared Torch
unless that environment was explicitly configured for the integration.

## Native llama.cpp packages

Set `target.llama.package_backend` to `triton`, and configure `llama.triton.python`
plus an optional `pythonpath` list for package preparation. The existing repeated
`--replace ROLE=ARTIFACT` options now call `triton.Package.create` on the exported
native artifacts, then deploy the package. There is no target-side relinking or
old Weft compiler fallback. Source replacements additionally need role bindings
with `signature`, `constexprs`, `toolchain` and backend options.

Use a compatible integrated llama.cpp binary and native SDK. A package role is
not a phase selection: `ggml.ime1/q4_centered_prefill` only matches that hook;
it cannot stand in for a decode IME kernel. `--phase` still selects which model
measurement to run. Report the measured-window role counters alongside timings;
`not_hit` means the observed phase did not call the replacement even if elapsed
time changed. Fixed normal llama.cpp results remain read-only baselines.
