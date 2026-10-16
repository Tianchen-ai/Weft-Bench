# Select, compare, inspect

Install with `python -m pip install -e .`. Copy `configs/example.json` outside the
checkout, fill in the compiler/library/device paths, then set:

```sh
export WEFT_BENCH_CONFIG=/path/to/bench.local.json
```

## Operators

```sh
weft-bench list
weft-bench list --kind inputs
weft-bench show rms_norm --inputs decode
weft-bench op q4_k_dot --target rvv128
weft-bench op rms_norm --weft /work/norm.py:my_norm --inputs decode --target rvv128
weft-bench op rms_norm --weft /work/artifact --inputs /work/input.json --target rvv128
```

Default comparison is GGML versus Weft. Omit `--weft` to select the configured
author kernel for the operation. A source candidate uses `file.py:symbol`; an
artifact candidate uses its directory. GGML source/library selection comes from
the target's `ggml` configuration. The Python API can also pass a `GGML` object.

Choose `default`, `decode` or `prefill` inputs; `--shape M=2,N=64` overrides
dimensions, and `--seed 123` selects repeatable generated data. Use
`--inputs /work/input.json` for custom buffers and parameters. See the input guide
for supported fields. Input files are copied into the run for both implementations
and for replay.

`--suite quick` selects eight predefined operations. `--suite llama-hot` selects
the optional model-related workloads. Dot leaf, multiple independent dots, full
projection and packed IME leaf retain distinct computation boundaries.

## Input and output handling

`show OP` reports each named input/output, dtype, dimensions and layout.
Each worker prepares one input instance, allocates distinct outputs, checks each
implementation against the workload's oracle, and times only its invocation.

`--save-outputs` writes each implementation's output to `Y.bin`; its dtype and
shape are in the result's `details.ports`, and the file path in `details.outputs`.
Floating buffers use native little-endian float32; quantized outputs retain their
GGML block encoding. Allocation, loading, validation and output export are untimed.
Requested output files are retained even when numerical checking fails.

Defaults: three warmups and five sample groups, approximately 10 ms per group for
short calls. `--cache cold` evicts before each call; `--cache reuse` reuses prepared
inputs. Required work such as projection activation quantization remains timed.
GEMM/prefill projection use a bounded scalar output sample plus finiteness checking;
other built-in floating workloads check their complete outputs.

## llama.cpp replacement

```sh
weft-bench list --kind replacements
weft-bench e2e llama2-7b-q4_0 --target rvv128 \
  --replace ggml.vec_dot/q4_0=/work/dot.py:my_dot
weft-bench e2e llama2-7b-q4_0 --target rvv256-ime \
  --replace ggml.ime1/q4_centered_prefill=default --phase prefill
weft-bench e2e llama2-7b-q4_0 --target rvv128 \
  --replacement-file /work/replacements.json --prefill 64 --decode 16 --depth 256
```

`default` on the right selects the configured kernel for that role. A source,
artifact directory, or JSON entry with source/symbol/options selects your own.
Repeat `--replace` to replace several roles. An explicit mapping creates a fresh
package containing exactly those roles; it does not inherit unrelated replacements
from a previously configured package.

`--scenario default` uses PP128/depth0 and TG32/depth512 with three repetitions.
`--scenario smoke` uses PP16 and TG4/depth32 once. Model names resolve to configured
GGUF paths; an explicit target-side `.gguf` path is accepted as well.
Only the requested model runs. Prefill and decode have separate rows.

Use `prepare-model --target TARGET --replace ROLE=KERNEL --output DIR` to prepare
a package without running the model. It saves the replacement selection, public
package bindings and deployed package location.

Configure the integrated build's source/build/include paths to let the tool prepare
measurement-window counters and incrementally build llama-bench. Subsequent calls
reuse the prepared harness. Without those paths, the supplied binary runs as-is;
a binary without usable counters reports `not_observed`.

## Saved comparisons and diagnosis

These are optional outer workflows:

```sh
weft-bench op q4_k_dot --impl ggml --target rvv128 --output runs/ref
weft-bench op q4_k_dot --target rvv128 --baseline runs/ref --output runs/current
weft-bench compare runs/ref runs/current
weft-bench explain runs/current --case q4_k_dot
weft-bench replay runs/current --case q4_k_dot --compiler /build/new/weft-compile
```

Model fixed references work the same way: `e2e ... --baseline runs/model-ref`
runs only Weft by default. Numerical oracles may still execute outside timing.

`check --target TARGET` remains a daily convenience using configured
`check.reference` and `check.previous`. It adds the historical multi-row dot
scenario and an IME scenario when supported. `checkpoint` and `import-history`
select existing observations without choosing historical best results.

Version 0.3 identifies the shared-input worker protocol as `native-workload-v3`.
Earlier results remain readable, with their original protocol. Establish the new
protocol's reference once; it is then reusable. Changed inputs or computation
boundaries are shown explicitly instead of receiving a misleading ratio.
