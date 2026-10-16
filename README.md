# Weft Bench

Compare a Weft kernel with GGML, or compare normal llama.cpp with explicitly selected
Weft replacements. Pick an operation, inputs and implementation; the tool owns native
measurement and output handling.

```sh
git clone https://github.com/Tianchen-ai/Weft-Bench.git
cd Weft-Bench
python -m pip install -e .
export WEFT_BENCH_CONFIG=/path/to/bench.local.json

# Preset kernel and input, compared with the configured GGML library.
weft-bench op q4_k_dot --target k1

# Your kernel, existing workload and reference. No benchmark driver to write.
weft-bench op rms_norm --weft /work/norm.py:my_norm --inputs decode --target k1

# Select exactly which llama.cpp entry to replace.
weft-bench e2e llama2-7b-q4_0 --target sg2044 \
  --replace ggml.vec_dot/q4_0=/work/dot.py:my_dot --scenario default
```

The supported scope is intentionally concrete: GGML operator references and preset
llama.cpp workloads, with Weft source or artifact candidates. Runtime/library paths,
compiler targets and local/SSH deployment are external configuration.

| Select independently | Meaning |
|---|---|
| Operation | Computation, input/output contract and numerical oracle |
| Inputs | Preset dimensions, seed, custom arrays/files and scalar parameters |
| Weft kernel | Source + symbol, compiler options, or an existing artifact |
| GGML library | Native reference build and its headers |
| Model replacements | Explicit GGML role → Weft kernel bindings |
| Execution environment | Toolchains, ISA/VLEN, local or SSH worker, affinity |

Both operator implementations consume the same prepared input instance and have
separate output buffers in one native worker. Implementations supply calls; the
worker supplies checking, warmup and timing. Setup, compilation and transport are
outside the timed region.

```python
from weft_bench import compare, Inputs, WeftKernel
from weft_bench.loading import load_config, select_target

target = select_target(load_config("/path/to/bench.local.json"), "k1")
rows = compare(
    "rms_norm",
    candidate=WeftKernel(source="/work/norm.py", symbol="my_norm"),
    inputs=Inputs(shape={"M": 1, "N": 4096}, seed=410),
    target=target,
)
```

Terminal tables, CSV and JSON are the normal outputs. Optional output tensors,
compiler diagnostics and captured IR support inspection and replay. History comparison
and fixed result reuse are outer conveniences; no previous result is required to bench.

| Guide | Contents |
|---|---|
| [Quick start](docs/quickstart.md) | Choose workloads, kernels, inputs and replacements |
| [Input and implementation API](docs/extending.md) | IO contracts, custom data, Python usage |
| [Environment setup](docs/targets.md) | GGML, Weft, llama.cpp, RVV/IME and SSH |
| [Ownership and migration](docs/migration.md) | Core versus orchestration; older saved results |
| [Codex plugin](docs/codex-plugin.md) | Install the agent workflow and connect local configuration |
| [中文入口](README.zh-CN.md) | 产品与日常使用 |

Requires Python 3.10+ and the selected target's native C/C++ toolchain. GGML-only
measurement does not import Weft. NumPy is optional for NumPy input files/arrays.
Models, machine configuration and generated runs remain outside version control.
Licensed under Apache-2.0.
