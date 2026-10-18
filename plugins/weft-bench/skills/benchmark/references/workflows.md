# Environment and command recipes

These examples call the existing Weft Bench CLI through the plugin launcher.
Set `BENCH_SKILL` to the directory containing the loaded `SKILL.md`, `TARGET` to a
configured target name, and `RUNS` to the task's output directory. Weft Bench 0.3.0+
and Python 3.10+ are required. Use absolute source/artifact paths when switching
between Weft worktrees. Source files are host-side; model/library paths are target-side.

## Local installation and discovery

The plugin does not bundle the compiler, benchmark executable, GGML or models.
Install the bench from its checkout with `python3 -m pip install -e /path/to/weft_bench`
in the intended Python environment. A virtual environment is also supported.

Optional machine-only settings live at
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/codex.local.json`:

```json
{
  "command": ["/path/to/venv/bin/weft-bench"],
  "config": "/path/to/bench.local.json"
}
```

`command` is an argument array, never a shell expression. A Python module command
such as `["/path/to/python", "-m", "weft_bench.cli"]` is also accepted. Without local
settings, the launcher uses `weft-bench` from PATH and, when present,
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/bench.local.json`. Existing
`WEFT_BENCH_CONFIG` and explicit CLI `--config` still take priority.

```sh
python3 "$BENCH_SKILL/scripts/bench.py" --plugin-info
python3 "$BENCH_SKILL/scripts/bench.py" list --kind inputs
python3 "$BENCH_SKILL/scripts/bench.py" show rms_norm --inputs decode
```

Target configuration selects compiler, RVV/VLEN/extensions, native toolchain,
GGML libraries, optional SSH deployment and normal/integrated llama.cpp builds.
Use the bench checkout's `configs/example.json` and `docs/targets.md` for initial
setup. Keep real settings outside the plugin and repository.

## Operators and saved references

```sh
# Current source vs the target's GGML library, using an existing workload.
python3 "$BENCH_SKILL/scripts/bench.py" op rms_norm \
  --target "$TARGET" --weft /work/norm.py:my_norm --inputs decode \
  --output "$RUNS/norm-current"

# Use custom data or dimensions, retaining the same input for both implementations.
python3 "$BENCH_SKILL/scripts/bench.py" op rms_norm \
  --target "$TARGET" --weft /work/norm-artifact \
  --inputs /work/norm-inputs.json --save-outputs --output "$RUNS/norm-custom"

# Establish a reference once for this scenario.
python3 "$BENCH_SKILL/scripts/bench.py" op q4_k_dot \
  --target "$TARGET" --impl ggml --output "$RUNS/dot-ref"

# Reuse it. This does not re-time the reference implementation.
python3 "$BENCH_SKILL/scripts/bench.py" op q4_k_dot \
  --target "$TARGET" --weft /work/dot.py:kernel \
  --baseline "$RUNS/dot-ref" --output "$RUNS/dot-current"
```

Default short-call protocol: 3 warmups, 5 groups of approximately 10 ms, reused
inputs and one numerical check. `--cache cold` selects eviction before each call;
retain the same protocol when comparing history. Numerical oracle work is untimed.
Use `show` to obtain the actual port names, dtype, layout and shape before binding
a new kernel. Weft artifacts carry the ABI; no hand-built C benchmark is required.

## llama.cpp replacements

```sh
python3 "$BENCH_SKILL/scripts/bench.py" list --kind replacements
python3 "$BENCH_SKILL/scripts/bench.py" e2e llama2-7b-q4_0 \
  --target "$TARGET" --phase decode \
  --replace ggml.vec_dot/q4_0=/work/dot.py:kernel \
  --replace ggml.from_float/q8_0=/work/quant.py:kernel \
  --output "$RUNS/model-decode"
```

Repeat `--replace` for distinct roles or use `--replacement-file /work/replacements.json`:

```json
{
  "ggml.vec_dot/q4_0": "/work/dot.py:kernel",
  "ggml.from_float/q8_0": {"source": "/work/quant.py", "symbol": "kernel"}
}
```

The same role can have one implementation per invocation. A mapping creates a fresh
package containing exactly the requested roles. The tool delegates ABI validation,
packaging and native linking to Weft, then supplies the package to the integrated
llama.cpp build through `WEFT_GGML_PACKAGE`. Actual GGML dispatch determines whether
the hook is reached; repack and IME can take different paths from ordinary vec-dot.

Default phase is `both`: PP128/depth0 and TG32/depth512, each repeated 3 times.
`--scenario smoke` is a shorter integration check, not the default performance
comparison. `--prefill`, `--decode`, `--depth`, `--threads` and `--repetitions`
override the scenario. Keep prefill and decode as separate result rows.

`--impl normal` measures only normal llama.cpp without a replacement package.
Once its result is established, `--baseline MODEL_REF_RUN` measures only the
candidate by default. Read measured-window hit counts, not warmup totals. If the
binary lacks observers, the result reports `not_observed`. Configuring candidate
source/build/include lets the tool prepare observers and incrementally rebuild
the candidate harness. Source integration hooks are a prerequisite.

`prepare-model --target TARGET --replace ROLE=KERNEL --output DIR` prepares a
package without running a model. `--profile` is opt-in; use `hotspots --help` to
associate an existing profile with operator cases without re-running the model.

## Compare and reproduce

```sh
python3 "$BENCH_SKILL/scripts/bench.py" compare "$RUNS/before" "$RUNS/after"
python3 "$BENCH_SKILL/scripts/bench.py" explain "$RUNS/after" --case q4_k_dot
python3 "$BENCH_SKILL/scripts/bench.py" replay "$RUNS/after" --case q4_k_dot \
  --compiler /build/new/weft-compile --output "$RUNS/replay"
python3 "$BENCH_SKILL/scripts/bench.py" check --target "$TARGET" --case q4_k_dot
```

`check` requires explicitly configured `check.reference` and `check.previous`.
Do not select a historical best result as the checkpoint. Failed or incompatible
rows remain results; inspect their diagnostics/scope before attributing a change
to the compiler. A replay uses saved inputs and compilation material; `--source`
selects current author source instead of the saved canonical IR when intended.
