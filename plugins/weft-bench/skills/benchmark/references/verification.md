# Configured compiler and runtime verification

Use `verify` when the task calls for numerical, compiler or installed-runtime
validation. For latency and comparisons use `op`, `check` or `e2e`. Verification
reuses the same workload inputs, numerical oracles, typed ABI and native adapters;
it does not remeasure fixed GGML or llama.cpp baselines.

Select the task's configured environment and verification config explicitly:

```sh
python3 "$BENCH_SKILL/scripts/bench.py" --environment triton-weft verify \
  --config /work/bench.local.json --target cpu --list
python3 "$BENCH_SKILL/scripts/bench.py" --environment triton-weft verify \
  --config /work/bench.local.json --target cpu --case 'operators/q4_k_dot' \
  --output /work/verify/cpu
```

Environment, target, group and case names come from configuration. `--list` does
not connect to a device. Select cases affected by the task; a complete configured
inventory is appropriate only when that is the requested scope.

The repository's `configs/verification-riscv.example.json` lists optional scan,
atomic RMW/CAS/contention, noinline-helper and loop/context-return probes. Add
selected entries to the task's external config; keep the actual installed compiler
and target Python explicit. These probes do not imply support on older wheels.
For a framework upgrade, first verify the ordinary CPU Tensor and required Torch
interfaces in that private environment, then select the existing framework cases.
Use wheels matching architecture, Python ABI and libc, or build a compatible wheel;
do not copy a different target's installed package across incompatible ABIs.

Product examples can be configured as `kind: "usage"`: `contract: "triton_basics"`
selects one `operation`, `autotune_export` checks tuning/export/reload, and
`pytorch_cpu` checks the hand-written call and Inductor output. Supply the real
example source and its `imports` roots. The existing worker runs that source and
checks complete outputs; it does not treat printed previews as numerical evidence.

## Prepare, execute and resume

- `--prepare-only --workers N` compiles source entries on the host using Triton's
  existing compiler. Its `preparation.json` reports `compiled`, an expected
  `rejected`, or the actual failure. It is not numerical evidence.
- `verification.prepared_entries` in the external target config selects one or
  more preparation manifests. Execution consumes their exported native libraries,
  without recompiling C. Program source entries use the product's
  `Program.signature`, `Config` and `Program.compile` APIs.
- Native operator validation invokes the prepared candidate once and checks its
  outputs. There is no warmup or timing calibration. Program and framework probes
  retain their configured sequences, such as two stateful calls or autotuning.
- Keep one execution queue per target. Host preparation can use the worker count
  suitable for the available machine; it does not parallelize device measurement.
- `--resume` continues pending entries. `--case ID --rerun failed` retries selected
  failures; `--case ID --rerun all` explicitly replaces a selected result.
  `--stop-after-current --output RUN` requests a stop at a case boundary.
- Keep the selected compiler/runtime installation stable while a queue runs.
  A changed identity requires an explicit selection for rerun; an environment
  change during execution is recorded as `stale`, not a numerical failure.

## Read evidence and reproduce

`verification.json` includes both selected and whole-ledger counts. Use the selected
counts for a partial run: older pending entries may belong to a different selection.
Each entry links its current diagnostics under `cases/ID/current`; reruns retain
the first and latest failure. `failures.json` keeps the non-pass entries visible.

For a native operator, the entry's `result` points to ordinary Bench results:

```sh
weft-bench explain /work/verify/cpu/cases/operators/q4_k_dot/current/results.json
weft-bench replay /work/verify/cpu/cases/operators/q4_k_dot/current/results.json \
  --output /work/replay/q4_k_dot
```

Native replay defaults to the saved artifact when one exists. Source replay uses
the recorded Triton Python/toolchain, signature and options; `--compiler` is only
for the legacy Weft CLI. For a historical flat source snapshot with package-relative
imports, pass `--source /work/author/package/kernel.py` from the matching checkout;
the flat snapshot alone does not preserve its package dependencies. Framework,
runtime and usage entries are rerun through `verify --case ID --rerun all`.

Input/output snapshots and partial compiler IR are diagnostic evidence. A sampled
oracle covers only its declared output indices; exact unchanged-artifact evidence
may be reused with its original execution provenance, not relabeled as a new run.
`--coverage` maps an explicitly configured source inventory to cases and static
helper calls. Static reachability is not proof of runtime execution.

Report unsupported framework capabilities and hardware-inapplicable cases
explicitly. An expected compiler rejection has `checked: false`; its successful
expectation check is not a numerical pass. No correctness result implies a speedup.
