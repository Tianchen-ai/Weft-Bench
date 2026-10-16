# Use Weft Bench from Codex

The `weft-bench` plugin gives a Codex agent the existing operator and model
workflows, plus a launcher that finds this machine's CLI and configuration.
It is a local skill plugin: agents call the real CLI through their terminal.
It does not introduce another benchmark engine or require an MCP service.

## Install

From a checkout, with Codex CLI installed:

```sh
python3 -m pip install -e .
codex plugin marketplace add .
codex plugin add weft-bench@weft-tools
codex plugin list --marketplace weft-tools --json
```

The repository marketplace points only at `plugins/weft-bench`. Installing it
enables the plugin for the user, so agents in other repositories can discover it.
Start a new Codex chat after installation; if the desktop plugin list is stale,
restart the desktop app. This installs a local plugin, not a public directory entry.

For a virtual environment or a configuration outside the current shell, create
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/codex.local.json` outside the repository:

```json
{
  "command": ["/path/to/venv/bin/weft-bench"],
  "config": "/path/to/bench.local.json"
}
```

Otherwise the launcher finds `weft-bench` on PATH and uses
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/bench.local.json` when it exists.
Explicit `--config` and `WEFT_BENCH_CONFIG` override these defaults.
Machine addresses, SSH credentials, model paths and saved results are not packaged.
See [environment setup](targets.md) for the bench target configuration.

## Ask the agent

Examples:

- “使用 Weft Bench 插件，把这个 Q4_K kernel 与 GGML 比较，复用已有 baseline。”
- “使用 Weft Bench 插件，替换 llama.cpp 的 Q4_0 vec-dot，分别测 prefill 和 decode。”
- “使用 Weft Bench 插件，解释这次结果和上次的差别，并复现失败项。”

The bundled skill is `weft-bench:benchmark`. It is eligible for automatic selection
on relevant Weft performance tasks. It discovers supported workloads, input presets
and replacement slots from the installed CLI. User intent determines whether to
read existing results, run selected operators or measure model stages.

The launcher can be checked directly from this checkout without connecting a target:

```sh
python3 plugins/weft-bench/skills/benchmark/scripts/bench.py --plugin-info
python3 plugins/weft-bench/skills/benchmark/scripts/bench.py list --kind replacements
```

After installation Codex loads its cached plugin copy. To update edited plugin
content, bump the plugin manifest version, run `codex plugin add weft-bench@weft-tools`
again, and verify the installed version. Editable installation keeps the CLI tied
to the bench checkout independently of the plugin instructions.

The package layout follows the supported
[Codex plugin manifest and local marketplace format](https://developers.openai.com/plugins/build/plugins).
