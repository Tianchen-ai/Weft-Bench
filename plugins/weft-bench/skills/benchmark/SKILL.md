---
name: benchmark
description: 使用 Weft Bench 测量 Weft/RVV/IME kernel 与 GGML 的耗时，比较编译器或 kernel 改动前后表现，或按入口替换 llama.cpp 并分别测量 prefill/decode。适用于 Weft 主线性能开发、baseline 复用、热点关联与失败复现。
---

# Weft Bench

直接使用已安装的 `weft-bench`。算子使用现有 workload、输入与 GGML reference；
模型使用正常 llama.cpp 与显式选择的 Weft package。插件负责操作指引和本机环境发现，
测量、编译、package、SSH 与结果比较由现有工具完成。

## 找到入口

以下 `BENCH_SKILL` 指向本 `SKILL.md` 所在的绝对目录。不要推定插件缓存版本或当前工作目录。

```sh
python3 "$BENCH_SKILL/scripts/bench.py" --plugin-info
python3 "$BENCH_SKILL/scripts/bench.py" list
```

启动器将原参数交给 CLI，不改变当前工作目录。它读取本机
`${XDG_CONFIG_HOME:-~/.config}/weft-bench/codex.local.json` 中的 `command` 和 `config`。
没有该文件时使用 PATH 中的 `weft-bench` 与常规 `bench.local.json` 配置位置。
显式 `--config` 优先于 `WEFT_BENCH_CONFIG`，后者优先于本地默认值。
`--plugin-info` 只显示入口、配置位置和可用 target 名称，不连接设备。

配置缺失时，先使用任务已给出的配置；需要查找已保存的机器资源或凭据时使用资源插件。
依赖安装和配置方式见 [环境与操作](references/workflows.md)。

## 按任务选择流程

| 用户要完成的事 | 入口 |
|---|---|
| 当前 kernel 对 GGML | `op CASE --weft SOURCE:SYMBOL --target TARGET` |
| 已有编译产物对 GGML | `op CASE --weft ARTIFACT_DIR --target TARGET` |
| 改输入或 shape | `show CASE`，然后 `--inputs PRESET_OR_JSON` / `--shape ...` |
| 使用固定 ref | 同一场景加 `--baseline REF_RUN`；默认只测候选 |
| 改前改后比较 | `compare BEFORE AFTER`；配置了旧版时也可用 `check --case CASE` |
| 指定 llama.cpp 替换 | `e2e MODEL --replace ROLE=KERNEL --phase prefill|decode|both` |
| 解释或复现失败 | `explain RUN --case CASE` / `replay RUN --case CASE` |

使用 [环境与操作](references/workflows.md) 中与当前任务有关的例子。
用 `list --kind inputs`、`list --kind replacements` 和各命令 `--help` 查询当前安装版支持项。
换 kernel 使用参数，复用已有输入和 ref；只有计算语义或 IO 边界确实不同才需要增加 workload。

## 性能开发中需要保留的语义

- 用户只要求调查或解释时读取已有结果；要求测量或优化时运行相关 case。
  默认选择改动涉及的 workload，用户要求套件时再运行套件。
- 固定 ref 已匹配当前场景时直接复用。核对 shape、数据、cache 模式、线程、设备和计算范围；
  工具报告不匹配时呈现差异，补建对应场景的 ref，保留原结果。
- dot leaf、多行 dot、完整 projection、IME packed leaf 是不同边界。
  把普通 dot 的改善与实际模型路径关联，不能用 projection 成绩充当 dot 成绩。
- E2E 的 `--phase` 选择测量阶段；同一个 `--replace` 映射用于所有所选阶段。
  同一入口需要两种阶段专用 kernel 时分两次调用。
- `--replace` 只作用于已有 Weft llama.cpp hook 的对应入口，使用已集成候选构建。
  保留正常 baseline 的 RVV、repack、OpenMP、IME 等配置。
  从测量区间计数确认命中；`not_observed` 不能解释成零命中或已命中。
- 不根据一次微基准自动选择或更换模型 package。实际替换由用户请求及显式参数确定。
- 默认数值检查、预热和计时由工具处理。无需为每次 bench 再写测试脚本或扩展全量矩阵。
  只有用户要求时采集 `--profile`；已有热点使用 `hotspots` 复用。

输出保存在任务选择的目录，通常位于仓库外的运行目录。返回结果表或 CSV/JSON 链接、
对应范围和下一步定位入口；区分编译失败、数值失败、运行成功和性能改善。
输出与 replay 材料可能包含本机路径、SSH 别名及配置，只在任务需要时共享。
