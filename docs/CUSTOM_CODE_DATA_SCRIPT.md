# 一键生成原始或清洗后 custom 数据的训练代码

入口为 `scripts/generate_custom_code_data.sh`。新版入口同时支持清洗前的原始数据和清洗后的数据，自动选择是否需要清洗边界确认。生成算法沿用已经在清洗后 4345 条轨迹、285001 帧上运行成功的 v2，将环境检查、工具准备、代码生成和保存后核对放在一次执行中。不修改现有训练启动文件或 YAML，不重新统计动作。

## 下载运行

```bash
curl -fL \
  'https://github.com/youku-vip/agentic-wam/raw/refs/heads/codex/custom-data-review-files/scripts/generate_custom_code_data.sh' \
  -o /tmp/generate_custom_code_data.sh

PROJECT_DIR=/data/algorithm/agentic-wam \
PY=/usr/local/bin/python3.11 \
DATA_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30 \
bash /tmp/generate_custom_code_data.sh
```

以上生成原始数据的代码。清洗后数据使用相同脚本，仅切换根目录：

```bash
DATA_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30_translation_v2 \
bash /tmp/generate_custom_code_data.sh
```

已有项目和依赖可直接使用。脚本优先使用随脚本打包的生成器，或项目中已有的完整生成器；缺少生成器时，下载固定版本工具包，检查包的 SHA256 和内部文件哈希后安装到独立目录。不会自动安装或升级 Python 依赖。需要现有 Python 环境中的 NumPy、PyTorch、Pandas、PyArrow，以及项目 v3 reader。

完整工具包 `deliveries/custom-code-generator-v3-complete.zip` 包含新版一键脚本、全部三个生成源码文件、说明、测试和固定版本的原始 v2 生成器包。v3 表示新增原始/清洗后输入的自动识别，代码生成算法仍为 v2。解压到独立目录后，也可运行其中的 `scripts/generate_custom_code_data.sh`，通过 `PROJECT_DIR` 指定现有项目。旧版 v2-complete 压缩包保留。

## 输入和输出

通过 `DATA_ROOT` 指定原始或清洗后的数据根目录。不读取旧代码缓存、episode order 或 audit。默认仍为此前的 translation_v2 数据集；旧的 `CLEAN_ROOT` 参数继续兼容。两者同时设置时以 `DATA_ROOT` 为准。

默认 `CONFIRM_CURATED_BOUNDARIES=auto`：

- 原始数据没有清洗映射和 curation source_root，自动使用 `raw` 模式，不要求清洗元数据，也不读取其他根目录的状态。
- 清洗后数据同时具有 `curation_source_map.json` 和 info 中的 curation source_root，自动使用 `curated` 模式，仅对未确认的末尾开爪读取源夹爪上下文。
- 清洗元数据只存在一部分时，auto 会提示补齐，避免静默当成原始数据；也可明确设置 `CONFIRM_CURATED_BOUNDARIES=0`，仅使用当前数据。

所有代码位姿、角度均来自 `DATA_ROOT` 对应的数据。源证据不足时保留待核实候选。设置 0 不会把未确认的末尾开爪强制补成 release；设置 1 则明确要求完整清洗元数据。

每次默认生成到一个新目录：

```text
<DATA_ROOT>/coding_v4_regenerated_YYYYMMDD_HHMMSS/
    policy_code_idx_custom.pt
    episode_order.json
    gripper_code_audit.jsonl
    code_generation_report.json
```

目录时间采用北京时间。已有 `coding_v4_regenerated` 和其他缓存均保留。也可明确指定一个尚不存在的输出目录：

```bash
OUT_DIR=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30_translation_v2/coding_v4_complete \
bash /tmp/generate_custom_code_data.sh
```

若该目录已存在，脚本在生成前退出，不删除或覆盖。脚本报错时会保留已生成的输出以便检查，仅清理自己的临时工作目录。

## 自动执行的流程

1. 检查项目、输入元数据、Python 依赖并识别原始/清洗后输入；需要时检查清洗映射，记录 info/map/已有归一化统计文件的哈希。
2. 准备生成器；首次需要下载安装时校验固定工具包。
3. 直接从状态生成四任务标签，细化旋转窗口，合并 release/place 回撤，清理所有代码注释。
4. 重新读取写出的 PT、audit、order 和生成报告，核对每帧索引、阶段覆盖、episode/frame 总数和 API 调用计数。
5. 确认代码无注释、输入元数据和已有统计文件哈希不变；打印缺少旋转的 episode 和未确认末尾候选。
6. 打印实际训练需要的 `data.dataset_root` 和 `data.code_cache`。将这两项写入现有启动参数列表；脚本本身不会启动训练。

`saved_cache_checks=passed` 表示写出文件通过一致性检查，不替代物理旋钮接触、角度正负方向或实机成功率验证。非空的 `task2_without_rotate`、`unresolved_terminal_candidates` 仍需查看相应轨迹。

## 参数

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| PROJECT_DIR | /data/algorithm/agentic-wam | 现有项目路径 |
| PY | /usr/local/bin/python3.11 | Python 解释器 |
| DATA_ROOT | 上述 translation_v2 根目录 | 原始或清洗后的主数据集 |
| CLEAN_ROOT | DATA_ROOT 未设置时使用 | 兼容此前的调用方式 |
| OUT_DIR | 主根目录下带时间的新目录 | 输出缓存目录 |
| TOOL_DIR | 自动选择完整的现有或随包生成器 | 手动指定生成器位置 |
| CONFIRM_CURATED_BOUNDARIES | auto | 自动识别；也可显式设 0 或 1 |
| KNOB_AXIS | auto | 旋钮的夹爪局部轴，可设 x/y/z |
| KNOB_ANGLE_SIGN | 1 | 角度正负校准，可设 -1 |

对于当前已生成成功的清洗后数据，参考调用数为 grasp=2171、move_arm=3258、release=2171、pick=1086、place=1087、rotate=1087。原始数据应按自己的 4348 条轨迹、285262 帧核对，调用数由实际事件决定。脚本按所选输入的元数据核对，不硬编码这些数值。训练也应使用所选根目录自己的代码缓存和统计文件。

一键入口覆盖现有生成器的完整生成/保存后核对/拒绝覆盖、离线安装、错误下载拒绝、原始/清洗后自动识别、显式关闭清洗边界确认和原始数据拒绝强制清洗确认。验证输入和统计文件不变，生成算法保持不变。
