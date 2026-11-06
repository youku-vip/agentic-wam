# 清洗后 coding_v4 的旋钮标签修复

此工具只修改代码标签。清洗后的 Parquet、视频、筛选范围和 `dataset_stats_eef64.json` 均不变，输入缓存也不覆盖。输出为新的 `coding_v4_fixed`。

## 本次缺口的原因

上传的清洗后 audit 为 4345 个 episode、285001 帧。任务 2 的 1087 条轨迹中，只有 886 条包含 `rotate`。

逐项对照原始 audit 和 `curation_source_map.json` 后，缺少的 201 条全部有原始 audit 记录的细化旋转窗口，并且窗口完整保留在新数据中。当前基础生成器检查整个“夹爪闭合到打开确认”的区间，因其中的平移超过阈值而省略了旋转；旧结果在区间内进一步选取了平移较小的旋转窗口。问题出在这一步生成逻辑的差异，不需要放宽清洗阈值。

另有一条缺少 `release`：新 episode 3072 对应原 episode 3075，保留原帧 `[0, 44)`。原第 43 帧已经开爪，第 44 帧再次开爪使“两帧稳定确认”成立，但第 44 帧已被清洗掉。原始已上传轨迹的第 42、43、44 帧夹爪值为 `[0, 1, 1]`；原第 25–43 帧姿态可复算出约 83.5233°。修复仅使用第 44 帧的夹爪值确认边界事件，不使用其异常位姿，也不补入被删除的回撤帧。

`pick=1086` 则有另外的原因：原 episode 817 仅保留原帧 `[27, 76)`，抓取闭合在已删除的第 24 帧。工具保留现有的放置阶段，不补造一次抓取。

## 修复前的校验

程序要求原数据集仍可读取，以及一份包含细化窗口的原始 audit；通常是原根目录 `coding_v3/gripper_code_audit.jsonl`，此前基于该文件合并生成的旧 `coding_v4` audit 也可以使用。清洗后重新生成的 `coding_v4_base` 不包含这些旧窗口，不能作为参考。

程序先检查当前 PT、audit、order、info 的一致性，再核对每个源 episode 映射。每个待恢复旋转必须同时满足：

- 原 audit 存在唯一的细化 `rotate`，窗口完整落在保留范围内。
- 新旧 axis/sign 配置一致，代码角度与 audit 元数据相符。
- 真实 Parquet 中，保留窗口的左臂 XYZ、rotation6D 与源窗口逐值一致。
- 从保留姿态的相邻局部 SO(3) 旋转复算角度，与原标签相差不超过 0.05°；容差用于已记录轴和角度的四舍五入。
- 边界释放另外要求源夹爪确有 `闭、开、开`，且新数据末尾两帧与其一致。

任一检查不通过就退出，不写新缓存。程序不把参考 audit 当作可执行指令；代码只通过 AST 解析读取，PT 使用 `weights_only=True` 加载。所有生成的 policy code 均无注释。

## 下载与运行

在已有项目环境执行。压缩包解压到独立工具目录，不覆盖项目源码或现场配置；程序通过 `PYTHONPATH` 使用项目现有的 v3 reader。依赖是训练环境已有的 PyTorch、NumPy、Pandas、PyArrow，无需 GPU、模型或视频解码。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
TOOL_DIR="$PWD/curated_knob_code_repair_v1"

curl -fL \
  'https://github.com/youku-vip/agentic-wam/raw/refs/heads/codex/custom-data-review-files/deliveries/custom-curated-knob-code-repair-v1.zip' \
  -o /tmp/custom-curated-knob-code-repair-v1.zip
"$PY" -m zipfile -e /tmp/custom-curated-knob-code-repair-v1.zip "$TOOL_DIR"

SOURCE_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
CLEAN_ROOT="${SOURCE_ROOT}_translation_v2"

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/repair_curated_knob_code.py" \
  --root "$CLEAN_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --cache-dir "$CLEAN_ROOT/coding_v4" \
  --reference-audit "$SOURCE_ROOT/coding_v3/gripper_code_audit.jsonl" \
  --out-dir "$CLEAN_ROOT/coding_v4_fixed"
```

`--out-dir` 必须不存在；如之前已有结果，使用另一个目录名。程序会输出预计修复数量和每 50 条姿态校验的进度。输出包含：

- `policy_code_idx_custom.pt`
- `episode_order.json`
- `gripper_code_audit.jsonl`
- `knob_repair_report.json`：每条修复的源 episode、窗口、复算角度、夹爪确认、输入文件哈希。

如果服务器上的实际文件与上传文件一致，并通过全部姿态/夹爪检查，预计结果为：

| API | 修复前 | 修复后 |
|---|---:|---:|
| grasp | 2171 | 2171 |
| move_arm | 3258 | 3258 |
| release | 2170 | 2171 |
| pick | 1086 | 1086 |
| place | 1087 | 1087 |
| rotate | 886 | 1087 |

episode 数 4345、帧数 285001、所有阶段帧范围和有标签帧数不变。原物理接触、旋钮轴和角度正负方向的待核实信息保留；数学复算通过不能替代这些校准。

本地已完成上传文件的全部映射和参考窗口核对，并对已上传的原 episode 3075 原始姿态/夹爪做了数值验证；其余 201 条没有上传原始姿态，需以上命令在服务器完成验证。程序不会用样例证据代替这 201 条真实数据的检查。

## 训练使用

在实际训练启动参数列表中设置：

```text
data.dataset_root=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30_translation_v2
data.code_cache=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30_translation_v2/coding_v4_fixed/policy_code_idx_custom.pt
```

继续使用清洗后根目录现有的 `dataset_stats_eef64.json`，无需因本次代码标签修复重新统计动作。当前 `train.sh` 没有透传额外的命令行参数，应修改其实际启动参数列表或对应 YAML；不要仅把这些参数附加在 `bash train.sh` 后面。

若之前的 checkpoint 使用旧的 4348 条轨迹，改到清洗后的 4345 条及新标签属于新一轮训练，应使用旧权重初始化新 run。只有数据、标签和采样配置不变时，才使用完整训练状态继续原 run。

## 工具验证

`tests/test_curated_knob_code_repair.py` 检查实际合成 Parquet 的完整 CLI 流程、缓存一致性、原文件不变、窗口丢失/校准不符/姿态改变/夹爪未确认时拒绝修复，以及重复运行不覆盖输出。合成数据只用于测试，不是本次真实数据的验收结果。
