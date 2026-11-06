# 直接从清洗后数据生成代码：无需旧 audit

需要一次执行环境检查、工具准备、生成和保存后核对时，可使用[一键完整脚本](CUSTOM_CODE_DATA_SCRIPT.md)。它默认创建带时间的新目录，保留此前已经生成的结果。

旧 `coding_v3/gripper_code_audit.jsonl` 已删除时，使用本工具重新生成，不再使用依赖 `--reference-audit` 的缓存修补命令。输入只需清洗后的数据集；不加载任何旧 PT、order 或 audit。

## 生成方式

`scripts/generate_custom_code_cache.py` 调用包内新版 `build_v3_gripper_code_fixed.py`，从当前数据的 `observation.state` 和 `task_index` 生成标签，并直接完成 release/place 回撤合并。无需先生成 `coding_v4_base` 再运行重建脚本。

- 任务 0：夹爪闭合对应 grasp，持握移动与释放对应 move_arm + release；释放后的回撤并入 release。
- 任务 1：pick 与 place；放置后的回撤并入 place。片段若只有已闭合夹爪到打开的过程，生成 place，不补造片段外的 pick。
- 任务 2：grasp，然后从本片段保留的 rotation6D 计算 rotate，确认开爪后添加 release。
- 任务 3：恒开夹爪推门时，使用观测到的推门和回撤 move_arm，保持物理接触待核实信息，不生成不存在的 grasp/release。

所有 policy code 均无注释。位姿目标、旋转角度均来自清洗后保留的状态；不改变 Parquet、视频、归一化统计或旧缓存。

## 旋转窗口细化

完整夹持区间通过检查时，沿用原计算结果。仅当该区间 XYZ 漂移超过阈值时，枚举区间内部的连续窗口，仍使用相同的漂移阈值 0.02 m、最小角度 8°、轴一致性 0.8。候选窗口至少包含 3 帧。

在合格窗口中选取绝对净旋转角最大的一个；角度按输出精度相同的情况下，优先更长窗口，再优先更早起点。audit 记录原区间、候选数、所选窗口、漂移、轴和角度。角度为相邻局部 SO(3) 增量的累积，不把超过 180° 的多帧旋转折回最短角。

不通过漂移检查以外的故障不会触发片段搜索。没有合格窗口、非法旋转、单帧接近 180° 的方向歧义仍会报告待核实，不为了凑齐调用数量而硬补 rotate。物理旋钮接触、局部轴是否为夹爪接近轴、角度正负方向仍需校准。

原始轨迹 3075 的真实样本验证：完整 `[25,44]` 区间漂移 0.06135 m；新版直接从姿态找到 `[25,43]`，漂移 0.00888 m，得到 83.52°，无需读取旧 audit。全量清洗后数据不在当前环境中，需在服务器生成并查看报告；新选窗口和角度可能与旧缓存略有不同。

## 末尾开爪的确认

默认仍要求 2 帧稳定状态。清洗可能保留第一次开爪、删除下一帧确认，因此在片段末尾留下未确认候选。

`--confirm-curation-boundaries` 会利用清洗后根目录的 `curation_source_map.json` 和 `meta/info.json` 中的 `trajectory_curation.source_root`，仅对这些候选读取源状态上下文，不需要旧代码审计：

1. 核对源任务和保留段完整状态与新数据逐值相同。
2. 源后续夹爪样本必须连续满足打开阈值，补足原来的稳定帧数。
3. 扩展后的同一夹爪检测器必须保留已有事件，并恰好确认这次打开。
4. 确认事件映射到最后一个保留帧，所有坐标和角度仍使用保留数据，不导入被删除的位姿、动作或回撤帧。

若源数据不可用或不满足确认条件，就保留待核实候选，不强制添加 release。去掉此参数可只读清洗后数据，但原 3075 这一单帧末尾开爪将继续未确认。

## 下载与运行

直接复制以下命令。新程序解压到独立工具目录，使用包内新版生成器；不覆盖项目现场源码、YAML、train.sh 或已有缓存。仍使用已有项目环境中的 PyTorch、NumPy、PyArrow、Pandas 和 v3 reader。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
TOOL_DIR="$PWD/custom_code_generator_v2"
CLEAN_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30_translation_v2

curl -fL \
  'https://github.com/youku-vip/agentic-wam/raw/refs/heads/codex/custom-data-review-files/deliveries/custom-code-generator-v2.zip' \
  -o /tmp/custom-code-generator-v2.zip
"$PY" -m zipfile -e /tmp/custom-code-generator-v2.zip "$TOOL_DIR"

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/generate_custom_code_cache.py" \
  --root "$CLEAN_ROOT" \
  --out-dir "$CLEAN_ROOT/coding_v4_regenerated" \
  --confirm-curation-boundaries \
  --knob-axis auto \
  --knob-angle-sign 1
```

输出目录必须尚不存在，不覆盖旧结果。程序每 200 条输出进度，完成后写入：

- `policy_code_idx_custom.pt`
- `episode_order.json`
- `gripper_code_audit.jsonl`
- `code_generation_report.json`

新缓存会完整检查 cache/audit/order/数据总量的一致性。报告列出各任务调用数、细化窗口、已确认的边界打开、未确认候选和缺少 rotate 的任务 2 episode，检查成功后才发布新输出目录。无需 VLM、GPU 或视频解码，也无需重算已有的清洗后归一化统计。

查看摘要：

```bash
"$PY" - "$CLEAN_ROOT/coding_v4_regenerated/code_generation_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding='utf-8'))
for key in ('episodes', 'frames', 'calls', 'task2_without_rotate', 'unresolved_terminal_candidates'):
    print(key, r[key])
print('boundary_open_confirmations', len(r['boundary_open_confirmations']))
PY
```

应保持当前清洗后的 4345 个 episode、285001 帧。若任务 2 仍缺旋转或存在未确认末尾事件，先查看所列 episode 的 audit/视频，不通过放宽阈值或降低稳定帧数凑齐数量。

生成并复核后，实际训练参数使用 `data.dataset_root` 指向上述 `CLEAN_ROOT`，`data.code_cache` 指向其中 `coding_v4_regenerated/policy_code_idx_custom.pt`；仍使用该根目录已有的 `dataset_stats_eef64.json`。更换数据/标签属于训练条件改变，不应当作原 run 的严格续训。

## 验证范围

相关 24 项测试通过：覆盖新的完整 Parquet → 缓存流程、四任务行为、窗口细化、多圈旋转、符号校准、无合格窗口、歧义旋转、缺少抓取的放置片段、截断开爪确认及拒绝、源状态不一致、无旧 audit/缓存、只读清洗后数据生成、输入不变和输出不覆盖。真实轨迹样本与全量服务器数据的验证范围分别记录，不以合成测试代替全量物理任务验证。
