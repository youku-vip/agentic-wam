# 遥操数据的平移稳定性筛选（v2.1）

工具：`scripts/curate_custom_trajectories.py`；版本 `custom-paper-curation-v2.1-translation`。

v2.1 修复静止窗口与最短片段长度耦合的问题：不足 32 帧的片段跳过静止判断，由 min_segment_frames 独立决定是否保留。默认最短长度仍为 33 帧，因此在当前配置下保留范围不变；已有 v2 计划可继续导出，不需要仅为修正原因名称重跑全量。

依据 [Open-HumanScale，arXiv:2611.06817v1](https://arxiv.org/pdf/2611.06817) 第 2.2 节、第 6–7 页，按高质量遥操数据的需求收窄为四项：平移稳定性、异常处理、静止筛选、平滑器与数值阈值。论文没有公开具体平滑器和阈值，本实现的参数是可修改的起点，不是论文原值或已经校准的硬件上限。

## 默认处理规则

| 项目 | v2 默认行为 |
|---|---|
| 平移稳定性 | 检查左右 observation.state 的 XYZ：平滑残差、相邻位移、加速度、jerk |
| 异常处理 | 残差超阈值，且该帧落在另一项平移动态超阈值指标的支持范围内，才列为剔除候选；按候选帧切连续片段 |
| 静止筛选 | 默认开启，只裁长时间静止的首尾，保留中间等待、接触及停顿；旋钮任务 2 默认跳过静止裁剪 |
| 平滑器与阈值 | 7 帧局部二次拟合，使用实际 timestamp；拟合结果只用于检测，保留片段的原始 state 不平滑写回 |

稳定性判定只使用 XYZ。旋转、关节和夹爪的变化不作为运动异常筛选条件，也不按完整状态重复自动删除记录。保留读取、有限值、时间顺序和导出索引等基础完整性检查。

已上传样本中的动作是下一帧状态标签，因此只用记录的 state 做运动检测，避免把同一次回跳又提前一帧重复剔除；导出时再逐 episode 验证 action 的下一帧标签一致性。夹爪切换覆盖只写入复核报告，不参与筛选。

## 参数

默认配置：`configs/trajectory_curation.example.json`。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| smoothing.window_frames | 7 | 局部窗口，奇数 |
| smoothing.polynomial_degree | 2 | 拟合次数；支持非均匀 timestamp |
| thresholds.position_residual | 0.025 m | 原始位置与拟合位置的距离 |
| thresholds.position_step | 0.08 m | 相邻采样位移 |
| thresholds.acceleration | 8 m/s² | 位置二阶差分的范数 |
| thresholds.jerk | 150 m/s³ | 位置三阶差分的范数 |
| anomaly_rule | residual_and_dynamics | 两类指标共同确认后筛选 |
| min_segment_frames | 33 | 当前筛选参数；loader 支持短片段末尾补帧，33 不是读取数据的硬性下限 |
| idle.mode | trim_edges | 只裁首尾静止，保留中间停顿 |
| idle.window_frames | 32 | 静止滑窗长度，10 Hz 时约 3 秒 |
| idle.position_range_m | 0.002 m | 窗口内左右 XYZ 变化范围都不超过 2 mm 时记为静止 |
| idle.skip_task_indices | [2] | 旋钮任务允许位移很小；此列表按本数据实际 task_index 设置 |

`position_range_m` 越小，越容易把微小位移视为活动，静止裁剪越少。各轴极差合成为一个欧氏范数，左右臂取最大值。

加速度涉及 3 帧，jerk 涉及 4 帧。默认联合判据只标记同时具有残差证据的帧，单独动态峰值保留为复核事件，不展开剔除整个支持范围。该策略用于减少正常启停的误删，不能保证已经精确定位所有坏帧，也可能漏过残差较小的异常。

需要更严格的任一阈值触发规则，可设置 `anomaly_rule: "any"`；此模式剔除所有超阈值指标的完整支持范围。它仍只检查平移。单项阈值可以设为 `null`，但联合判据要求启用残差和至少一项动态阈值。

静止默认不使用“活跃窗口占比小于 70% 则丢弃整段”的规则。如需该论文式规则，将 `idle.mode` 改为 `paper`，`idle.min_active_fraction` 为 0.7；task 2 的静止豁免仍以配置为准。其他数据集应根据真实任务编号调整 `skip_task_indices`。

片段不足静止窗口时，idle_evaluation 为 insufficient_window，active_fraction 为 null，表示尚未评估静止；如果它不足最短保留长度，丢弃原因记为 short_segment。降低 min_segment_frames 后，短于静止窗口但达到保留门槛的片段可以保留，不会因无法计算静止窗口而被删掉。使用较短片段时应评估末尾 padding 比例及代码分段效果。

时间戳若仅为导出时生成的 frame_index/10，导数表示该时间假设下的结果，不证明真实采集加速度。阈值应结合正常遥操样本和全量分布调整，不能只用已挑出的异常样本调参。

## 获取与使用

工具只在 CPU 上运行，不加载模型或 VLM，分析不解码视频。依赖现有项目的 NumPy、Pandas、PyArrow 和 v3 reader。解压到单独目录，不覆盖现有训练配置：

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
TOOL_DIR="$PWD/custom_paper_translation_v2_1"

git fetch origin codex/custom-data-review-files
git show FETCH_HEAD:deliveries/custom-paper-trajectory-curation-v2.1-translation.zip \
  > /tmp/custom-paper-trajectory-curation-v2.1-translation.zip
unzip -n /tmp/custom-paper-trajectory-curation-v2.1-translation.zip -d "$TOOL_DIR"
```

先对四种任务样本生成只读筛选计划，输出目录必须是新目录且位于原始数据根目录之外：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/curate_custom_trajectories.py" plan \
  --root "$ROOT" --episodes 0 1 2 3261 \
  --out ./trajectory_translation_sample_v2_1
```

用 `--config /path/to/your_config.json` 指定调整后的参数。`--no-idle-filter` 可临时关闭静止裁剪；`--idle-filter` 临时开启配置中的静止策略。

`--raw-json /path/to/raw_episodes.json` 可替代 `--root`，用于已有数值样本的预览；没有完整元数据和视频，不能直接导出训练根目录。

## 输出和全量处理

- summary.json：分任务保留情况、候选帧数、其他丢弃帧数、实际参数。
- events.csv：所有超阈值事件；used_for_exclusion=false 表示只复核，未用于剔除。
- frame_flags.csv：实际剔除候选帧及原因，帧数已去重。
- segments.json：保留的 [start_frame,end_frame)、源数据哈希和参数。
- curated_raw_episodes.json：raw JSON 输入时的数值预览，末帧规则见下文。

`stability_rejected_frames` 是稳定性候选帧数，`other_dropped_frames` 包括静止裁剪与过短片段。被裁成不足 33 帧的有效片段也会丢弃，因此保留比例不能仅从候选帧数推断。报告中的夹爪切换缺失提示用于复核切段是否破坏关键阶段，不证明物理成功或失败。

v2.1 增加 dropped_segment_frames_by_reason（各原因对应的完整丢弃片段帧数）、idle_trimmed_edge_frames（首尾静止裁剪帧数），两者之和为 other_dropped_frames。source_retention_counts 区分完全保留、部分保留、完全丢弃的源轨迹；fully_dropped_source_episodes 和 partially_retained_with_missing_gripper_transitions 分开报告，避免把整段未保留的切换与仍保留片段的切换覆盖混为一谈。

## 子片段与代码标签

一个子片段可以只包含放置或回撤阶段，不要求每段都从抓取开始。切换缺失列表描述原始整条轨迹的覆盖范围，不是子片段不合格的判据。

对于 task 1，只有视频确认片段起点已经拿住物品，才可视为放置子片段；夹爪闭合本身不能证明物品在手。现有 build_v3_gripper_code_fixed.py 的 label_episode/code_for_event 支持初始闭合、随后检测到打开事件的片段，生成以下放置标签，并不需要补造抓取事件：

```python
def policy_chunk(obs, state, robot, skills):
    return robot.place('pink cake', 'air fryer', side='right')
```

重新生成代码时应保留真实子片段边界，不跨被剔除的帧拼接，也不把没有观测到的抓取阶段写入代码。

## 全量计划与导出

复核样本后，去掉 `--episodes` 生成全量计划：

```bash
PLAN_DIR="$PWD/trajectory_translation_all_v2_1"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/curate_custom_trajectories.py" plan \
  --root "$ROOT" --out "$PLAN_DIR"
```

确认每个任务和关键阶段的保留情况后，在新根目录导出：

```bash
CLEAN_ROOT="${ROOT}_translation_v2"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/curate_custom_trajectories.py" export \
  --root "$ROOT" --plan "$PLAN_DIR/segments.json" \
  --video-mode symlink --out "$CLEAN_ROOT"
```

导出不会跨异常间隙拼接。每个连续片段成为独立 episode，重新建立全局索引；每路视频起点为原始起点加 start_frame/fps。symlink 复用原始 MP4，原始视频必须保留；copy 复制实际用到的 MP4，生成视频副本。两种模式均不重新编码。

片段内 state 数值保持原样，timestamp 减去首帧时间而保留时间差。导出要求完整源 episode 严格满足 action[t]=state[t+1][14:34]，源末帧为保持；其他记录约定会被拒绝。片段非末帧 action 保持原样，新末帧的完整 20 维 action 改为当前 state 的 14:34，避免目标指向段外或被剔除状态。这个终止保持规则会改变末帧动作，包括其姿态及夹爪标签，记录在 curation_source_map.json 中，需结合训练效果评估；它不属于平移异常判据。

导出前验证数据与元数据哈希，输出到临时目录，验证后发布；拒绝覆盖原始数据或已有目录。v2.1 兼容已有 v2 计划并按计划中的原区间导出，报告记录 input_plan_version。若调整最短长度等筛选参数，需另建目录重新生成计划；v1 计划与配置仍不适用。

新索引需要重建 code 和按帧/episode 索引的缓存；生成代码继续使用已确认的无注释流程。归一化统计按新根目录重新计算：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/diagnose_custom_data.py" stats \
  --root "$CLEAN_ROOT" --aggregation project --stride 1 --padding repeat \
  --out "$CLEAN_ROOT/dataset_stats_eef64.json"
```

训练/验证应按 curation_source_map.json 的原始 source_episode 分组，避免同一原始轨迹的片段泄漏。完整数据、真实视频和训练效果仍需在服务器复核。

## 验证

```bash
PYTHONPATH="$PWD/src" "$PY" -m unittest discover \
  -s "$TOOL_DIR/tests" -p test_paper_trajectory_curation.py -v
```

包括平移联合判据、正常启停峰值、旋转变化不影响筛选、静止首尾裁剪、中间停顿、短片段静止状态未知、最短长度独立生效、task 2 豁免、非均匀时间拟合、异常切段、三相机视频起点与实际 loader 解码、v2 计划兼容、数据源不变、末帧保持、过期计划拒绝及失败回收。
