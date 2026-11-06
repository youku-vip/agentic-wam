# 按 Open-HumanScale 的思路筛选 custom 动作轨迹

依据：[Open-HumanScale，arXiv:2611.06817v1](https://arxiv.org/pdf/2611.06817)，第 2.2 节，第 6–7 页。工具为 `scripts/curate_custom_trajectories.py`，版本 `custom-paper-curation-v1`。

论文针对人类第一视角数据的估计手部轨迹，描述的是“检测异常、剔除、切成连续片段”。原始与平滑位置的差用于检测；论文没有说要把平滑后的整条轨迹当作新的动作标签。

同时核对了作者公开分支的 [datasets/config 源码（5ca7629）](https://github.com/DAGroup-PKU/HumanNet/tree/5ca7629a413218cb0e5b2f34e866c82cf28f8d11/humanwam_vanilla)，这些文件未提供可直接复用的轨迹清洗平滑器和数值阈值，因此本工具明确标为论文思路的适配实现。

## 原文与本项目的适配范围

| 内容 | 论文描述 | 本工具 |
|---|---|---|
| 平移稳定性 | 原始/平滑位置残差、加速度、jerk、相邻位置变化 | 检查原始 state 和 action 的左右 EEF |
| 姿态稳定性 | 相邻朝向变化 | rot6D 前两列还原 SO(3)，计算最短旋转角，不直接差分欧拉角 |
| 异常处理 | 删除异常帧，再保留连续片段 | 保守剔除超阈值指标涉及的全部支持帧，片段之间不拼接 |
| 静止筛选 | 32 帧滑窗，裁剪首尾静止段，活跃窗口占比至少 70% | 可选；机器人旋钮转动、接触与等待可能只有很小位移，因此默认关闭 |
| 手部可见性/相机运动 | 可见手才参与监督；单独检查相机位姿 | 未实现：现有 custom34/action20 没有这些标定、可见性及动态相机数据 |
| 平滑器与数值阈值 | 本文未列出具体算法、窗口与阈值 | 使用可配置的时间感知局部二次拟合，仅计算残差；所有阈值明确标为本地示例 |

本工具保留现有 custom34/action20 和训练时的 64 维转换。论文第 2.3 节的 16 维表示、第 5.2 节的 49 维表示及相机坐标变换涉及另一套模型接口，未移植到当前检查点。没有相机外参时不能把机器人坐标直接改名为相机坐标。

非连续的完整 34 维重复状态另行报告，不因为数值重复就自动删除某次记录。重复可能出现在正常停顿或返回位置中，无法只凭相等关系认定哪一帧有误。

## 获取与运行

在已有本项目和 v3 loader 的环境中运行。依赖 NumPy、Pandas、PyArrow；不加载模型、训练 GPU 或 VLM。导出复用原始视频，分析阶段不解码视频。

可在仓库直接运行脚本，也可将 `deliveries/custom-paper-trajectory-curation-v1.zip` 解压到独立工具目录。以下按解压后的目录演示，保留服务器当前项目配置：

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
TOOL_DIR="$PWD/custom_paper_curation_v1"

git fetch origin codex/custom-data-review-files
git show FETCH_HEAD:deliveries/custom-paper-trajectory-curation-v1.zip \
  > /tmp/custom-paper-trajectory-curation-v1.zip
unzip -n /tmp/custom-paper-trajectory-curation-v1.zip -d "$TOOL_DIR"
```

第一步只生成候选计划。输出目录必须是新目录，并位于原始数据根目录之外：

```bash
PLAN_DIR="$PWD/trajectory_curation_sample_v1"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/curate_custom_trajectories.py" plan \
  --root "$ROOT" --episodes 655 2970 247 3075 \
  --out "$PLAN_DIR"
```

这几个 episode 用于复核已发现的问题，不能代表四个任务的完整分布。四任务基础样本可以选 `0 1 2 3261`，任务仍以 `task_index` 为准。

工具也支持已有的少量原始数值导出：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/curate_custom_trajectories.py" plan \
  --raw-json ./trajectory_focus_raw_v2/raw_episodes.json \
  --out ./trajectory_curation_raw_preview_v1
```

raw JSON 计划用于数值对比；缺少完整元数据和视频，不能直接导出完整 LeRobot 数据集。只有验证了下一帧动作约定，才额外生成 `curated_raw_episodes.json`。

## 如何选参数

示例 `configs/trajectory_curation.example.json` 使用以下值。**这些不是论文原参数，也不是已经确认的机器人物理上限。正式筛选前应结合正常轨迹、接触阶段和视频调整。**

| 参数 | 示例 | 含义 |
|---|---:|---|
| `smoothing.window_frames` | 7 | 局部拟合窗口，奇数 |
| `smoothing.polynomial_degree` | 2 | 局部多项式次数；使用实际 timestamp 的相对时间 |
| `position_residual` | 0.025 m | 原始位置到局部拟合位置的距离 |
| `position_step` | 0.08 m | 相邻帧位移 |
| `rotation_step` | 25° | 相邻帧最短旋转角 |
| `acceleration` | 8 m/s² | 原始位置的二阶差分范数 |
| `jerk` | 150 m/s³ | 原始位置的三阶差分范数 |
| `min_segment_frames` | 33 | 切段后保留的最短长度，配合当前 33 帧观察窗口 |

单项阈值可设为 `null` 禁用；至少启用一项。复制示例后用 `--config /path/to/your_thresholds.json` 指定，生成的计划会保存实际参数。

超阈值条件按 OR 合并。位移和转角涉及 2 帧，加速度涉及 3 帧，jerk 涉及 4 帧；本工具剔除整个支持范围，不能将结果理解成对单个坏帧的精确定位。残差可能使邻近正常帧也被标记，适合保守筛选。

差分不跨 episode、坏时间间隔或大间隙；局部拟合也不跨坏时间/无效位置。时间异常规则沿用现有审计脚本：`dt<=1e-4 s` 或超过中位正间隔 5 倍。时间戳若只是导出时生成的 `frame_index/10`，计算得到的是该时间假设下的指标，不能据此确认真实采集加速度。

需要尝试论文的静止筛选时，增加 `--idle-filter`。本实现将 32 帧内的 XYZ 各维极差合成为位置变化量，左右臂取最大值，阈值示例为 5 mm；这是对论文未明确给出的“位置变化”定义的适配。先裁剪首尾无活跃窗口部分，再检查剩余窗口的活跃占比。32 帧在 10 Hz 下约 3.2 秒，不能直接当作论文数据的相同时长。纯旋转、等待、保持接触或夹爪操作可能被位置规则误删，应先检查任务 2 的保留比例。

## 查看计划

| 文件 | 用途 |
|---|---|
| `summary.json` | 分任务保留比例、稳定性剔除帧数、其他裁剪帧数、片段数 |
| `events.csv` | 每个超阈值指标及其完整支持帧范围；一个位置可对应多项事件 |
| `frame_flags.csv` | 去重后的稳定性候选帧及全部原因 |
| `segments.json` | 原始 episode 的半开区间 `[start_frame,end_frame)`、数据哈希、参数及丢弃原因 |

`stability_rejected_frames` 是去重后的候选帧数，`other_dropped_frames` 还包括切分后过短的片段及可选静止裁剪。两者相加才是总减少量。不能将这些统计当作失败演示数量。

`episodes_with_missing_gripper_transitions` 列出切分后丢失夹爪切换的源 episode；逐段详情见 `gripper_transition_coverage`。只有同一保留片段同时包含切换前后两帧，才记作保留切换。用于机器人后训练时，应重点复核这些片段是否缺少释放或抓取阶段，必要时调整阈值或排除整个 episode，不自动补造夹爪动作。这个指标检查的是记录覆盖范围，不证明物理接触或任务成功。

复核应包括：每个任务是否仍有足够数据；抓取、接触、释放是否被保留；动作是否与视频对应；任务是否在切分后缺少关键阶段。少量异常样本的结果不能作为全量阈值调优的唯一依据。

## 全量计划与导出

阈值复核后，去掉 `--episodes`，为全数据生成新计划：

```bash
PLAN_DIR="$PWD/trajectory_curation_all_v1"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/curate_custom_trajectories.py" plan \
  --root "$ROOT" --out "$PLAN_DIR"
```

按计划导出新根目录，不修改原始目录：

```bash
CLEAN_ROOT="${ROOT}_curated_v1"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/curate_custom_trajectories.py" export \
  --root "$ROOT" --plan "$PLAN_DIR/segments.json" \
  --video-mode symlink --out "$CLEAN_ROOT"
```

默认 `symlink` 模式链接原始 MP4，因此必须保留原始视频目录。`--video-mode copy` 复制实际用到的共享 MP4，生成独立视频副本；复制的是原文件，不重新编码。

每个保留片段成为一个新的 episode：

- 新 `episode_index/frame_index/index` 连续，task_index 保留原值。
- 段内状态、关节、位姿及夹爪数值保持原样；timestamp 仅减去该段首帧时间，保持时间差。
- 视频起点为 `原 episode 视频起点 + start_frame/fps`，每路相机分别计算；不会用采集 timestamp 直接索引共享 MP4。
- 导出要求原始动作严格满足 `action[t]=state[t+1][14:34]`，原 episode 最后一行也为保持动作。其他动作记录约定会被拒绝，避免错误改写。
- 对切段的新末帧，将 action 改为该帧的 `state[14:34]`，形成明确的终止保持，避免末帧动作仍指向被剔除或段外状态。它是本项目适配规则，会增加停止标签，需结合实际训练评估；不把它描述成论文原算法。
- `curation_source_map.json` 记录新旧 episode/帧区间及末帧动作是否变化。

导出前校验计划中的原始状态、动作、时间、任务与视频元数据哈希；导出后由实际 V3EpisodeReader 校验新枚举和下一帧动作约定。导出使用临时目录，验证通过后发布到目标目录；失败不会留下可误用的目标数据集。已有目标目录拒绝覆盖。

## 训练前必须同步的文件

导出不会把旧的 code、归一化统计、instruction_idx、cycle 或其他平面索引缓存复制到新根目录。直接沿用旧缓存会与新的全局帧索引错配。按新根目录重新生成这些缓存；生成代码时继续使用你已核对的 coding_v4 参数及无注释流程。

先重新计算当前项目使用的相对动作统计，采用已验证的 project 聚合口径：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  "$TOOL_DIR/scripts/diagnose_custom_data.py" stats \
  --root "$CLEAN_ROOT" --aggregation project --stride 1 --padding repeat \
  --out "$CLEAN_ROOT/dataset_stats_eef64.json"
```

原本正确的 `dataset_stats_eef64.json` 不需要被覆盖；这里统计的是经过筛选的新数据分布。由于原项目按 episode 等权聚合，切分成多个片段也会改变权重。

如果划分训练集与验证集，应按 `curation_source_map.json` 中的 `source_episode` 分组，避免同一原始轨迹的不同片段分别进入训练和验证；同时按照各自 loader 的实际枚举准备代码缓存。当前配置的 `val_set_proportion: 0.0` 会让两边读取同一数据，不能当作独立泛化评测。

全量数值/视频复核完成后，再将训练配置中的数据根目录、统计路径、代码缓存及其他缓存路径指向新生成文件。静态任务文字到提示词的映射可以沿用，但按帧或 episode 编号索引的文件必须重建。模型效果应通过相同独立验证集和实际任务成功率比较，不以筛后峰值变低作为性能提升的证明。

## 验证

```bash
PYTHONPATH="$PWD/src" "$PY" -m unittest discover \
  -s "$TOOL_DIR/tests" -p test_paper_trajectory_curation.py -v
```

覆盖恒速、非均匀时间局部拟合、SO(3) 跨 ±180°、尖峰支持范围、完整状态重复、二值夹爪、切段丢失释放事件、坏时间、静止规则对纯旋转的影响、末帧保持、新旧动作对应、源文件哈希不变、过期计划拒绝和导出失败回收。合成三相机共享 MP4 的集成检查使用各自不同的视频起点，通过实际 v3 loader 解码片段首尾帧，确认视频像素一致且采样不会跨片段边界。验证范围为本项目 v3 loader，未验证其他 LeRobot 版本、完整真实数据集或 GPU 训练效果。
