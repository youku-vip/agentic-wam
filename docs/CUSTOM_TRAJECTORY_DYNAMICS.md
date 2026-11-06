# 专家轨迹的速度、加速度与跳变检查

检查对象是原始 Parquet 中的完整 episode，不把训练 chunk 拼接起来求差分，也不对 code 中的稀疏目标坐标求导。custom 数据的两类轨迹要分别看：

不直接对归一化的 64D 相对动作求导：每个 chunk 的参考位姿会变化，直接拼接会制造假跳变，应回到原始绝对位姿。

- `observation.state[34]`：前 14 维为左右关节，14:23/23:32 为左右实际 EEF 位姿，32/33 为夹爪。
- `action[20]`：0:9/9:18 为左右绝对 EEF 目标，18/19 为夹爪目标。目标序列的导数表示指令变化，不能直接当作机器人真实速度。

位置按米处理。旋转 6D 采用项目的前两列约定，还原旋转矩阵后计算 `log(R_next @ R_current.T)`，得到固定参考坐标系中的最短旋转增量；不会将 `179° → -179°` 当成 `358°` 的转动。关节单位默认标为 stored，未确认单位前不标成 rad 或 deg；可用 `--joint-unit rad` 或 `--joint-unit deg` 声明。关节值不自动 unwrap，以免掩盖编码器或 IK 分支问题。

速度不连续需看速度**向量**之差，不仅看速度大小之差。例如 `+0.1 m/s → -0.1 m/s` 两边速率相同，但速度变化为 `0.2 m/s`。此外检查加速度、jerk（加速度变化率）和对应角运动指标。

## 获取工具

脚本只读取元数据、位姿和时间列，不解码视频、不加载模型、不占训练 GPU，也不会平滑、修复或删除原始数据。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
git fetch origin codex/custom-data-review-files
git show FETCH_HEAD:scripts/audit_custom_trajectories.py > /tmp/audit_custom_trajectories.py
```

也可下载 `deliveries/custom-trajectory-audit-v1.zip` 后解压。在项目根目录设置 `PYTHONPATH=src`，调用解压后的 scripts/audit_custom_trajectories.py；工具使用项目现有 V3EpisodeReader。核心依赖 NumPy、Pandas、PyArrow，不要求 SciPy。

## 先检查四种任务的样本

```bash
DATA_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src "$PY" /tmp/audit_custom_trajectories.py \
  --root "$DATA_ROOT" --episodes 0 1087 2174 3261 \
  --out ./trajectory_audit_sample_v1
```

全量检查时去掉 `--episodes`，并使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src "$PY" /tmp/audit_custom_trajectories.py \
  --root "$DATA_ROOT" --out ./trajectory_audit_all_v1
```

共享 Parquet 的选定列每文件读取一次，不重复为每个 episode 扫描同一文件。全部数值分析在 CPU 上执行。输出目录已存在时拒绝覆盖，请更换 `--out`。

## 输出与判断

| 文件 | 作用 |
|---|---|
| `events.csv` | 每个指标最多 30 个重点事件，包含 episode、task、左右侧、state/action/state_joints、数值、单位、支持帧范围和夹爪开合附近标记 |
| `episode_metrics.csv` | 每段轨迹各指标的最大值、任务和夹爪切换次数，可自行排序筛选 |
| `summary.json` | 分任务、source、左右侧的中位数、P95/P99、最大值、候选统计阈值，以及时间戳/位姿有效性问题 |

主要指标为 `speed`、`velocity_change`、`acceleration`、`jerk`、`angular_speed`、`angular_acceleration`、`angular_jerk`。末端指标单位分别为 m/s、m/s²、m/s³ 和 deg/s、deg/s²、deg/s³。`position_step` 和 `rotation_step` 是相邻样本增量。joint 指标取七个关节中绝对变化最大的一个，事件给出原始 state 的 `joint_index`（0–13）。

按以下顺序复核：

1. 先看 summary 的时间戳、无效位姿和接近 180° 的旋转增量。错误或极短的 dt 会制造假加速度；很长的采样间隔无法解析其中的峰值。
2. 看 events 的 `source`。只有 action 突变而 state 平稳，优先核对标签时序、指令插值和控制器；两者都突变，需要结合原始视频、位置记录和控制日志判断。
3. 检查 `frame_start..frame_end` 附近的连续运动，而不是只看一个最大值。速度至少涉及两帧、加速度涉及三帧、jerk 涉及四帧；事件帧号标在支持窗口的末帧。
4. 结合任务与夹爪事件判断。抓取接触、启停、换向可能产生正常峰值；二值夹爪开关本身不作为末端速度异常。
5. 与机器人实际允许的末端/关节速度、加速度等限值比较。没有这些规格时，报告只能给出需要复核的候选点，不能自动认定演示不合格。

回看视频应使用事件的帧号与 episode 视频偏移（项目 V3EpisodeReader.video_timestamps 可换算编码时间），不要直接把这里的采集 timestamp 当成共享 MP4 的播放时间。

默认统计候选阈值为各任务/source/侧/指标的 `median + 6 × 1.4826 × MAD`；MAD 近零时退到该组 P99.5，并忽略极小浮点误差。候选阈值来自当前数据，可能包含正常峰值，也可能吸收常见问题。`--mad-k` 可调整系数，`--top-k` 可调整每个指标的导出上限。统计计数为“指标点”，同一帧可能由多个指标重复计数，不等于异常帧或失败 episode 数。

提供真实机器人上限时，可用 `--limits /path/to/robot_limits.json`。文件是指标名到正数的 JSON 字典，例如 speed、acceleration、jerk 等，数值单位必须与报告一致；默认没有编造任何机器人限值。`limit_exceeded` 表示超过所提供的上限，仍需按 state/action 的不同含义解释。

## 画曲线复核

绘图额外需要 Matplotlib：

```bash
"$PY" -m pip install matplotlib
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src "$PY" /tmp/audit_custom_trajectories.py \
  --root "$DATA_ROOT" --episodes 0 1087 2174 3261 \
  --plot-episodes 0 1087 2174 3261 --out ./trajectory_audit_plots_v1
```

每个 episode 输出一张独立 PNG，左右臂分别一列，显示平移与旋转的速度、加速度、jerk；state 与 action 叠加，灰线为夹爪切换。也可以先全量扫描，再将 events 中的可疑 episode 放入 `--episodes` 和 `--plot-episodes`。

## 时间和差分约定

默认使用 Parquet 的 `timestamp`，假设它是采集时刻、单位秒。需要确认这列没有在导出过程中改成播放时间。项目的视频编码 fps 可能与原始采集时间不同，不直接用 MP4 的 10 fps 代替。

非均匀时间下，速度为相邻位移除以对应 dt；加速度采用 `2*(v_next-v_prev)/(t[i+2]-t[i])`；jerk 使用三阶差商，常加速度/常 jerk 多项式在不均匀采样测试中可正确还原。旋转导数是 SO(3) 有限差分近似；超过 180° 的帧间真实旋转无法只靠两帧姿态唯一恢复。

默认不跨 episode 求导。重复、倒退、非有限或 `dt<=1e-4s` 的相邻时间不求导，也不跨这些断点计算高阶导数；`dt>5×本 episode 正时间间隔中位数` 的大间隙同样单独报告、跳过。可用 `--gap-factor 0` 关闭大间隙屏蔽，适合已经确认的合法不均匀采样。

缺少 timestamp 时不会自动假定视频 fps。只有确认了物理采样率后，才使用 `--time-source fps --fps 真实采样率`，报告会记录该假设。

数值微分会放大量化和测量噪声，尤其 jerk。原始轨迹不是连续时间测量，稀疏采样无法证明采样间的真实运动始终平滑。本工具不修改专家数据；建议先核对事件及机器人控制方式，再考虑针对性重采样、标注或平滑。

## 验证范围

本地使用合成 v3 打包 Parquet 验证恒速、恒加速度、非均匀采样的常 jerk、方向反转、SO(3) 跨 ±180°、坏时间戳、大间隙、退化旋转、关节编码器跨界、夹爪切换、state/action 区分、单帧 episode、分任务统计、读取顺序、原始文件不变和 PNG 绘制。没有用户真实数据或机器人规格，不能据此报告真实数据已有跳变或确定根因。

本地 18 项测试通过，包含非均匀时间下的平移/旋转常 jerk 和真实 PNG 输出。
