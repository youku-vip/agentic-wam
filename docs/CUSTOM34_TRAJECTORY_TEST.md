# Custom34 多阶段测试与逐维轨迹对比

入口为 deploy-custom/evaluate_custom34_trajectories.py。run 用原训练 loader、checkpoint 和统计文件做离线自由生成，输出一段预测与对应专家动作。plot 可仅从已有 chunks.npz 画图，不加载模型或视频。整个流程不启动 HTTP 服务，不执行生成的 Python，不训练模型，也不修改数据、统计值或权重。

## 手动安装

备份后按相同相对路径复制这三个文件：

~~~text
deploy-custom/evaluate_custom34_trajectories.py
deploy-custom/evaluate_custom34_checkpoint.py
src/agenticwam/utils/custom34_review.py
~~~

配套 evaluator 为 v1.4，新增逐案例计划和采样时间信息。原来的 --episodes/--start-frames/--plots 命令继续可用。训练端 RoPE、显存修复和 batch=2 文件保持原样。

## 先测一个完整 32 步窗口

~~~bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
RUN="$PWD/runs/custom_code_repair_20261110_024220"
OUT="$PWD/trajectory_one_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/src" "$PY" \
  deploy-custom/evaluate_custom34_trajectories.py run \
  --train-config "$RUN/recipe/config.yaml" \
  --checkpoint "$RUN/checkpoints/weights/step_000300.pt" \
  --stats "$RUN/normalization_reference.json" \
  --episodes 2 --frames 5 \
  --seeds 42 --inference-steps 10 --canvas-tokens 128 \
  --padding error --device cuda:0 --out "$OUT"
~~~

CUDA_VISIBLE_DEVICES=1 将物理第二张卡映射为进程内 cuda:0。此例只做一个预测，与 episode 2 的 action[5:37] 对照（当前数据 stride=1）。--padding error 要求完整真实 32 步，不足时直接报错。

打开 OUT/index.html，或查看：

~~~text
OUT/plots/episode000002_frame000005_c128_n10_s42/
  index.html
  action20_all_dimensions.png
  dimensions20/00_left_x_m.png ... 19_right_gripper.png
  physical_errors.png
  action20_comparison.csv
  per_dimension_metrics.csv
  metrics.json
  chunks.npz
  target.py
  prediction.py
~~~

蓝线是专家，橙线是模型。一个 5×4 总图包含全部执行维度，另有 20 张单维图，CSV 保存每步预测、专家值、误差、时间及有效标记。原始模型输出、图像输入与代码指标还保存在 OUT/evaluation/对应案例目录。

## 维度与对齐

| 维度 | 含义 | 单位 |
|---|---|---|
| 0–2 | 左臂绝对 XYZ | m |
| 3–8 | 左臂 rotation6D 两列 | 无量纲 |
| 9–11 | 右臂绝对 XYZ | m |
| 12–17 | 右臂 rotation6D 两列 | 无量纲 |
| 18、19 | 左、右夹爪 | 数据集原来的绝对夹爪值 |

专家取自原始 action 字段，而非未来 observation.state。预测使用训练 normalizer 逆变换一次，再以同一个 observation.state[t] 为基准将局部相对 EEF64 还原为绝对 EEF20。对应关系为 action[t + k×stride]，不额外平移一帧，不做时间重排或动态时间匹配。

横轴用真实有效采样率 fps/stride，当前 stride=1、10 Hz 时，32 步覆盖 0 到 3.1 秒。physical_errors.png 另外给出左右臂位置误差（mm）、SO(3) 姿态角误差（度）与夹爪曲线。rotation6D 分量自身不是角度。

在 run 或 plot 后加 --relative64，还会输出反归一化的局部相对 64 维总图和 CSV。44 个未使用的手部占位维度在总图中为灰底；比较以同一个当前观测状态为基准。

## 自动按实际代码阶段选点

~~~bash
OUT_MULTI="$PWD/trajectory_stages_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/src" "$PY" \
  deploy-custom/evaluate_custom34_trajectories.py run \
  --train-config "$RUN/recipe/config.yaml" \
  --checkpoint "$RUN/checkpoints/weights/step_000300.pt" \
  --stats "$RUN/normalization_reference.json" \
  --episodes 3 1 2 3258 --points middle \
  --seeds 42 --inference-steps 10 --canvas-tokens 128 \
  --padding repeat --device cuda:0 --out "$OUT_MULTI"
~~~

省略 --frames 后，工具从实际训练枚举和 code_id_by_idx 中找每个 episode 连续代码区间，默认选各区间中点。同样的 move_arm 若目标不同，仍分成独立区间。可用 --points start middle end 增加阶段内和边界观测点；也可增加 --episodes 中的真实 episode ID。

根据本次上传的 cache/order，以上四个 episode 的中点计划有 8 个窗口：开门 grasp 与 move_arm+release，放物 pick 与 place，旋钮 grasp 与 rotate+release，关门两段不同目标的 move_arm。服务器会以实际训练 loader 重新生成并核对计划。release 已合并在部分代码标签中，此工具不会凭空拆成独立 release 标签。

阶段取样标签写入 cases.json，包含 episode/frame、代码哈希、阶段边界与有效步数；只用于选点和评分，不作为模型输入。模型仍从当前观测、状态及原提示词自由生成。边界来自数据标签，不等同于人工验证过的物理接触阶段。

后续阶段往往不足完整 32 步：--padding repeat 保留模型 32 步输出，但补齐尾部只显示灰色阴影，不连成真实轨迹，也不参与误差统计；页面显示实际 N/32。希望全部窗口至少有真实 32 步，可加 --min-valid-steps 32，这会跳过不满足的阶段并记入 cases.json 的 skipped。

OUT_MULTI/stage_metrics.csv 按任务和标签阶段汇总代码语法、API、API+左右侧序列匹配和 AST 完全匹配。API 序列匹配本身无法区别同为 move_arm 的推进/回撤，也不比较数值与对象；请结合 AST、目标/预测代码和轨迹图判断。32 步动作窗口可能跨越当前代码阶段边界。

## 重用同一计划比较另一个权重

~~~bash
OUT_200="$PWD/trajectory_stages_200_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/src" "$PY" \
  deploy-custom/evaluate_custom34_trajectories.py run \
  --train-config "$RUN/recipe/config.yaml" \
  --checkpoint "$RUN/checkpoints/weights/step_000200.pt" \
  --stats "$RUN/normalization_reference.json" \
  --cases-file "$OUT_MULTI/cases.json" \
  --seeds 42 --inference-steps 10 --canvas-tokens 128 \
  --padding repeat --device cuda:0 --out "$OUT_200"
~~~

此处不重新选点；cache、枚举或目标代码哈希不匹配时会拒绝评分。仍需核对两个报告的输入指纹和推理设置。每次输出需用新目录，避免覆盖已有结果。

## 已有 chunks.npz 单独作图

将 CASE_DIR 改为已有 chunks.npz 所在目录：

~~~bash
CASE_DIR=/path/to/episode000002_frame000005_c128_n10_s42
PLOT_DIR="$PWD/trajectory_replot_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  deploy-custom/evaluate_custom34_trajectories.py plot \
  --case-dir "$CASE_DIR" --fps 10 --out "$PLOT_DIR"
~~~

旧 npz 缺少采样率时需明确传 --fps；若 stride>1，填写有效 fps/stride。新版 evaluator 将采样率写入 npz/reference.json，可以省略 --fps。与已有采样率不一致时会报错。该模式不改变输入文件。

## 验证范围

35 项 CPU 检查通过：阶段选点、实际枚举索引、代码哈希拒绝、恶意文本不执行、FP32/BF16 模型准备、真实 EEF64↔EEF20 与 normalizer、时间轴、padding 排除、20 张逐维图/总图/CSV、完整模拟评测到 HTML 的流程，并回归原 RoPE 和专家代码对照工具。还用用户上传的 code cache 与 episode_order 核对了四个已知评测索引和目标标签。尚未在用户 GPU/新权重上运行本次多阶段推理；不能从 CPU 测试推断模型质量。
