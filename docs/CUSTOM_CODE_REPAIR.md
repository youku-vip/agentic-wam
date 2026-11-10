# Custom34 代码专家修复试验

本包提供可选的训练端 RoPE 保护、固定 MASK 画布与有限 EOS 监督，以及一个只在 CPU 上生成和核对试验配置的工具。GPU 训练由用户执行生成的启动脚本。本环境的检查使用小型 CPU 模型和测试 tokenizer，没有对用户的大模型进行训练，也不预先承诺生成质量改善。

## 手动安装

将运行文件按相同相对路径复制到项目中，先保留原文件副本：

| 文件 | 用途 |
|---|---|
| src/agenticwam/models/experts/code_expert.py | 可选复数缓存保护、MASK 尾部与有限 EOS 标签 |
| src/agenticwam/models/cosmos_mot/build.py | 将修复设置从模型配置传入专家，加载新配置时继续生效 |
| scripts/custom_code_repair.py | CPU 预检查、配置生成和启动脚本生成 |
| deploy-custom/evaluate_custom34_checkpoint.py | 配套自由生成评测，含已验证的精度、reference 和只读状态修复 |

现有 evaluate_custom34_expert_conditioning.py 可保留。它是提供专家代码的诊断工具；本试验的前后测使用自由生成评测。

新字段都有兼容默认值。旧配置默认使用 legacy RoPE 和原掩码行为，不自动改变旧权重的推理。新试验配置显式选择 complex；普通浮点参数继续按 BF16/FP16 配置转换，非持久复数频率缓存保留其复数精度并跟随设备移动。新增配置属性不改变权重参数名称或形状。

## 试验设置

准备工具从已有 run/config.yaml 复制完整结构、数据路径和超参数，只覆盖本次试验设置：

- 从指定 step_008000.pt 作为 foundation 权重初始化；resume=null，优化器、学习率日程和计步从零开始。
- 默认新增 300 步，学习率 1e-5，保留原微批次、梯度累积、模型损失权重与图像预处理。四任务批次平衡显式启用。
- code_rope_mode=complex，设置写入模型配置，后续训练和评测加载该配置时可继续生效。
- 训练与推理使用同样的代码画布。用本地 SDAR tokenizer 扫描全部目标代码，auto 选择至少 128 且足够容纳最长程序加 EOS 的 32 的倍数。
- 固定画布尾部填 MASK。仅监督程序、一个终止 EOS 和最多 8 个额外 EOS，其余尾部标签为 -100，避免把全部尾部 EOS 都计入准确率。
- mask_tmin=0.5、mask_fullp=0.25：一部分训练样本从全 MASK 开始，其余采用较高掩码比例。提示词仍可见。
- 保存新的权重快照，不保存大型优化器状态；保留最近两个新 run 权重快照。

这些数字是本地试验的起点，不是论文作者参数或已经验证的最优设置。训练损失和训练 code_acc 因条件变化不能直接与旧曲线比较，应以自由生成结果为主。

## 准备配置

以下路径对应当前清洗后 custom34 数据集的已验证 run：

~~~bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
RUN=/data/algorithm/agentic-wam/runs/air_fryer_zero2_10000_p012_v4_1
NEW_RUN="$PWD/runs/custom_code_repair_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" scripts/custom_code_repair.py prepare \
  --source-config "$RUN/config.yaml" \
  --checkpoint "$RUN/checkpoints/weights/step_008000.pt" \
  --stats "$RUN/dataset_stats.json" \
  --out "$NEW_RUN" --steps 300 --lr 1e-5 --nproc 2
~~~

准备过程读取小型代码缓存与本地 tokenizer，检查所有非空目标程序的 API 参数、最长目标与提示词预算、复数缓存的父模块类型转换，以及全 MASK 输入几何。不会读取大 checkpoint 张量或加载大模型，不解码视频，也不启动 GPU。

输出包括：

- recipe/config.yaml：固定的试验配置。训练后还会由原 runtime 保存 run 根目录的 config.yaml。
- normalization_reference.json：原 run 统计快照的逐字节副本，训练和两次评测均固定使用它。原统计和数据集保持原样。
- code_repair_audit.json：实际 tokenizer 的长度统计、缓存/统计指纹、修复行为测试和有效 batch 大小。
- run_code_repair.sh：默认模式是 CPU audit，可显式选择 baseline、train、eval。

准备工具拒绝已有输出目录、非法目标程序和画布溢出，不会静默截断程序。若最长提示词加画布超过原 max_code_tokens，会在 GPU 启动前失败并给出原因。

## 基线、训练与复测

先用相同的新配置、原 8000 步权重做自由生成基线：

~~~bash
CUDA_VISIBLE_DEVICES=1 bash "$NEW_RUN/run_code_repair.sh" baseline
~~~

然后用两张卡启动新增 300 步：

~~~bash
CUDA_VISIBLE_DEVICES=0,1 bash "$NEW_RUN/run_code_repair.sh" train
~~~

最后评测新权重：

~~~bash
CUDA_VISIBLE_DEVICES=1 bash "$NEW_RUN/run_code_repair.sh" eval
~~~

默认 eval 使用新 run 的 checkpoints/weights/step_000300.pt；也可传入指定快照作为第二个参数。只读预检查可随时执行：

~~~bash
bash "$NEW_RUN/run_code_repair.sh" audit
~~~

生成脚本使用当前项目的 ZeRO-1 accelerate 配置。它清除会覆盖本次代码配置或把专家代码注入自由评测的环境开关，保留其他性能选项；已有 DeepSpeed graph-count helper 时启用其缓存。原 train.sh 不需要改写或执行。

默认前后测窗口是当前清洗数据集的 episodes 3、1、2、3258，各取 frame 5，种子 42/43/44。其他枚举可在 prepare 时通过 --review-episodes 和 --review-frame 指定。这些是训练内回归窗口，不是独立验证集或实机成功率。

## 如何验收

先核对 preflight 的 passed=true、rope_dtype=torch.complex128 和 full_mask_probe_passed=true，再比较 eval_baseline_* 与 eval_repair_* 内的 summary.json：

- API 合法率是否提高。
- primitive_sequence_match 是否改善；该项只比较 API 名称和左右侧，对象与数字参数还需要结合 target.py、prediction.py、ast_exact_match 和动作误差判断。
- 每个任务的左右臂位置、SO(3) 姿态和夹爪指标是否改善，而非只看总体均值。
- 其他任务是否回退，是否出现空程序、过早 EOS 或长度不足。

本流程不会提供专家程序作为条件，专家标签只用于评分。此前专家代码对照实验的改善不能当作模型的自由生成准确率。新配置在训练端使用完整复数位置编码，是一次新的受控试验；是否继续长程训练以实际前后测结果决定。

## CPU 验证范围

测试覆盖默认旧模式的输入/标签/随机数序列与原始版本一致；父模块 BF16/FP32 转换保留复数缓存；SDAR 工厂在首次转换前接收配置；固定 MASK 画布、EOS 上限、空标签有限损失、溢出拒绝；预检查与配置生成不启动 CUDA、保留原配置与 checkpoint 文件；生成启动脚本通过 Bash 语法检查。实际 BPE 长度与短程微调效果需要在用户本地模型环境测量。
