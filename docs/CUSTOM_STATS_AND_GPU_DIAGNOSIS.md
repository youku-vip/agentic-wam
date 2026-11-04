# Custom相对动作统计与四卡ZeRO-2诊断

本次代码标签去注释/回撤合并没有改变原始state/action或转换器，因此没有自动重算统计。之前提供的dataset_stats_eef64.json已验证64D形状、有限值、hand零槽、归一化初始化，但没有用原始Parquet复算，不足以证明统计来自正确的相对动作。

## 新旧统计差异：先匹配原项目聚合口径

原项目BaseLerobotDataset.get_dataset_stats有明确的聚合约定：先算每个episode的统计，再对episode均值等权平均；标准差合并episode内的无偏方差和episode均值间方差。q01取各episode q01的最小值，q99取最大值，是覆盖各轨迹的边界，不是全局真正的1%/99%分位数。

本工具最初默认pooled策略：每帧/窗口等权，分位数按合并分布估计。两者的均值、标准差和分位数可以不同，即使min/max完全一致。这种差异不能直接证明旧统计错误，也不能归咎于相对动作变换。夹爪保持绝对值，夹爪均值不同尤其提示加权策略差异。

要严格验证原项目口径，使用更新脚本的--aggregation project：

```bash
PY=/usr/local/bin/python3.11
ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
"$PY" scripts/diagnose_custom_data.py stats --root "$ROOT" \
  --existing "$ROOT/dataset_stats_eef64.json" \
  --aggregation project --stride 1 --padding repeat \
  --out "$ROOT/dataset_stats_eef64.project_policy_check.json"
```

project模式不做分位数reservoir抽样，--max-samples在此模式不使用。它复现原项目计算定义，但使用float64聚合以减少误差，因此与原float32结果可能有约1e-6量级差异。模式需要每条轨迹至少2帧、stride=1和repeat padding。

如果同口径比较各字段差异降至浮点误差量级，就支持旧统计来自该原项目策略，可继续保留原文件；若仍有明显差异，再检查旧统计生成程序、选择的episode、转换器和输入版本。不要在训练中途仅因pooled统计不同而替换文件，checkpoint的训练/部署应使用一致的归一化。

## 相对动作的定义

CustomUnifiedEEFTransform以动作块首帧state[t]为基准：

```
delta_position[k] = R_base.T @ (action_position[t+k] - state_position[t])
delta_rotation[k] = R_base.T @ action_rotation[t+k]
```

两只手分别计算，夹爪保持绝对目标。所有32步共用首帧基准，不是每步减前一步。统计必须在该变换后的64D空间计算；不能使用raw20D或绝对64D统计。

## 在数据机器复算

命令在仓库根目录执行，原文件不覆盖：

```bash
PY=/usr/local/bin/python3.11
ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30

# 先选两条验证工具与真实数据兼容；该输出仅供检查，不用于全量训练
"$PY" scripts/diagnose_custom_data.py stats --root "$ROOT" \
  --existing "$ROOT/dataset_stats_eef64.json" --episodes 0 1 \
  --stride 1 --padding repeat --out "$ROOT/stats_relative_sample_check.json"

# 全量复算。32步动作窗口不跨episode，末端按训练默认重复最后一帧
"$PY" scripts/diagnose_custom_data.py stats --root "$ROOT" \
  --existing "$ROOT/dataset_stats_eef64.json" \
  --stride 1 --padding repeat --max-samples 20000 \
  --out "$ROOT/dataset_stats_eef64.relative_recomputed.json"
```

要求：torch/numpy/pandas/pyarrow及仓库src；不需要视频、模型权重或GPU。输出路径已存在时会拒绝覆盖。

复算使用原Custom转换器和项目PooledStats聚合：对扫描值完整计算min/max/mean/std；q01/q99默认从固定seed的最多20000个state帧/动作窗口做reservoir估计。--max-samples 0会保留全部值做精确分位数，285262个32×64动作窗口仅float32数组就约2.34GB，且分位数有额外内存开销，需确认主存余量。只选部分episodes的结果不能替代全量统计。

若已有统计明确只用完整32步窗口，选择--padding valid做同口径对比；若训练用了episode子集、不同global_sample_stride或采样权重，需要相同口径，本工具当前按给定episode的单位帧间隔构造窗口，不自动推断这些运行时修改。--stride改变窗口起点采样，改变统计权重，不能混用作严格等值验证。

输出JSON旁有.report.json：

- max_xyz_gripper_roundtrip_error / max_rotation_matrix_roundtrip_error应很小，超过1e-4直接失败。旋转比较重正交化后的矩阵，避免把原始6D噪声误判为逆变换失败。
- max_absolute_stat_difference比较新旧各统计字段。差异不自动判错：原始生成程序的padding/stride/分位数采样必须相同才能严格比较。
- existing_action_preclip_outside_5_fraction表示现有统计归一化后、clamp前超出±5的有效动作坐标比例；比例明显偏高需检查旧统计、异常数据或坐标约定。正常超出±1可能来自分位数尾部，不等同于错误。
- 新统计记录窗口策略、episode列表、transform源码摘要，便于追溯。

确认新统计口径后，训练可用覆盖参数：

```
data.train.pretrained_norm_stats=<新统计文件绝对路径>
data.val.pretrained_norm_stats=<同一新统计文件绝对路径>
```

未复核前不要替换原统计，也不要在同一次性能对比里同时切换统计和多个训练参数。

## 低GPU利用率：代码中确认的排查线索

1. BaseLerobotDataset给三路相机各请求33帧，v3 loader进行视频解码后，RobotVideoDataset才下采样成各9帧。torchcodec/PyAV请求会新建解码器并seek。AV1随机窗口可能受CPU解码、共享存储与GOP长度限制，稀疏请求也不意味着实际解码工作量线性减少。
2. 每个进程8个DataLoader worker、prefetch_factor=4。四卡是32个worker，叠加解码线程可能超过容器CPU配额；预取/pinned memory也占用大量内存。增加worker不一定更快。
3. 默认ZeRO-2没有CPU optimizer/param offload；不能仅因使用ZeRO-2就归因于CPU卸载。它增加的通信、GPU互联、rank快慢差异，需要现场时间数据。
4. YAML里num_processes=1会被launch_train.py的--nproc覆盖。若直接运行accelerate而未覆盖，则可能实际没有启动4个rank。需看启动命令和四卡进程，而不能仅看机器有4卡。
5. 已核对仓库train.sh：--nproc 4、--zero 2、batch_size=6、gradient_accumulation_steps=4、num_workers=8、bf16，全局batch=96。它设置AGENTICWAM_PHASE_TIMING=0，关闭了阶段计时。AGENTICWAM_MOT_NO_CKPT_EVERY=1关闭混合attention的checkpoint，但model.use_gradient_checkpointing=true仍影响其他专家模块，不能说所有checkpoint已关闭。
6. 每步仍包含冻结VAE等前向计算；“冻结”不等于没有计算开销。train.sh还启用save_optimizer_state=true、save_every=500和在线W&B，前者会周期性写较大文件并等待各rank，不应和持续性低利用率混为一谈。脚本没有覆盖data.code_cache，仍读取custom.yaml默认coding_v3；若要测试新code，需要显式指定路径。

发现并修正一处计时口径问题：原trainer的data累计所有microstep，而分母n仅累计抽样计时microstep，data_ms可能被放大约log_every倍。本分支使data也仅在抽样步累计；这只是计时修正，不代表训练已加速。既有性能日志须考虑这个偏差。另已修正samples/s漏乘gradient_accumulation_steps的问题；train.sh累积4时原日志低报约4倍，这不意味着GPU物理利用率也低了4倍。

## 现场采集

训练运行时另开终端采集30秒GPU信息，不需要停止训练：

```bash
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit --format=csv
nvidia-smi topo -m
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/memory.max
timeout 30s nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv -l 1 > gpu_30s.csv
```

timeout到时返回124属正常。另提供启动命令、约30行包含speed和phase(ms/microstep)的日志，不提供环境变量全集或密钥。trainer默认开启计时，但train.sh显式关闭了它；需要在train.sh内部将export AGENTICWAM_PHASE_TIMING=0改成1，再启动新一段可比较运行（仅在外面export会被脚本内的0覆盖）。日志包含data/skew/fwd/bwd/opt；对比吞吐时保持设置一致。

## 独立测数据读取

训练暂停后做更干净的对比；不建议与4卡训练同时跑，否则会竞争CPU/存储。

```bash
"$PY" scripts/diagnose_custom_data.py loader --root "$ROOT" --workers 0 --batch-size 2 --batches 16
"$PY" scripts/diagnose_custom_data.py loader --root "$ROOT" --workers 4 --batch-size 2 --batches 16
"$PY" scripts/diagnose_custom_data.py loader --root "$ROOT" --workers 8 --batch-size 2 --batches 16

# 只请求最终需要的9帧，作局部瓶颈对比，不修改实际训练逻辑
"$PY" scripts/diagnose_custom_data.py loader --root "$ROOT" --workers 4 --batch-size 2 --batches 16 --sparse-video
```

原始loader基准返回float32原分辨率33帧，比训练最终送入模型的张量更大，batch和worker应从小开始。此基准排除processor/文本cache/H2D/模型/分布式，不等同于端到端训练速度；单进程结果不能直接乘4推断训练能力。默认强制pyav用于可控对比，可用--backend torchcodec另测并检查fallback日志。

如果data等待明显、解码CPU饱和且增加worker无益，优先考虑仅解码所需帧、复用解码器、本地缓存/更适合随机访问的编码；若skew/bwd高、data低，再看rank负载、NVLink/PCIe及NCCL。若fwd高，再分析VAE、vision tower和模型kernel；必要时用PyTorch profiler，不能仅凭nvidia-smi一个百分比定因。

仍需GPU具体型号/显存、运行日志、GPU指标和CPU配额。当前环境无4卡GPU且无真实数据，本次不能给出实际根因排名或复算后的真实统计文件。工具已用带大绝对坐标、90度基座旋转、packed双episode的合成数据验证相对统计与边界，并实际运行视频loader基准测试。
