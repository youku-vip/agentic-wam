# Custom34 / LeRobot v3：使用原项目标注引擎

本适配使用原项目 **Route A**，不是固定五阶段模板：

1. 从真实夹爪信号检测事件，按事件与时间约束切分。
2. 原 VLM 提示词查看片段画面，为片段命名。
3. 原 actor 使用 YAM 双臂夹爪 API 和冻结技能库生成程序。
4. 原静态校验与 gripper reconciliation 检查代码事件。
5. 从原 program.py / lines.json 构建训练平铺缓存。

不会硬编码 task0=开门等映射，而是读取 meta/tasks.parquet。不会把 grasp/release 当成单纯闭合/打开；原技能包含的接近/搬运/回撤由原 actor 按语义选择。

## 适配内容与限制

- 状态从34维读取：左XYZ14:17，右XYZ23:26，夹爪32/33；无需转成YAM的82维，也不修改原始数据。
- v3元数据定位共享Parquet和MP4；使用head相机和episode视频偏移。
- 以info.json中的fps换算原30Hz检测器的平滑、最小运行时长、速度阈值和防抖时长；custom路径使用边缘填充，避免恒开信号被零填充制造事件。原有embodiment的调用默认值保持不变。
- 子任务最短/最长时长仍默认1秒/8秒；**并非每个夹爪事件都一定形成独立片段**，原算法会合并过短边界。必须先检查dry-run报告是否漏掉你关心的事件边界。
- 只有两个事件无法证明接触/任务成功。Task3无夹爪事件时允许原VLM描述运动，但不能以零事件匹配率当作关门成功证明。
- cache阶段要求原gate通过且所有slice静态验证通过；缺结果、错帧数、过期state或sidecar会失败，不补齐、不截断、不用visual_diff伪装缺失监督。
- API、提示词、技能库原样复用。actor可能使用原允许的高层技能或基本动作；不强制统一四步或五步。
- 未在用户真实数据及真实VLM上执行标注；离线测试使用合成数据，不证明真实标签质量。

## 1. 环境

在项目根目录执行。使用你已有Python，安装原引擎依赖并保留项目兼容的NumPy版本：

```bash
PY=/usr/local/bin/python3.11
"$PY" -m pip install -r codegen/pretrain/pipeline/requirements.txt av torch \
  "numpy==1.26.4" "opencv-python-headless==4.11.0.86"
```

如果在项目固定numpy==1.26.4的环境中，OpenCV可使用opencv-python-headless==4.11.0.86；不要为标注工具升级整个训练环境。独立标注虚拟环境也可。

```bash
ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
WORK=/data/algorithm/agentic-wam/outputs/custom_original_v1
ENGINE="$PWD/codegen/pretrain/pipeline"
export CUSTOM_V3_DATASET="$ROOT"
export CUSTOM_V3_WORK="$WORK"
export CUSTOM_V3_OPEN_IS_HIGH=1
export CODEGEN_EMBODIMENT=custom_v3
export YAM_FRAME_GAP=13   # 10fps下约1.3秒，对应原30fps的40帧间隔
```

若低值代表打开，设置CUSTOM_V3_OPEN_IS_HIGH=0，且下面wrapper命令在subtasks/cache前增加--open-is-low。不要把归一化训练state传给检测器。

## 2. 免费检查分段（先运行这个）

```bash
"$PY" scripts/custom_v3_codegen.py --dataset "$ROOT" --work "$WORK" \
  subtasks --episodes 0 1 3075 3261
```

生成WORK/boundaries_report.json与WORK/units.txt。检查events、dropped_events、spans及指令是否正确。
该步骤不调用VLM，不生成可用于actor的subtasks.json。原始audit的事件帧不一定与原速度检测器相同，这不是自动判错；应对照原始夹爪信号。

## 3. 配置VLM，再为片段命名（会产生调用费用）

配置原引擎支持的VLM_URL、CODER_MODEL；VLM_URL需要是完整的OpenAI兼容chat/completions地址，模型需支持多图输入。VLM_API_KEY通过你自己的安全环境配置注入，不写入Git或命令历史。也支持原引擎CODEGEN_CONFIG配置方式。

```bash
# 此处假设 VLM_URL、CODER_MODEL、VLM_API_KEY 已配置。
"$PY" scripts/custom_v3_codegen.py --dataset "$ROOT" --work "$WORK" \
  subtasks --episodes 0 1 3075 3261 --label
```

生成WORK/subtasks.json。同设置已有命名会复用，改变设置请使用新WORK目录。每条记录保存原始state摘要与设置；数据源路径/metadata变更会拒绝复用。视频如被替换应使用新WORK和tag（没有对大视频做逐字节哈希）。

## 4. 原actor生成代码并执行原gate（会产生调用费用）

```bash
"$PY" "$ENGINE/scripts/run_batch.py" \
  --embodiment custom_v3 --tag custom_original_v1 \
  --tasks-file "$WORK/units.txt" --parallel 1 --skip-done
```

原引擎输出在codegen/pretrain/pipeline/outputs/custom_v3/custom/<model-short>/episode_NNN/custom_original_v1/，包含program.py、lines.json、trace.json和采样图。任务3仍需人工判断，不能仅看reconciled。

## 5. 导出样例缓存

使用上一步实际输出的model-short目录名，例如模型gcp/google/gemini-3.1-pro-preview对应gemini-3.1-pro。下例需替换成你实际模型：

```bash
MODEL_SHORT=gemini-3.1-pro
"$PY" scripts/custom_v3_codegen.py --dataset "$ROOT" --work "$WORK" \
  cache --episodes 0 1 3075 3261 --tag custom_original_v1 \
  --model-short "$MODEL_SHORT" --out-dir "$ROOT/coding_original_trial_v1"
```

试验缓存保留全部285262帧索引，未选择episode的标签为空并屏蔽代码loss；**不要把trial当成全量监督缓存**。原coding_v3不受影响。

输出policy_code_idx_custom.pt、episode_order.json、gate_report.json、generation_manifest.json。记录trial_masked/full模式与覆盖数。原actor输出是按源代码行抽取的块，不保证符合旧check_cache.py对单一policy_chunk函数/audit.jsonl的专用假设；使用本入口的严格gate/对齐检查，不能照搬旧检查器。

## 6. 样例人工验收后，再全量

相同设置、同一数据、同一WORK与tag可以续跑。将subtasks的--episodes列表改为--all并加--label，会先复用已完成命名，再生成全量units.txt；随后重跑原run_batch并保留--skip-done。最后cache阶段改为--all，指定新的out-dir（例如coding_original_full_v1）。

全量前检查调用预算、模型token限制和服务并发限制。若变更设置/数据/视频，使用新WORK与tag，避免旧生成结果复用。输出目录必须为空或不存在。

## 离线验证

```bash
python -m pytest codegen/posttrain/test_custom_v3_codegen.py -q
```

测试覆盖实际v3 packed文件读取、head共享视频offset、原API/提示词/技能库选择、10fps夹爪事件、恒开信号、原gate导出、试验全局索引与空监督，以及过期结果拒绝。没有发出真实VLM调用。
