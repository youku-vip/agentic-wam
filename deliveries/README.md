# Custom 数据代码审查交付包

清洗后旋钮代码修复：`custom-curated-knob-code-repair-v1.zip`。利用原 audit 中完整保留的细化旋转窗口，逐条核对真实新旧 Parquet 姿态并复算角度；边界释放需源夹爪确认。写入新缓存目录，不覆盖数据、统计或现场配置。见 [诊断、校验和运行命令](../docs/CURATED_KNOB_CODE_REPAIR.md)。

专家轨迹检查：`custom-trajectory-audit-v2.zip`，只读扫描实际位姿、目标位姿和关节，输出速度/加速度/jerk 候选点、时间戳问题及可选曲线图；增加少量原始轨迹导出，修复 v1 关节夹爪上下文标记。旧 v1 包保留。见 [运行和解释结果](../docs/CUSTOM_TRAJECTORY_DYNAMICS.md)。

256×448 图像适配包：`custom-image-letterbox-v2-256x448.zip`，包含 v1 功能，增加动态部署尺寸和一键同步尺寸配置。当前原图 `240×424` 不缩放，上下各补 8 行、左右各补 12 列；保留原始像素和现场路径。见 [安装和切换命令](../docs/CUSTOM_IMAGE_256X448.md)。

图像比例修复包：`custom-image-letterbox-v1.zip`。在保持模型输入 `224×320` 的情况下等比例缩放并补边，同步训练、验证和部署；默认保留旧 stretch 行为，需显式启用。安装器备份源码并保留现场 YAML/`train.sh`，配置工具只修改两个 processor 的 resize mode。见 [安装、启用和回退](../docs/CUSTOM_IMAGE_LETTERBOX.md)。

四任务均衡采样包：`training-throughput-fixes-v2-task-balanced.zip`，包含原有优化和均衡 batch 功能。开启后，当前双卡 batch=4/GAS=8 下每次更新四任务各 16 个样本。见 [安装和验证](../docs/TASK_BALANCED_BATCHES.md)。

训练等待优化包：`training-throughput-fixes-v1.zip`，包括可选的 DeepSpeed 反向参数计数缓存、图像提前抽帧、worker decoder LRU、只读一致性检查与保留本地配置的安装器。优先阅读 [安装与验证命令](../docs/TRAINING_THROUGHPUT_FIXES.md)。默认不开启优化，需安装后显式设置环境变量。

以下说明对应原始 `custom-data-review.zip`：

- uploaded/：用户提供的元数据、代码缓存、顺序表、审计与归一化统计文件，内容保持原样，仅统一文件名。
- generated/：本次生成的检查脚本、报告、JSON检查结果、图文对照脚本及测试产物。
- MANIFEST.json：各文件长度与SHA256。

优先阅读 generated/真实数据代码审查.md。
图文对照：安装 av pandas pyarrow pillow 后运行 generated/render_code_video.py --help，root/audit/out由你指定。
原始真实视频和帧Parquet未提供，因此包内没有真实数据集视频。

注意：generated/synthetic/和generated/render-test/全部为合成测试数据，不能当作真实数据。synthetic/bad.pt是故意损坏的测试缓存，不能用于训练。
generated/审查结论.md是收到真实附件前的初步报告；真实附件结果以generated/真实数据代码审查.md为准。
generated/analyze_uploaded.py包含本次环境绝对路径，换机器需调整；check_cache.py和render_code_video.py使用命令行路径。
generated/code_video_review_tool.zip是先前提供的独立工具包。

本压缩包不包含Python虚拟环境、模型权重、Git元数据或编译缓存，不执行其中的代码标签。

文件：custom-data-review.zip
