# Custom34 代码损失显存修复（v1 的增量包）

用户实测在首个训练前向的 code_loss 中，整段 logits 转 FP32 需要再分配 1.43 GiB，剩余显存只有约 64–328 MiB。此时 PyTorch 已分配约 76.9 GiB，真实张量占用接近上限。该日志中的 W&B 已在线初始化成功。

此补丁保留代码专家的复数 RoPE、128 token 画布与现有训练目标，改变交叉熵的内存安排：

- 先选择 label != -100 的 logits 行，忽略原本就不计入损失的视觉前缀、提示词和画布尾部。
- 每次最多将 32 行转换为 FP32，计算交叉熵。按当前 SDAR 的 151936 词表，该 FP32 logits 缓冲约为 18.55 MiB；这不是总显存估计。
- 使用非重入 checkpoint 在反向时重算每个交叉熵块，避免保留整段 FP32 log-softmax。
- 标准代码损失、sigma 按样本加权和 RL 按样本加权均使用此方法。损失分母、EOS 权重和已有样本权重保持原来的定义。RL 专用非零 entropy bonus 仍使用原实现；本次普通微调不启用它。
- 空标签仍产生有限的零损失，并保留到已参与前向参数的零梯度路径。

CPU 检查对照原来的 FP32 CE，验证数值和梯度在浮点容差内一致，覆盖 FP32/BF16/FP16、AR 位移、EOS 权重、sigma/RL 权重、空标签、无梯度和非连续输入。另验证每次 CE 输入行数上限，以及前向不会保留 FP32 词表矩阵。含原训练修复与 RoPE/专家代码评测回归共 31 项 CPU 检查通过。尚未在用户两卡 GPU 上实测峰值显存或训练吞吐。

## 手动替换

先备份原文件，再将增量包中的两个文件按相同相对路径覆盖到项目：

~~~text
src/agenticwam/models/experts/code_expert.py
src/agenticwam/models/cosmos_mot/agenticwam_model.py
~~~

两者需配套替换，因为按样本加权分支调用新增的 token_cross_entropy。保留先前安装的 build.py 和评测文件。

包中的 scripts/custom_code_repair.py 是可选更新：用于今后新建试验时，自动开启混合注意力的梯度检查点。替换它不会重新生成已有 run_code_repair.sh。本次已有试验通过下面的环境变量开启即可。

## 重启已有 300 步试验

~~~bash
cd /data/algorithm/agentic-wam
NEW_RUN="$PWD/runs/custom_code_repair_20261110_024220"

export AGENTICWAM_MOT_NO_CKPT_EVERY=0
export AGENTICWAM_DISABLE_MOT_ATTN_CKPT=0

CUDA_VISIBLE_DEVICES=0,1 bash "$NEW_RUN/run_code_repair.sh" train
~~~

NO_CKPT_EVERY=1 会跳过每一层混合注意力的 checkpoint；0 表示不跳过。DISABLE_MOT_ATTN_CKPT=0 允许启用它。此处仍需 recipe 中 model.use_gradient_checkpointing=true。对本次日志只能确认发生显存不足，不能据此断言之前某个环境开关的实际值。

保留当前 batch_size=4、gradient_accumulation_steps=8、两卡共 64 样本及每任务 16 样本的设置，保留图像尺寸、画布和统计文件。上次失败尚未完成优化器更新，因此继续从指定的旧 8000 步权重初始化本次试验。现有基线结果沿用，无需重新 prepare。

完成后评测：

~~~bash
CUDA_VISIBLE_DEVICES=1 bash "$NEW_RUN/run_code_repair.sh" eval
~~~

本补丁不改变推理路径或权重名称/形状。已有基线可继续用于前后比较。checkpoint 重计算会增加计算量，以换取较低的激活显存；两卡训练的实际结果仍需用户环境验证。
