# 不使用VLM：复用已知阶段，去除release/place重复回撤

本脚本属于原项目Route B。使用已经确认任务语义的custom缓存、episode_order和逐阶段audit重建代码标签，不重新检测夹爪、不重新估计坐标/旋钮角度。可保留上传结果中202条旋转窗口细化记录，不依赖当前仓库缺少的细化生成器。

该脚本**不是**从原始视频重新标注的程序。它解决已有代码中的语义重复，不证明输入阶段本身正确；若要调整接近/抓取/抬起边界，需要另行提供明确帧范围或原始数据。

## 变换规则

|任务|重建后的语义|
|---|---|
|开门|grasp覆盖接近和闭合；move_arm + release覆盖持握操作、释放及其回撤|
|放物品|pick；place覆盖放置、打开及其回撤|
|旋钮|grasp；rotate + release覆盖旋转、释放及其回撤|
|恒开夹爪推门|保留原来的推门move_arm和回撤move_arm，因为不存在release/place可承接|

具体做法：将release/place阶段与其后连续的retreat阶段合并，保留原release/place阶段代码，删除额外回撤阶段的move_arm标签。回撤帧并入合并阶段，不是把其监督清空。回撤后的空闲标签仍保持为空。原事件帧、坐标、角度及警告保留；被合并的回撤边界和运动元数据仍在motion.semantic_retreat_merge中。

**适用前提：你确认这段回撤确实属于该技能的语义。**原API中release包含“竖直回撤”；如果实际轨迹还包含横向归位或另一独立运动，不能认为它必然被release覆盖。可用--keep-retreat-episodes保留这些episode的独立运动。place包含放置后的回撤，但其他操作也不能混入。

这是粗粒度符号计划，不是可直接执行的机器人程序；动作专家继续使用原始action。它也没有重新定位pick的抬起终点，不应把旧close边界解释成已经完成抬起。

## 本地运行

从项目根目录执行，无需VLM配置：

```bash
ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
CACHE_DIR="$ROOT/coding_v3"

/usr/local/bin/python3.11 scripts/rebuild_semantic_code_cache.py \
  --cache "$CACHE_DIR/policy_code_idx_custom.pt" \
  --order "$CACHE_DIR/episode_order.json" \
  --audit "$CACHE_DIR/gripper_code_audit.jsonl" \
  --info "$ROOT/meta/info.json" \
  --out-dir "$ROOT/coding_semantic_v1"
```

例如确认episode3075包含额外归位，需要保留时，在上面命令添加--keep-retreat-episodes 3075（只是用法示例，不代表本次已判定该episode必须保留）。输出目录必须新建或为空，不覆盖coding_v3。

输出policy_code_idx_custom.pt、episode_order.json、gripper_code_audit.jsonl、semantic_merge_report.json。可以继续使用此前check_cache.py做结构检查，并使用render_code_video.py生成新旧图文对照。

输入以torch.load(weights_only=True)读取，检查总量、索引、完整阶段覆盖、cache/audit/order逐项一致。只合并同一只手、纯move_arm的紧邻retreat，出现其他动作则拒绝。输出重复相同检查，不执行任何代码标签。

## 对上传文件的结果（默认按技能覆盖回撤，不排除episode）

- 4348 episodes / 285262 frames保持一致。
- 合并3261个回撤阶段，移除9782个重复move_arm调用（按阶段中的代码语句计数，不是执行次数）。
- move_arm语句从13043降为3261；grasp2174、release2174、pick1087、place1087、rotate1087均保持不变。
- 任务3全部推门/回撤代码保留。
- 每任务有标签帧数保持不变，总计284969。全量cache/audit/order核对通过。
- 旋转角度没有重新计算，所有警告保留。未读取真实视频，因此这份输出是供人工复核的候选缓存，不能宣称物理语义已通过验收。

试用时修改训练命令的data.code_cache指向新文件；原始state/action与统计文件不变，不因本次仅改代码标签而重算归一化统计。正式替换前先核对语义覆盖前提、真实loader顺序和token长度。
