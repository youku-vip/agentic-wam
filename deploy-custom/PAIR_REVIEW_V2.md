# Custom34客户端/服务端第二轮联动审查

基线：仓库custom34_server.py与用户上传的同步客户端，已包含上轮同步循环修正。未连接真实机器人或远程推理服务，未加载GPU模型权重。

## 结论

协议和坐标变换链路能够在模拟模型、真实HTTP接口下联通：state34 + head/left/right PNG -> 当前状态归一化 -> 模型输出归一化delta64 -> 一次动作反归一化 -> 以本次state为基准解码绝对action20 -> 客户端转Euler交底层控制器。

没有发现服务端重复反归一化。该结论不证明trans_utils的外部实现、真实相机通道或控制器坐标系正确，也不证明实际GPU模型抓取成功。

## 已修正

### 服务端

1. 非joint模型原先按episode缓存代码，Custom34逐阶段闭环可能复用过期阶段。现在Custom34服务的非joint路径每次重新生成代码；joint路径不受影响。没有清空已配置的学习型code_memory来冒充“无历史模式”。
2. 指令字符串变化时，服务端自身也执行episode reset，避免只依赖客户端。同一字符串的重复任务仍需显式reset。
3. reset同时清理最近代码、canvas与RTC暂存。RTC字段在正常返回和异常时均由finally清理。
4. 拒绝输入state及RTC committed中的退化rotation6D；输出action20也检查旋转列非零且线性独立，避免有限值但无效姿态被当作合法目标。
5. 控制台/代码文件日志写入失败不再把已完成的推理变成HTTP错误。日志仍可能阻塞耗时，需看latency，不能声称日志已异步化。
6. 增加服务实例server_id、episode_count、code_objective、fast_joint、code_memory、history_modes诊断信息。健康接口不暴露密钥或完整模型路径。
7. 补齐deploy/common模块搜索路径，允许源文件位于deploy或deploy-custom位置时找到fast_joint_infer。

### 客户端

1. 保留上轮修正：同步一块一次推理、只用新反馈判断夹爪异常、按episode刷新恢复姿态及夹爪稳定器、对应远端reset、正确计数已执行动作。
2. 相机输入必须为uint8 HxWx3，禁止将float图像直接转uint8造成近黑画面；不会自动猜测RGB/BGR。
3. 检查双臂xyz/Euler形状和有限值；未知或缺失夹爪反馈不再静默当成closed。
4. 恢复HTTP(S) URL检查，同时允许未显式填写端口的标准HTTP/HTTPS地址。
5. 记录服务实例变化警告，帮助发现重启/多副本路由。仅诊断，不实现跨副本会话隔离或自动重连握手。
6. 同步模式在chunk内每条策略命令前检查回放状态，发现回放则丢弃剩余旧chunk。但检查与底层命令不是原子的，不能保证回放线程与已在途控制命令互斥。

## 仍需真实依赖/实机证据

- trans_tools.trans_utils：Euler的弧度/度、旋转6D前两列或前两行、左右臂参考系是否与采集端一致。数值往返测试不能替代坐标约定确认。
- RealManGym及底层控制器：env.step返回是否表示动作完成、反馈时间戳、运动限位/速度/碰撞限制，以及get_obs是否三相机和状态同步。
- 夹爪稳定器锁15步和arm_actpos<50判异常：保持原值与原恢复动作，需针对夹爪/物体校准。低开度不必然是抓空；自动打开双臂并恢复姿态需由用户确认。
- 回放线程缺乏控制器级互斥/停止确认，不应在策略动作仍在执行时启动回放。当前补丁只减少后续策略动作，不能中断已发出的控制命令。
- 服务端状态仍为单一全局policy，并不是多会话隔离服务。多个客户端或多副本需独立实例或明确会话路由。
- 实际服务端启动命令尚未提供。用户提供的python -m environment.rtc_client_agenticwam是客户端命令，无法据此确认服务端checkpoint、推理模式及环境开关。
- 推理代码canvas默认64，训练上限512包含prompt。尚未用实际模型tokenizer检查各阶段代码是否超出推理canvas；长move_arm/多语句片段需特别核对，不能仅用字符数判断。
- 协议目前没有观测采集时间戳、请求序号和幂等重试标识；超时不能证明服务端未完成请求。客户端失败后重新观测再重规划，不能视为取回同一次预测。
- 模型是否随运行变差仍未复现。需相同输入、受控随机采样的离线回放与延迟记录；可对照fast-joint编译/非编译路径，但不能把固定随机种子当作精度修复。

## 部署文件

先暂停客户端请求，分别备份并更新实际运行文件：

- 服务器：custom34_server.py，通常放在仓库deploy/目录；或设置AGENTICWAM_ROOT后使用实际路径。
- 机器人客户端：environment.rtc_client_agenticwam对应模块。沿用上轮用importlib.util.find_spec查模块路径的方法，不要用服务端文件覆盖客户端。

原H=32、s_min=20、10Hz、同步无RTC启动参数保持不变。本次未修改任务模型、统计文件或训练代码。

启动新版服务后检查（替换成你的地址）：

```bash
curl -sS http://127.0.0.1:8765/health
```

查看server_id、episode_count、code_objective、fast_joint、code_memory、history_modes；如果没有这些字段，说明仍在运行旧服务文件。客户端出现backend instance changed时检查是否重启或存在多个后端，并不能仅凭这一条日志判定路由错误。

## 验证证据

```bash
python -m pytest deploy-custom/test_client_server_contract.py deploy-custom/test_sync_client_control.py -q
python deploy-custom/test_custom34_rtc_adapter.py
python deploy-custom/validate_custom34_bridge.py
```

15项pytest检查通过：同步控制流、任务边界、实际HTTP handler与真实客户端HTTP方法联动、模型替身下的坐标/动作输出、非joint代码缓存刷新、日志失败和RTC清理。原HTTP/RTC异步前缀测试通过。Custom34变换往返最大误差约1.8e-7。

所有模型预测和机器人控制均由模拟对象代替。未验证真实checkpoint预测精度、CUDA编译结果、真实RTC引导或机械臂执行。检查通过不等于生产环境抓取问题已解决。
