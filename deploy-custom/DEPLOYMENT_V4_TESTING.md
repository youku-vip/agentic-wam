# Custom34 v4 图像与专家动作对比验证

基于 v3 增加每次推理的视频解码帧保存、专家 32 步动作比较、PNG 曲线与 CSV/JSON/NPZ 导出，以及数据集离线 HTTP 评测脚本。客户端与 v3 相同。

验证环境为 CPU Python 3.12、PyTorch 2.7.1，使用 Matplotlib、Pillow、PyAV、PyArrow、Pandas。HTTP 仅监听本机测试端口。模型和机器人控制均使用替身，未加载用户的真实 checkpoint，也没有控制机器人。

结果：70 项 pytest 检查、18 个图像子测试通过；原有 HTTP/RTC 前缀时序回归通过。动作转换往返最大误差为 1.7881393432617188e-7。测试工具另有一条 anyio/pytest 接口弃用提示，不影响检查结果。

仓库 checkout 中运行：

```bash
MPLCONFIGDIR=/tmp/wam-review-mpl PYTHONPATH=src python -m pytest -q \
  deploy-custom/test_chunk_review.py \
  deploy-custom/test_deployment_v3.py \
  deploy-custom/test_deployment_install.py \
  deploy-custom/test_client_server_contract.py \
  deploy-custom/test_sync_client_control.py \
  tests/test_letterbox_image_preproc.py::LetterboxGeometryTests \
  tests/test_letterbox_image_preproc.py::TrainDeployParityTests
python deploy-custom/test_custom34_rtc_adapter.py
PYTHONPATH=src python deploy-custom/validate_custom34_bridge.py
```

图像一致性测试仍依赖仓库的参考配置和基础模块；小型部署包只携带必要部署文件。

本轮新增验证：

- 实际 policy 方法在 episode 中间、连续多次推理时都请求 return_video=True，并保存 CPU 视频张量；不依赖“第一次推理”判断。
- 按相机分组的 [1,3,nc*T,H,W] 正确拆成三路 PNG；验证 RGB 通道未交换、相机角色可重排、未来帧步索引正确。
- 20 维绝对动作、64 维相对动作使用相同的观测基准，导出的 NPZ 与服务端返回动作一致。
- 已知 3 mm 平移误差和 90° 旋转误差得到预期数值；179° 与 -179° 的误差为 2°，不错误报告 358°。
- 夹爪准确率按与客户端相同的 >0.5 阈值计算。
- 尾部无效行不进入 MAE/RMSE/最大误差统计；无效参考形状、退化旋转及 mask 被拒绝。
- 实际输出 20 张逐维曲线、20 维总览、64 维总览与物理误差曲线；CSV、JSON 和 NPZ 完整。
- 专家动作 reference 不进入传给 policy 的 observation；仅在预测完成后用于比较。
- 写盘或绘图错误不丢弃已经计算的动作，响应明确报告 review.status=error；核心数组在绘图前落盘。
- 临时生成真实共享 MP4/Parquet：第二个 episode 具有 4 秒视频偏移；清洗样本以软链接复用视频并从 4.5 秒开始。即使 Parquet timestamp 故意不同，读取的观测 PTS、action[t] 和窗口边界仍正确。
- 实际离线脚本连接真实 HTTP handler，下载 review ZIP，并核对请求 ID、预测数组及专家数组；不能通过下载接口读取任意路径。

既有配置加载、技能提示词、文本缓存、同步控制、备份安装器和 224×320/256×448 图像一致性回归一并覆盖。真实模型预测质量和实际 VAE 解码显存、耗时仍需在用户 GPU 环境验证。报告中的模拟误差只用于验证计算，不代表 checkpoint 的模型精度。
