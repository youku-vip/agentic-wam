# Custom34 deployment v3 验证记录

本次基于用户上传的 custom34_server.py 和 rtc_client_agenticwam.py 修改；服务端继续调用项目的 CustomUnifiedEEFTransform 和三专家模型。默认模式为同步、RTC 关闭、夹爪锁定 0 步、自动开爪/回位恢复关闭。

在 CPU Python 3.12 / PyTorch 2.7.1 环境验证。HTTP 测试只监听本机临时端口；机器人环境与模型预测使用替身，没有向硬件发送命令。

结果：55 项 pytest 检查、18 个图像子测试通过；原有 HTTP/RTC 前缀及时序回归通过。测试工具有一条 anyio 对 pytest 接口的弃用提示，不影响检查结果。

仓库 checkout 中运行（需项目 CPU 测试依赖）：

```bash
PYTHONPATH=src python -m pytest -q \
  deploy-custom/test_deployment_v3.py \
  deploy-custom/test_deployment_install.py \
  deploy-custom/test_client_server_contract.py \
  deploy-custom/test_sync_client_control.py \
  tests/test_letterbox_image_preproc.py::LetterboxGeometryTests \
  tests/test_letterbox_image_preproc.py::TrainDeployParityTests
python deploy-custom/test_custom34_rtc_adapter.py
PYTHONPATH=src python deploy-custom/validate_custom34_bridge.py
```

图像测试依赖仓库现有 configs、processor 与其他辅助模块；小型部署包没有整份仓库，请在对应仓库 checkout 中复现。

覆盖内容：

- 读取已解析的 run/config.yaml，不依赖本机 task/data 分组文件；不修改原配置。
- run/dataset_stats.json 优先于数据集目录的统计文件，显式 --dataset-stats 可覆盖。
- 四种任务的 skill map 按当前 instruction 取值，复用训练的 skill_prompt_only/code_instruction_mode 组合方式。
- 实际 policy 推理方法使用传入配置的提示词，绕过旧 TASK/技能/历史环境分支。
- 文本缓存使用训练相同的完整提示词 SHA256 和 .cosmos_reason1.pt 文件格式；CPU 缓存有容量上限，未知任务、零向量和非有限值报错。
- 实际服务端 handler 与客户端 HTTP 方法配对，校验 state34、三路 RGB、绝对 action20、request_id、实例 ID、reset 与 chunk 长度。
- 无效旋转、错误响应形状、HTTP 错误和实例变化不会返回本次可执行动作；实例变化后重新握手/reset。
- 实际同步循环每个 chunk 一次推理、每次重试采集新观测；同文任务重新提交、清空/恢复、任务切换与回放会停止旧 chunk。
- 默认不锁爪、不根据低开度自动恢复；显式开启时保留旧恢复流程。
- 224×320 与 256×448、stretch 与 letterbox 的实际训练/部署图像逐像素一致；256×448/no-upscale 对 240×424 原图上下各补 8、左右各补 12。
- 安装前验证包哈希、先备份后替换；安装中失败回滚；重复安装无修改；回退拒绝覆盖安装后的用户编辑。
- 只读旋转检查可区分 SDK 前两列/前两行、弧度/角度，且无需导入机器人控制包。

数值转换往返：raw state [1,1,34] → unified state [1,1,64]，raw action [1,32,20] → relative action [1,32,64] → absolute action20，最大误差 1.7881393432617188e-7。

仍未验证：用户 checkpoint 的真实推理、CUDA/bf16 与可选编译性能、SDK 实际旋转约定、相机/状态采集同步、机器人控制器跟踪与任务成功率。此包保留用户原有 SDK Euler/6D 正反转换；机器人端的只读检查结果仍需核对。
