# v4.1 非有限图像与局部 VAE 解码验证

CPU Python 3.12、PyTorch 2.7.1：80 项 pytest 检查、18 个图像子测试通过。未使用用户 checkpoint、真实 VAE 权重或 CUDA，不能据此保证用户 GPU 上的 NaN 已解决。

```bash
MPLCONFIGDIR=/tmp/wam-review-mpl PYTHONPATH=src python -m pytest -q \
  deploy-custom/test_review_vae_decode.py \
  deploy-custom/test_chunk_review.py \
  deploy-custom/test_deployment_v3.py \
  deploy-custom/test_deployment_install.py \
  deploy-custom/test_client_server_contract.py \
  deploy-custom/test_sync_client_control.py \
  tests/test_letterbox_image_preproc.py::LetterboxGeometryTests \
  tests/test_letterbox_image_preproc.py::TrainDeployParityTests
```

本轮覆盖：

- FP32 解码在外层 bf16 autocast 下仍以 FP32、autocast disabled 调用 VAE。
- 逐相机解码保持 batch/camera/time 布局。
- 解码前后参数 dtype、值、原始 storage pointer 精确一致，包括混合 float64 buffer；模型其他参数不改变。
- 解码异常也恢复原参数；重复安装 hook 不会叠加包装；native 模式保留原方法。
- 非有限 latent 不被静默替换，诊断正确记录异常。
- NaN、+Inf、-Inf 帧有单独诊断，不输出伪造的黑色有效帧；有限帧正常保存。
- 全部视频帧 NaN 时，实际导出 20 维/64 维动作图和原始数组，状态为 partial。
- 通过真实本地 HTTP handler 下载 partial 报告，脚本仍保存有效动作比较，并以退出码 2 报告图像警告。
- 既有图像布局、专家窗口、尾部 mask、数值误差、同步客户端与备份安装器回归。

上一次 CPU 模拟只覆盖了有限的视频张量，因此没有复现用户这次的非有限像素。这轮补充了该失效路径，而真实解码根因仍需要用户使用新诊断运行一个样本确认。
