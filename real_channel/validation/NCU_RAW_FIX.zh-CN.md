# 2026-10-25：Nsight Compute raw 导出修复

用户在 RTX 3070/WSL 上完成性能计数器授权后，真实采集已产生 `probe.ncu-rep`。
旧版代码在 CSV 导出阶段因下列参数组合失败：

```text
--page raw --print-metric-name name
Option '--print-metric-name' is only supported for the details page.
```

这是提供的代码中存在的 CLI 参数兼容性问题。原先的 CPU 测试没有捕获此项实际 ncu 限制。

## 修复内容

- raw 导出移除仅支持 details 页的 `--print-metric-name`。保留 raw 页、base 单位及 NVTX kernel 重命名。
- CSV parser 同时支持 long 布局（Metric Name / Metric Value）和 wide 布局（每个 metric 一个列），兼容 wide 的单位行。
- 缺失 counter、截断 CSV、无效值及缺失 anchor 继续严格报错。
- 错误提示只在实际出现 `ERR_NVGPUCTRPERM` 时指向 Windows/WSL counter 权限文档，其他命令错误提示查看具体日志。

修复后的导出命令形如：

```bash
ncu --import probe.ncu-rep --page raw --csv --print-units base --print-nvtx-rename kernel
```

依据：[NVIDIA Nsight Compute CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
与 [2022.4 CLI 选项说明](https://developer.nvidia.com/nsight-compute-20224-new-features)。

## 验证范围

- observer/orchestration 针对性回归：32 项通过，包括使用独立子进程模拟 ncu 的合法参数契约并贯通导出与恢复。
- 完整 CPU 回归：**54 passed in 12.96s**，实际结果见 `ncu_fix_regression.log`，使用临时目录中的 Transformers 4.51.3 / tokenizers 0.21.1 / huggingface-hub 0.30.2，以及本机 PyTorch 2.11.0。
- 重新执行 Python 语法检查、Bash 语法检查及压缩包解压后逐文件 SHA-256 校验。

本机仍无 CUDA GPU；子进程契约测试不是实际 ncu 二进制或 GPU 验收。目标机器需用新版重新运行 doctor。

## 更新方法

解压 v2 包覆盖原 `real_channel` 代码文件，保留现有 `cache/`、`artifacts/`。Python 源码通过 bind mount 从主机加载，因此本次修复不需要重建依赖镜像。正常执行 `bash run.sh --doctor-only` 即可。

也可以在实验目录使用已有镜像直接复测，以省略 build 步骤：

```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN --shm-size=2g \
  -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
  -v "$PWD:/workspace" moe-real-channel:cu124-v1 --doctor-only
```

此前失败发生在 doctor 阶段，不要对该不完整目录执行 `--resume`；正常启动会建立新的运行目录。
