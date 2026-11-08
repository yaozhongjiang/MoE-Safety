# ncu failed to connect to the CUDA driver：诊断与保持 v4 恢复能力

该错误发生在 ncu 初始化驱动连接时。错误文本中的 `stub libcuda.so[.1] on path?` 是工具给出的可能原因，不能单凭此文字断定加载了 toolkit 的链接 stub。

已知同一次运行此前通过了 doctor，并且已经进入 `full/shard_000000`。按 v4 的顺序，前一个 none 分片已经完成了恢复与校验。因此还应区分驱动库解析、WSL/驱动连接状态、profiler 与驱动的兼容性，不能把路径猜测当作已确认根因。

参考：[NVIDIA 相同错误讨论](https://forums.developer.nvidia.com/t/nsight-compute-failed-to-connect-to-the-cuda-driver-stub-libcuda-so-1-on-path/315446)、[NVIDIA profiling 故障说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

## 首先用同一份 v4 代码恢复

完全退出 Docker Desktop。在 Windows PowerShell 执行：

```powershell
wsl --shutdown
```

重新启动 Docker Desktop 和 Ubuntu，在原 real_channel 目录执行：

```bash
set -o pipefail
bash run.sh --resume artifacts/20261025T222451Z_c60ddd 2>&1 | tee paper-resume.log
```

这不会改变库路径或安装依赖。已有 `.done.json` 分片经 hash 校验后跳过，失败且无完成标记的 full 分片重跑。不要删除 cache、artifacts 或完成标记，不要编辑 run.json 绕过校验。若升级了驱动或运行时，程序可能按设计拒绝混合恢复；此时需要新建运行。

## 若重复失败，收集实际证据

将补充包解压到原 real_channel 的父目录。它只增加 validation 下的诊断脚本和文档，不改 v4 核心代码、测试或要求文件。

在 real_channel 目录、Docker Desktop 已运行时执行：

```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN --shm-size=2g \
  -v "$PWD:/workspace" --entrypoint python3 \
  moe-real-channel:cu124-v1 \
  validation/diagnose_cuda_driver.py \
  --output artifacts/20261025T222451Z_c60ddd/driver-diagnostics.json
```

脚本仅收集：

- nvidia-smi、实际 ncu 版本、有限白名单内的驱动搜索环境变量。
- 在独立进程中分别加载 `libcuda.so.1`、`libcuda.so`，记录真实 `/proc/self/maps` 映射路径、cuInit 返回值和 GPU 数量。
- 候选 runtime 库的路径与初始化结果，记录 toolkit stub 的存在但不主动加载它。
- ldconfig 的 libcuda 条目。

它不会修改 LD_LIBRARY_PATH、链接库、驱动、实验配置或分片记录；也不下载模型、数据或软件包。诊断报告不是硬件实验成功标记。

如需进一步区分连续 profiler 调用是否失败，可添加 `--profile-repeat 2`，仅采集两个极小 probe，不启动完整模型；日志写入诊断 JSON 同目录的 `driver-probes/`。

## 如何理解报告

- 默认加载的真实路径位于 CUDA Toolkit `stubs/`：有直接证据支持链接 stub 被误用于运行时。根据实际可初始化的 runtime 目录调整搜索顺序；先保留报告，不要随意删除或重建符号链接。
- 默认 libcuda 的 cuInit 不为 0，而指定实际 runtime 路径可初始化：支持动态加载路径问题。
- 默认 cuInit 成功且 GPU 数量正确，但 ncu 失败：继续核查 profiler/Windows 驱动兼容性及驱动连接状态；不能仅凭报错字符串断言 stub 冲突。
- 两次小 probe 都成功、完整模型采集失败：保留该分片完整日志，进一步检查大规模采集触发的驱动重置、资源问题或特定工具问题。

WSL 正常的 `/usr/lib/wsl/lib/libcuda.so.1` 是 Windows 提供的驱动接口，不等同于 Toolkit 的链接 stub；不要删除它。依据：[CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

## 补充包验证

诊断脚本的隔离加载、错误记录、候选库识别和重复 probe 日志有独立测试。它们放在 validation 下，不加入 v4 自动测试目录，也不影响 v4 的 code_fingerprint。补充包不声称已修复尚未确定根因的目标机器驱动问题。
