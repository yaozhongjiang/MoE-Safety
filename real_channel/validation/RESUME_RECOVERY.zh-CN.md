# v4 的间歇性 ncu 驱动连接错误：有上限的恢复

已运行到 `none/shard_000071` 才再次出现驱动连接错误。此前大量分片在同一环境成功，因而固定的 stub 加载错误不能作为未经验证的结论；尚需实际的 libcuda、cuInit、Windows 驱动/ncu 版本等诊断证据。

补充脚本 `validation/resume_guard.py` 用于减少反复手工重启的负担。它是有限次数的恢复机制，**不是对驱动根因已经修复的承诺**。

## 使用

只更新本补充包，保留正在使用的 v4 核心代码和镜像，不用其他整包覆盖正在恢复的实验。

在 WSL Ubuntu 的原 real_channel 目录执行：

```bash
python3 validation/resume_guard.py \
  --run artifacts/20261025T222451Z_c60ddd \
  --attempts 10
```

默认只尝试 5 次，可显式选择 1–20 次。默认间隔为 10、20、40 秒，后续最多 60 秒。达到上限后明确失败退出，已完成进度保留，可在排查后继续恢复。

该脚本使用当前已安装的 Docker 镜像 `moe-real-channel:cu124-v1`，每次启动一个新容器，调用原 CLI 的 `--resume`。不重复 build 或 pull，不更改驱动/模型/精度/采集参数，不跳过采集，不重抽样本，不修改指纹与完成标记。原 CLI 的输入、模型文件和分片 hash 检查照常执行。

只对以下真实日志组合重试：ncu 明确报驱动连接失败，并且原 CLI 因命令失败退出。权限、NaN/Inf、OOM、完整性失败、用户取消及其他错误均停止。退出码 0 还必须对应成功的 COMPLETE.json 才会打印完成。

一次只运行一个该实验的恢复进程，避免同时让多个容器争用同一 GPU 的 profiler 资源。NVIDIA 说明 profiler 所需的一些 GPU/驱动对象需要独占访问：[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#serialization)。

## 日志

每个恢复会话的全部启动输出与重试信息保存在：

```text
artifacts/20261025T222451Z_c60ddd/recovery/<UTC时间_随机后缀>/
  attempt_01.log
  attempt_02.log
  ...
  attempts.json
```

核心采集日志仍在原 `profiles/`；完成分片仍由原 CLI 的 `.done.json` 管理。所有失败和重试均保留在 recovery 日志中，不作为成功采样写入分析。成功后原 results.json、report.md 和 COMPLETE.json 路径不变。

Ctrl+C 会停止脚本本次创建的唯一容器，并且不自动重试；它不会停止其他容器。

## 若持续失败

重复失败需要实际诊断证据，不能仅用重试替代根因排查。执行：

```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN --shm-size=2g \
  -v "$PWD:/workspace" --entrypoint python3 \
  moe-real-channel:cu124-v1 \
  validation/diagnose_cuda_driver.py \
  --output artifacts/20261025T222451Z_c60ddd/driver-diagnostics.json
```

详细解释见 DRIVER_CONNECTION.zh-CN.md。请提供该 JSON 以及 recovery 最近一次 attempt 日志末尾，而非只重复同一段错误文本。现有成功分片可以保留。

## 验证边界

补充代码与测试均放在 validation 下，已验证 v4 code_fingerprint 仍为 `83bf6a5cab9f49243ffbbd4a07b78141d76fa63583ae3a6a2d6ac5b81d2c4e54`。

本补充包的恢复与驱动诊断测试共 **21 项通过**，实际日志见 resume_guard_tests.log。覆盖精确错误分类、其他错误不重试、同一个运行目录的有限重试与等待上限、保留进度、完成标记、取消及真实子进程日志，并包含两个独立的模拟 Docker 子进程贯通恢复 CLI 的测试。模拟 Docker 子进程不代表已经修复目标 WSL/NVIDIA 驱动；开发机没有对应 GPU。
