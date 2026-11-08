# 48GB 内存 / RTX 3070 8GB：完整 paper 实验的加速参数

优化目标是缩短相同实验的实际完成时间。当前代码固定 batch=1，每个 token 的各专家独立执行；Nsight Compute 逐 kernel 采计数器。这类测量中 GPU 利用率不一定高。不能通过增大 batch、并行多个 profiler、删掉 anchor、减少 token/层数来宣称同一项实验加速。

这里提供现有 v4 的补充启动器和真实硬件调参工具，不改核心代码。默认分片从 1 提高到 4，模型仍只保留一个实例，请求仍逐个处理。分片大小主要影响重复启动、加载、预热次数，以及 CPU 中的 Nsight 报告大小；不意味着同时把 4 个请求放进显存。

## 1. 先确认所在目录，再安装补充包

在 WSL Ubuntu 中进入**原来执行 `bash run.sh` 的目录**。这个目录必须同时能看到 `run.sh`、`cli.py`、`validation/`、`artifacts/`。把下载的 `real_channel_rtx3070_performance.zip` 放进该目录。

执行：

```bash
test -f run.sh && test -f cli.py && pwd
unzip -o real_channel_rtx3070_performance.zip -d .
ls validation/run_rtx3070_fast.sh validation/rtx3070_fast.py
```

这个补充 ZIP 根目录**直接是 `validation/`**，没有额外的 `real_channel/` 套层。解压后应为：

```text
原实验目录/                       ← 运行命令的位置
├── run.sh
├── cli.py
├── cache/                       ← 继续使用已下载的魔搭模型/数据
├── artifacts/                   ← 原来的结果全部保留
└── validation/
    ├── run_rtx3070_fast.sh       ← 新增
    ├── rtx3070_fast.py           ← 新增
    ├── RTX3070_PERFORMANCE.zh-CN.md
    └── ...
```

## 2. 立即开始新的加速版完整实验

同一个 GPU 一次只运行一项采集或调参任务。若旧实验仍在运行，先在其终端 Ctrl+C，等待容器退出，保留旧进度。

在上面的原实验目录执行：

```bash
set -o pipefail
bash validation/run_rtx3070_fast.sh launch --shard-requests 4 2>&1 | tee paper-fast-launch.log
```

也可以不装本补充包，直接用原有代码：

```bash
set -o pipefail
bash run.sh --preset paper --shard-requests 4 2>&1 | tee paper-fast-launch.log
```

两者的实验参数相同。补充启动器会复用已安装的 `moe-real-channel:cu124-v1` 镜像，避免每次 build/pull；镜像不存在时才按原 Dockerfile 构建，原镜像源仍为 `docker.1ms.run`。

| 项目 | 原 paper | 建议起点 |
|---|---:|---:|
| `shard_requests` | 1 | 4 |
| 采集请求总数 | 600 | 600 |
| 每请求测量 token | 32 | 32 |
| 测量层数 | 24 | 24 |
| 每 token batch | 1 | 1 |
| 精度 | BF16 | BF16 |
| none/full 采集进程合计 | 1200 | 300 |
| 采集时完整请求预热次数 | 1200 | 300 |
| 同时运行的模型 / profiler | 1 / 1 | 1 / 1 |

进程数减少 75% 不等于总耗时减少 75%，更不等于 4 倍速度。被测 kernel 总数没有减少。单个分片的报告和重试成本变大，所以建议先用 4，不直接设 16 或 64。

## 3. 想选实测最快的参数：先做相同请求的 GPU 对比

以已有的正式运行 `artifacts/20261025T222451Z_c60ddd` 作为只读参考。该目录需要有成功的 `validation_gpu.json` 和 `run.json`，不要求已经采集完毕。

```bash
set -o pipefail
bash validation/run_rtx3070_fast.sh tune \
  --reference-run artifacts/20261025T222451Z_c60ddd \
  --sizes 1,2,4 --samples 4 --repeats 1 \
  2>&1 | tee rtx3070-tune.log
```

此命令确实运行 GPU/ncu，不是耗时估算器。每个候选对同样的前 4 个请求各采 none/full，保持 32 token、全部 24 层和相同模型，合计启动 14 个采集进程。调参本身需要时间；不希望付出这部分时间可直接使用上一节分片 4。

它会先验证参考运行、模型文件及硬件身份，执行真实 GPU 正负对照和原有 CPU 测试；然后在新目录生成独立 pilot，逐个运行候选。每个候选须满足：观测完整、计数器恢复等于独立 truth、不同候选的恢复集合及 truth 一致。任意失败立即停止，不写成功推荐、不将失败计为快速样本，也不会自动重试掩盖驱动问题。

支持默认 BF16/code 的完整 paper 配置。参考目录若是 smoke/quick，或改了层数、token、精度等，它会明确拒绝据此推荐 paper 参数。

结果在：

```text
artifacts/rtx3070_tuning_<时间_后缀>/
├── hardware.json
├── preflight/
├── timings.json
├── recommendation.json          ← 全部通过后才生成
└── repeat_01/
    ├── shard_01/
    ├── shard_02/
    └── shard_04/
```

`median_seconds_per_request` 是一个请求的 **none+full 配对采集和校验**耗时；`measured_speedup_over_1` 是相对分片 1 的实测速比。差距不足 3% 时选更小分片，减少报告内存及失败重采成本。短 pilot 的热状态和噪声会影响推荐，它不是全程速度保证。需要更稳妥比较可用 `--repeats 2`，第二轮倒序运行候选，额外开销也翻倍。

调参成功后会打印启动命令，例如：

```bash
bash validation/run_rtx3070_fast.sh launch --shard-requests 2
```

只有完成了 1/2/4 对比且 4 明显更好、Ubuntu 的 `free -h` / `vmstat 1` 显示无内存压力时，再考虑 `--sizes 1,4,8 --samples 8`。这会生成更大的报告并重新采样，不会自动进行。

pilot 目录明确标为 `TUNING_ONLY.json`，不会生成正式实验的 `COMPLETE.json`。原有采集及训练/测试数据保持不动，pilot 不混入论文结果。

## 4. 48GB 主机上的 WSL 设置

建议给 WSL 32GB 内存、8GB swap，保留约 16GB 给 Windows 等进程。CPU 型号/核心数尚未提供，先保留 WSL 默认的逻辑处理器数量，不盲目限制为某个固定数量。

在 Windows PowerShell 打开配置文件：

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

在已有 `[wsl2]` 节中合并以下两行；如果文件为空，就写入三行：

```ini
[wsl2]
memory=32GB
swap=8GB
```

不要创建第二个 `[wsl2]` 节，不覆盖原文件中其他设置。在确认所有实验已经停止后，退出 Docker Desktop，在 PowerShell 执行 `wsl --shutdown`，再启动 Docker Desktop 和 Ubuntu。

在 Ubuntu 检查：

```bash
free -h
nproc
docker run --rm --entrypoint python3 moe-real-channel:cu124-v1 -c 'from pathlib import Path; print(Path("/proc/meminfo").read_text().splitlines()[0])'
```

主机有 48GB 并不代表 Docker/WSL 已获得 48GB；需要核对容器可见内存。swap 只是容量余量，持续换页会拖慢采集，不能用增大 swap 代替物理内存。

Microsoft 的配置说明：[WSL advanced settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)。

## 5. 文件放在 Ubuntu 自己的文件系统

模型、代码、报告尽量放在 Ubuntu 的 `/home/<Ubuntu用户名>/...`，而不是 `/mnt/c/...` 或 `/mnt/d/...`。本实验有大量报告写入、CSV 读取及 hash 校验，跨 Windows 文件系统可能拖慢这一过程。Docker 官方明确推荐把 bind mount 的文件放在 Linux 文件系统：[WSL 2 best practices](https://docs.docker.com/desktop/features/wsl/best-practices/)。

如果现在已经放在 `/home/`，无需搬动。如果在 `/mnt/` 下，先停止实验，再复制完整实验目录（包含 `cache/` 和 `artifacts/`）到 Ubuntu 的 home；原文件保留。之后从新目录运行，仍挂载为容器内 `/workspace`，避免改变容器内的模型路径。

## 6. 旧实验、结果位置和恢复

**旧实验不能在 `--resume` 时直接改成新的分片大小。** 原 CLI 会读取原来的 config，并校验完整性。不要编辑旧 `config.json`、`.done.json` 或指纹文件来强行改参。

若想保留并继续旧实验，仍执行原来的：

```bash
python3 validation/resume_guard.py \
  --run artifacts/20261025T222451Z_c60ddd --attempts 10
```

新启动器每次新建：

```text
artifacts/paper_fast_<时间_后缀>/
├── config.json                  ← shard_requests 记录为实际值
├── validation_gpu.json
├── profiles/none/
├── profiles/full/
├── report.md
├── results.json
└── COMPLETE.json                ← 仅完整成功后生成
```

若新实验出现之前的间歇性驱动连接错误，用 `resume_guard.py --run artifacts/paper_fast_<实际名称> --attempts 10` 恢复这个新目录，分片大小会从新目录的 config 中读取。不要把旧、新目录的 profiles 合并。

所有正式的数学正确性、stock greedy 一致、none/full bitwise 一致、GPU 正负对照、独立 truth 校验及断点 hash 校验，仍由 v4 原代码执行。

## 7. 提速边界与验证

NVIDIA 文档说明计数器采集有 replay、CUDA API 拦截等开销，并会串行化被测 kernel；部分 GPU/driver profiling 资源只能由一个进程使用：[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead)、[Serialization](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#serialization)。因此不能承诺靠参数把 GPU 长期跑到 90% 以上。本补充包将比较相同实验的耗时，而不是把显存占满或把 GPU 利用率读数抬高。

新增文件全部在 `validation/`，核心 v4 指纹仍为 `83bf6a5cab9f49243ffbbd4a07b78141d76fa63583ae3a6a2d6ac5b81d2c4e54`。开发机没有 RTX 3070/WSL；CPU 测试和模拟子进程校验只能验证软件契约，实际速度、显存和报告内存峰值以目标机调参/正式运行结果为准。

发布校验共三轮：新增工具与恢复/诊断 40 项通过；原有核心 86 项通过；将 v4 整包和本补充包解压到全新目录后，合并 126 项通过。记录见 `rtx3070_performance_validation.json` 和 `rtx3070_performance_round3.log`。这些数字是 CPU 软件检查，不是 RTX 3070 性能测试。
