# Windows + WSL2 Ubuntu + RTX 3070 8GB：从配置到实验完成

这份文档对应本目录的真实 GPU 性能计数器实验。建议依次执行：**配置 Windows → 配置 WSL → 打开计数器权限 → 配置 Docker Desktop → 拷贝代码 → doctor → smoke → quick/paper**。

本实验实现 MoEcho 的 GPU 性能计数器路径及本论文的全覆盖防御评估；GateBleed 是 Intel AMX 信道，不能在 RTX 3070 上运行。`doctor` 会真正读取计数器，不能只用 `nvidia-smi` 或一次 PyTorch 矩阵乘作为实验可运行的证明。

## 1. 准备要求

| 组件 | 建议 |
|---|---|
| Windows | Windows 11，安装系统更新；Windows 10 需要 build >=19044，并满足当前 Docker Desktop 支持范围 |
| GPU | RTX 3070，8GB；关闭游戏、其他训练/推理任务，至少留出 5GiB 空闲 |
| NVIDIA 驱动 | 安装适配 RTX 3070 的较新正式版 Windows 驱动；本容器 CUDA 12.4.1 建议驱动 >=551.78，优先较新的兼容版本 |
| WSL | WSL2，更新到当前版本；Docker Desktop 要求 WSL 至少 2.1.5 |
| Linux | Ubuntu 22.04 LTS，x86_64 |
| 内存 | Windows 主机至少 16GB；建议 32GB，WSL 分配约 12–16GB |
| 磁盘 | 先预留至少 50GB；完整 profiler 记录可能继续显著增长 |
| 网络 | 能访问 `docker.1ms.run`、魔搭、Ubuntu/NVIDIA apt 和 Python 包源 |

NVIDIA 的 WSL profiling 支持条件是 Windows 11 驱动 >=525，或 Windows 10 build >=19044 且驱动 >=545；这只是 profiling 的下限，不替代 CUDA 12.4.1 对驱动的要求。参考 [Nsight Compute 系统要求](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html) 和 [CUDA 12.4.1 发布说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html)。

## 2. Windows：安装驱动和 WSL2

1. 在 BIOS/UEFI 中开启 CPU 虚拟化（Intel VT-x / AMD SVM）。任务管理器 → 性能 → CPU 应显示“虚拟化：已启用”。
2. 从 [NVIDIA 驱动页面](https://www.nvidia.com/Download/index.aspx) 安装适配 RTX 3070 的 Windows 驱动，按安装提示重启。
3. 打开**管理员 PowerShell**，执行：

```powershell
wsl --install -d Ubuntu-22.04
```

按提示重启 Windows，然后首次启动 Ubuntu，建立 Linux 用户名和密码。如果已经安装 WSL，执行：

```powershell
wsl --update
wsl --set-default-version 2
wsl -l -v
```

输出中 `Ubuntu-22.04` 的 `VERSION` 应为 `2`。若为 `1`，执行：

```powershell
wsl --set-version Ubuntu-22.04 2
```

可随时查看环境：

```powershell
wsl --version
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber
nvidia-smi
```

CUDA on WSL 使用 **Windows 主机驱动**。不要在 Ubuntu 中安装 `nvidia-driver-*`、`cuda-drivers`，也不要通过 `apt install cuda` 拉入 Linux 内核显卡驱动。Docker 路径不需要在 Ubuntu 主机额外安装 CUDA Toolkit。依据 [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

## 3. Windows：开放 GPU 性能计数器

这是本实验额外需要的一步。单纯让 CUDA 可用不会自动开放 profiling 权限。

较新的 NVIDIA App：

1. 打开 NVIDIA App。
2. System → Advanced → Developer → Manage GPU Performance Counters。
3. 授权允许当前实验用户读取 GPU performance counters；若界面只提供管理员/全部用户选项，在这台完全可控机器上选择允许全部用户。
4. 根据提示以管理员身份确认。

使用 NVIDIA 控制面板的驱动：

1. 以管理员身份打开 NVIDIA Control Panel。
2. 菜单 Desktop → Enable Developer Settings。
3. 左侧 Developer → Manage GPU Performance Counters。
4. 选择 Allow access to the GPU performance counters to all users → Apply。

新驱动可能将入口迁移至 NVIDIA App，不必强行寻找旧控制面板菜单。官方步骤见 [ERR_NVGPUCTRPERM 说明](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters)。

设置后在 PowerShell 执行：

```powershell
wsl --shutdown
```

重新启动 Ubuntu。若驱动要求重启，则重启 Windows。此设置只用于你已授权的实验机器。

## 4. 可选：给 WSL 分配内存

在 Windows 用户目录 `%UserProfile%` 下编辑 `.wslconfig`。例如在 32GB RAM 的主机上可使用：

```ini
[wsl2]
memory=16GB
swap=8GB
```

16GB RAM 主机可从 `memory=10GB` 开始，保留 Windows 所需内存。不要给 WSL 分配主机全部内存。配置后执行 `wsl --shutdown` 再重新启动 Ubuntu。WSL RAM 与 GPU 显存是两种不同资源；增加 RAM 无法把 8GB 显存变大。

## 5. Windows：安装 Docker Desktop 并启用 WSL 集成

采用这一条路径即可，不需要同时在 Ubuntu 内安装另一个 Docker daemon。

1. 从 [Docker Desktop 官方安装页](https://docs.docker.com/desktop/setup/install/windows-install/) 安装适配 Windows x86_64 的版本。
2. 启动 Docker Desktop，等待 Engine running。
3. Settings → General：选择 Use the WSL 2 based engine（部分版本默认启用，不显示开关）。
4. Settings → Resources → WSL Integration：开启 `Ubuntu-22.04` 的集成。
5. Apply & Restart。
6. 确认处于 Linux containers 模式。

Docker Desktop 的 WSL GPU 支持不需要在该 Ubuntu 内重复安装 NVIDIA Container Toolkit；如果你选择完全不同的“Ubuntu 内原生 Docker Engine”路径，才按 [NVIDIA Container Toolkit 官方安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 配置那个 daemon。两套 daemon 不要混用。参考 [Docker WSL2 官方说明](https://docs.docker.com/desktop/features/wsl/)。

## 6. Ubuntu：确认 GPU 和 Docker 能互通

下列命令全部在 **Ubuntu 终端**中执行：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git unzip dos2unix python3
uname -m
uname -r
nvidia-smi
docker version
```

应看到 `x86_64`、包含 `microsoft` 的内核版本，以及 RTX 3070。若 `nvidia-smi` 不在 PATH，可以测试：

```bash
/usr/lib/wsl/lib/nvidia-smi
```

再测试容器 GPU。镜像直接使用你指定的源，不需要改全局 Docker registry 配置：

```bash
docker run --rm --gpus all docker.1ms.run/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

容器中也应看到 RTX 3070。这一步验证 GPU 透传，下一步 `doctor` 才验证性能计数器。`nvidia-smi` 的 “CUDA Version” 表示驱动支持的最高 CUDA 版本，不是说 Ubuntu 已安装同版本 Toolkit。

## 7. 将本项目复制到 WSL 的 Linux 文件系统

建议把实验放在 `~/projects`，不要长期直接在 `/mnt/c` 运行 profiler 和模型缓存。假设你已把完整项目复制到 Windows `D:\MoE`，且其中存在 `moe_experiments\real_channel\run.sh`：

```bash
mkdir -p ~/projects
cp -a /mnt/d/MoE ~/projects/MoE
cd ~/projects/MoE/moe_experiments/real_channel
dos2unix run.sh
```

如果你只复制了 `real_channel` 文件夹，将它放到任意 WSL Linux 目录即可运行；主体实验不依赖原目录的已下载模型或旧结果。只有复用旧 `36_channel_pipeline.py` 时才需要完整 `moe_experiments`。

本机 macOS 上的源目录 `/home/user/...` 在 WSL 中不存在，必须先复制文件，再使用复制后的 Linux 路径。

确认文件齐全：

```bash
ls run.sh Dockerfile requirements.txt cli.py adapter.py observer.py victim.py assets.py analyze.py config.py util.py probe.py
ls tests
```

## 8. 先验证计数器，再跑端到端冒烟

在 `real_channel` 目录执行：

```bash
bash run.sh --doctor-only
```

首次会构建镜像：CUDA/Nsight Compute/PyTorch/Transformers 等会安装到容器。后续构建使用缓存。Dockerfile 的基础镜像是 `docker.1ms.run/nvidia/cuda:12.4.1-devel-ubuntu22.04`；Python 包和 NVIDIA apt 包使用各自的软件包源，模型和数据只来自魔搭。

`doctor-only` 会先运行 CPU 测试，再执行一个真实 GPU 正负对照：四个静态专家槽位中只有专家 1 和 3 执行 SiLU，observer 必须从 GPU counter 恢复出 `[1,3]`，并验证四个 anchor 全部存在。成功时输出类似：

```text
[doctor] NVIDIA GeForce RTX 3070: CUDA / NVTX / actual instruction counters / empty-slot controls passed
```

它**不会下载模型和数据集**。若失败，先处理文末对应问题，不要跳过该检查。

然后执行真实小规模完整流程：

```bash
bash run.sh --preset smoke
```

smoke 会自动下载：

- 魔搭模型 `AI-ModelScope/granite-3.0-1b-a400m-base`。
- 魔搭数据 `AI-ModelScope/CodeAlpaca-20k` 和 `AI-ModelScope/databricks-dolly-15k`。

不需要 Hugging Face token，也不应另行把模型改到 Hugging Face 下载。若魔搭不可访问会明确失败。全部成功后应出现 `[complete] .../report.md`，并在对应目录出现 `COMPLETE.json`。smoke 样本极少，只用于功能验证，不应用作论文结论。

## 9. 正式启动与保留日志

正常缩小配置：

```bash
set -o pipefail
bash run.sh --preset quick 2>&1 | tee quick-launch.log
```

默认是一个属性 `code`，train/test 各每类 16 个、观测第 12 层最后 4 个提示词位置、五个训练 seed；none/full 使用相同请求。若要完整采样所有层和提示词位置：

```bash
set -o pipefail
bash run.sh --preset paper 2>&1 | tee paper-launch.log
```

`paper` 每类每 split 150 个请求，全部 24 层和 32 个提示词位置。此配置的 ncu kernel 记录数可能达到数千万级，时间和磁盘成本远高于普通推理；建议先完成 quick，根据真实日志与报告体积规划。采集阶段和攻击者训练阶段分开，GPU 主要用于前者。

可以先采用中等配置：

```bash
bash run.sh --preset quick --layers 6,12,18 --steps 8 --n-train 50 --n-test 50
```

需要跨终端保持运行时，可在 Ubuntu 安装并使用 tmux：

```bash
sudo apt-get install -y tmux
tmux new -s moecho
bash run.sh --preset quick
```

按 `Ctrl+B` 再按 `D` 脱离 tmux；用 `tmux attach -t moecho` 返回。Windows/WSL 停机、休眠或重启仍会中断任务。长实验前在 Windows 电源设置中避免自动休眠，并保持供电。

中断后，在原目录执行：

```bash
ls -1 artifacts
bash run.sh --resume artifacts/实际运行目录名
```

将“实际运行目录名”替换为此前 `[run]` 输出的目录。脚本会校验代码、输入和已完成 shard 的 hash；未完成 shard 重跑。如果代码、输入或硬件配置变了，需要新建实验，不能把不同条件拼接成同一个结果。

## 10. 查看结果与复制回 Windows

```bash
ls -lt artifacts
```

进入最新完成目录，检查：

```bash
cat COMPLETE.json
cat hardware.json
cat report.md
cat validation_gpu.json
```

主要文件：`results.json`（统计结果）、`table.tex`（论文表格）、`attack_accuracy.pdf/png`（图）、`channel_none/full.json`（实测观测）、`profiles/`（原始 ncu 报告）。没有 `COMPLETE.json` 就不算整套流程完成。

Ubuntu 中执行 `explorer.exe .` 可在 Windows 文件管理器打开当前目录。或复制单次结果：

```bash
mkdir -p /mnt/d/MoE-results
cp -a artifacts/实际运行目录名 /mnt/d/MoE-results/
```

`private/requests.json` 含 token IDs/标签，`cache/` 含原始公开数据；公开实验结果时通常只需要派生观测、统计和运行信息，无需随意打包整个缓存。

## 11. 常见问题

| 现象 | 处理 |
|---|---|
| WSL 中没有 GPU | 更新 Windows NVIDIA 驱动；`wsl --update`、`wsl --shutdown`；确认发行版是 WSL2；不要安装 Linux 显卡内核驱动 |
| `docker: command not found` | 启动 Docker Desktop，给 Ubuntu-22.04 开启 WSL Integration，再开新 Ubuntu 终端 |
| Docker 只有 Client、连不到 Server | Docker Desktop 尚未运行或集成未生效；不要为了此问题另起第二套 daemon |
| `could not select device driver ... gpu` | 检查 Docker Desktop Linux containers/WSL2 集成；先让第 6 节容器 `nvidia-smi` 成功 |
| `ERR_NVGPUCTRPERM` | 回 Windows 开放 GPU counter 访问权限；重启 WSL/必要时重启 Windows；本脚本已加 `SYS_ADMIN`，不要仅在 Ubuntu 反复 sudo |
| `Option '--print-metric-name' is only supported for the details page` | 旧代码导出参数不兼容 ncu；更新到 v2 修复包。raw 导出不传 `--print-metric-name`，无需重新配置计数器权限或重建镜像 |
| `NVTX kernel ...无法解析: 'slot_.../void ...'` | v2 只接受纯 NVTX label；更新到 v3，支持 label/原始 CUDA kernel 名称，无需更改 GPU 权限 |
| 无 `Manage GPU Performance Counters` 入口 | 按驱动版本查看 NVIDIA App 的 Developer 入口或以管理员身份使用旧控制面板；参照 NVIDIA 官方权限文档 |
| `driver too old` / `unsupported driver` | 更新 Windows 驱动；不要在 WSL 安装 Linux NVIDIA driver；驱动需要满足 CUDA 12.4.1 要求 |
| 镜像源 403/429/502 或 TLS 错误 | 检查 `docker.1ms.run` 可访问性、限流、Docker Desktop 代理/DNS 后重试；本脚本不会静默换镜像源 |
| `cuda-nsight-compute-12-4` 安装失败 | 查看 Docker build 日志，确认 NVIDIA apt 仓库可访问；不要把 ncu 步骤删掉后继续声称实测 |
| `anchor 不完整` / NVTX 重命名失败 | 查看 `profiles/.../*.log`；保留原始 CSV；重跑该未完成 shard；不要把丢失行当成未激活专家 |
| CUDA OOM | 关闭占用显存的 Windows/WSL 程序，检查 GPU 空闲；仍失败请保留日志，不能用静默 CPU offload 代替所声明的 GPU 实验 |
| smoke 校验出现 NaN/Inf，或 96.9% logits 不匹配 | 更新 v4：默认 BF16，显式 DynamicCache，匹配 stock 的 index_add；不要增大容差或用 nan_to_num 掩盖问题，见 validation/BF16_CACHE_FIX.zh-CN.md |
| 容器被 OOM killed / exit 137 | 增加 WSL RAM/swap、释放主机内存，确认 Docker 镜像构建和模型加载的 host RAM 足够 |
| `bash\r` / `$'\r'` 错误 | `dos2unix run.sh`，使用 LF 换行 |
| artifacts/cache 归 root | 若发生中断导致所有权未恢复，可执行 `sudo chown -R "$USER:$(id -gn)" artifacts cache` |

WSL 支持 profiling 不意味着每套驱动/虚拟化组合都已被这份代码实机验证；这里保留真实 doctor 检查作为运行门槛。若升级驱动并按官方步骤开放权限后仍不支持 counter，原生 Ubuntu 是可用的对照实验环境，不能把 WSL 普通运行时延包装成该性能计数器复现。
