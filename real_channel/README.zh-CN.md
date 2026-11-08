# RTX 3070 8GB：受控 MoEcho GPU 性能计数器实验

v4 将默认精度改为 BF16，显式使用 DynamicCache，并恢复 stock 模型的 index_add 求和方式。已用真实 Granite 权重验证两条 32-token 输入的 stock/none/full 输出在 CPU 上逐位一致。详见 [v4 数值修复记录](validation/BF16_CACHE_FIX.zh-CN.md)。同时包含 [v3 名称解析](validation/NVTX_NAME_FIX.zh-CN.md) 和 [v2 raw CSV 导出修复](validation/NCU_RAW_FIX.zh-CN.md)。

Windows 用户先按 [WSL_SETUP.zh-CN.md](WSL_SETUP.zh-CN.md) 配置 WSL2 Ubuntu 和 Windows 主机的 GPU 性能计数器权限。

环境就绪后，在本目录执行一条命令即可开始：

```bash
bash run.sh
```

默认会构建 Docker 镜像、执行 CPU 回归测试、在真实 GPU 上做计数器正负对照、从魔搭下载模型和数据、验证防御输出、采集无防御及全覆盖信道、训练攻击者并生成结果。`run.sh` 和 Dockerfile 使用 `docker.1ms.run` 镜像源。没有模拟信道回退。

## 实验与论文的关系

实现的是 [MoEcho §4.2.1 GPU Performance Counter](https://arxiv.org/html/2508.15036v1#S4.SS2.SSS1) 的**受控、NVTX 辅助的移植实验**：从实际 NVIDIA Nsight Compute 的激活核线程数和执行指令计数观测专家是否执行，再做属性推断。论文原工作用 DeepSeekMoE；这里使用本项目论文也评估的 Granite 模型，以适配 8GB 显存。

**GateBleed 使用 Intel AMX 电源门控信道，RTX 3070 无此硬件，不能复现。** 本实现也不声称复现 MoEcho 的 CPU cache/pageout 路径、GPU TLB Evict+Reload 路径、提示词逐字恢复或原论文全部攻击。依据：[GateBleed 原文](https://arxiv.org/html/2507.17033v1)。

威胁模型是：自有单 GPU、自有 victim，允许 profiler 附加和性能计数器访问，专家位置通过公开静态调度标注识别。它比无标注、独立进程的黑盒共租户观察更强。所有报告明确标记 `annotation_assisted=true`，不能直接写成“无权限共租户攻击已实现”。

`slot_R..._P..._L..._E...` 和 `act_R..._P..._L..._E...` 标注对**每个**专家无条件发出，包含公开的请求序号、位置、层、专家编号。无激活专家也发出同样的标注和一个固定 anchor CUDA 核。标注中没有真实路由、类别或 token 内容。恢复程序只读 `public.json` 和 ncu CSV；它必须先验证全部 anchor 存在，再按激活核的正指令计数恢复集合。没有采集完整的文件会失败，不会被当成“专家未执行”。

全覆盖对每个真实 token **实际执行全部 32 个专家**，只把原始 top-8 的加权输出加入结果。单 token GEMM 形状、top-k gating、加权乘法和 index_add 求和方式在 none/full 中相同；实验启动时检查 logits 位相等、NaN/Inf，同时与 stock 模型做数值和 greedy 输出校验。该保证针对当前运行的预检样本，不代替目标模型全部输入的形式化浮点证明。CPU 回归还对 FP32/BF16/FP16 分别计数真实专家矩阵乘调用，排除只伪造覆盖日志。

只评估专家执行**集合**信道。其他 CPU 分支、输出合并时间、attention、KV cache、总时延等不受“集合全覆盖”保证。这里的 eager 调度不是 vLLM 的 fused MoE 内核；独立耗时实验会标注这个区别。

## 资源与默认设置

| 项目 | 默认值 |
|---|---|
| 模型 | 魔搭 `AI-ModelScope/granite-3.0-1b-a400m-base`，BF16，不量化、不卸载 |
| 架构 | 24 层，32 专家，top-8；下载后从 config 再核对 |
| 数据 | 魔搭 `AI-ModelScope/CodeAlpaca-20k` vs `AI-ModelScope/databricks-dolly-15k` |
| 观测 | 每条提示词前 32 token 中最后 4 个位置；零起始第 12 层 |
| 样本 | train/test 各每类 16，合计 64 请求，成对执行 none/full |
| 攻击者 | mean-pool MLP、DeepSets、带位置的 Transformer；分别在每种防御下重新训练，5 seeds |
| 指标 | recall/FPR/precision，accuracy/std，按测试用户/样本分组 bootstrap 95% CI，标签打乱对照 |
| 显存 | 要求 >=7 GiB 总显存、>=5 GiB 当前空闲；实际峰值写入结果 |

BF16 权重常驻显存约 2.7GB，与 FP16 同为每参数 2 字节；下载的源权重文件可能更大。额外还需要 CUDA runtime、KV cache、profiler 等空间。是否足够由目标机实际预检及运行决定，不用 CPU offload 掩盖显存不足。默认要求 GPU 支持 BF16，不会静默回退 FP16。建议 Windows 至少 16GB RAM，32GB 更宽裕；磁盘为缓存/镜像/报告预留至少 50GB，完整配置按实际增长扩容。

数据按**输入的 token 窗口**去重后划分；共享窗口不会出现在 train/test 两侧。WildChat 任务另外按 `hashed_ip` 分组划分，重复窗口跨用户时丢弃。样本长度固定，不做类别模板包装。`code` 的结果包含来源数据集风格的影响，不能解释成剔除了数据来源混杂的纯语义分类。

## 启动命令

```bash
# 真实 GPU 的最小 counter 测试，不下载模型/数据
bash run.sh --doctor-only

# 最小端到端冒烟：每类 train/test 各 2 个、两个位置、一个层
bash run.sh --preset smoke

# 默认：完整流程的缩小采样配置
bash run.sh --preset quick

# 完整采样：每类 train/test 各 150 个；全部 24 层、32 个提示词位置
bash run.sh --preset paper

# 中等规模，自定义观测层和位置数
bash run.sh --preset quick --layers 6,12,18 --steps 8 --n-train 50 --n-test 50

# 只跑 CPU 回归测试
bash run.sh --test-only

# 完成的 shard 按 hash 校验后跳过；半截 shard 自动重跑
bash run.sh --resume artifacts/实际运行目录名
```

`paper` 可生成大量 kernel 记录，远慢于普通推理。Nsight 会序列化/重放 kernel；不提供未经目标机计时的耗时承诺。先运行 smoke 并查看其报告大小与时间，再规划完整运行。WSL、原生 Linux 的结果分别标注平台，不能混用为同一机器配置。

支持在已有 Python 3.10/3.11 + CUDA + Nsight Compute 的 Linux/WSL 环境直接执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run.sh --local --doctor-only
bash run.sh --local --preset smoke
```

本地模式需要用户能读取 GPU counter；Docker 模式的 `SYS_ADMIN` 能力不能替代 Windows 主机计数器开关。Docker 需要已经配置好的 GPU runtime；本脚本不自动改主机驱动和系统服务。

## 其他属性任务与数据配置

已核验魔搭 `allenai/WildChat-1M` 镜像存在；选择 `nationality` 或 `sensitive` 会只取首个 `00000` parquet 分片，不下载整个数据集：

```bash
bash run.sh --preset quick --tasks code,nationality,sensitive
```

可用样本不足会明确失败；不会更改标签阈值、替换样本或静默缩小样本数。WildChat nationality 标签为美国 0、俄罗斯/中国/摩洛哥/德国/印度 1；sensitive 为原数据 moderation 的五类最大 score <.01 或 >.1，中间区间丢弃。这里的地理元数据标签与真实国籍不是同一概念，与本项目已有定义保持一致但应在论文中说明。

**当前核验的两个常见 ChatDoctor 魔搭候选地址都返回 404，因此没有把不存在的下载链接写成默认任务。** medical 代码已实现，但必须提供可访问且内容相同的魔搭镜像，才能复现论文该项数据定义。通过 `--dataset-map 文件.json` 指定 `medical` 的 `repo`、`file`、可选 `revision`；JSON 键为 `medical`，值的格式和 `assets.py` 的 `code` 配置一致。支持 ChatDoctor `input` 列的 JSON、JSONL 或 parquet。也可覆盖其他来源文件，所有下载仍限定在魔搭 API，并记录实际文件 SHA-256 和修订号。不要把不同医学数据集的结果标成原 ChatDoctor 结果。

## 结果位置与复用

每次新运行建立 `artifacts/<UTC时间>_<随机后缀>/`，避免覆盖历史实验：

| 文件 | 用途 |
|---|---|
| `COMPLETE.json` | 所有阶段成功才生成的完成标记及输出 hash |
| `hardware.json` | 实际 GPU、驱动、CUDA、PyTorch、ncu、WSL 标记 |
| `assets.json` | 模型文件 SHA-256，魔搭数据集文件 hash/commit |
| `public.json` | 公开测量计划，无 prompt、标签或路由 |
| `private/requests.json` | 仅本地 victim 使用的 token IDs、split、标签 |
| `private/truth_*.json` | 独立真值；observer 不读取 |
| `profiles/{none,full}/*.ncu-rep` | 原始真实 profiler 报告，可用 ncu-ui 审核 |
| `profiles/{none,full}/*.csv` | 机器可读 counter 记录，保留原始 kernel ID |
| `channel_none.json`, `channel_full.json` | 实测集合及独立评分真值，兼容脚本 36 的核心格式 |
| `validation_gpu.json` | stock/none/full 输出校验、非 profiler 耗时与显存 |
| `results.json` | 三种 adaptive attacker、五 seed 结果及置信区间 |
| `report.md`, `table.tex` | 可检查的文字和表格 |
| `attack_accuracy.{png,pdf}` | 可插入论文的实验图 |

channel 文件的 `L` 是**观测层数**，原始层号保存在 `original_layer_ids`；集合是观测位置的并集，token 观测另存于 `recovered_tokens`。默认 quick 的单层/最后四个位置结果不能标成全层/全提示词结果。全覆盖没有未执行专家，因此其 FPR 分母为零，输出 `null`，不会伪造 0% FPR。

如需复用现有 `36_channel_pipeline.py`，在 `moe_experiments` 中使用其 `run` 命令，输入这两个 JSON 的绝对路径。它分析每请求专家集合的并集，和这里默认的 token 级自适应攻击特征不同，应分别报告。建议以本目录的 `results.json` 为该新增实验主结果，以原脚本做补充交叉校验。

代码完成后的三轮本机检查记录在 [validation/VALIDATION.zh-CN.md](validation/VALIDATION.zh-CN.md)。本开发主机为 macOS，无 NVIDIA CUDA；本机通过的 CPU/静态/数据源校验不等于 RTX 3070/WSL 的实机验收。运行目标机的 `--doctor-only` 和 `--preset smoke` 后才有真实 counter 与 GPU 输出证据。
