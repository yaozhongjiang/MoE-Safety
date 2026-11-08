# 三轮检查记录与实机验收边界

本代码在 macOS arm64 上开发。本机无 NVIDIA CUDA GPU。以下检查均为实际执行的结果；没有生成或填写任何真实 GPU 实验精度、延迟或信道保真度数字。

## 第一轮：实现和 CPU 功能回归

命令：`python -m pytest -q tests`。

- 结果：**36 passed in 18.37s**，见 `round1_cpu.log`。
- 环境：Python 3.13.12、PyTorch 2.11.0、Transformers 5.5.0。
- 覆盖：真实 Granite 小模型的 stock/adapter 输出、none/full 位相等、专家 GEMM 实际调用数、NVTX 静态标注不依赖路由；CSV 格式、counter 非法值、空文件、缺失 anchor、重复 shard；三类属性攻击；balanced split、用户/输入窗口不重叠；魔搭下载与 hash 的离线契约。
- 对源码的额外接口检查发现：Transformers 4.51 的 GraniteMoE 返回 `(hidden, router_logits)`，5.5 返回 `hidden`。已修复为在 profile 区域之外检查实际 stock 返回接口，并保留原契约。

## 第二轮：锁定版本兼容性与外部资源实测

命令：将 `transformers==4.51.3`、`tokenizers==0.21.1`、`huggingface-hub==0.30.2` 安装在独立临时目录，使用该目录的 PYTHONPATH 运行测试。未改动用户原 Python 环境。

- 结果：**36 passed in 14.89s**，见 `round2_transformers451.log`。
- PyTorch 仍使用本机 2.11.0；不是容器中的 CUDA PyTorch 2.5.1。两者不能视作相同运行环境。
- 实际联网下载并逐条解析魔搭数据：CodeAlpaca **20,022** 条、**7,030,293 bytes**；Dolly **15,011** 条、**13,085,339 bytes**。
- 验证模型 config：`granitemoe`、E=32、K=8、L=24。没有下载 2.7GB 模型权重。
- 验证 WildChat 魔搭镜像首分片元数据存在：`data/train-00000-of-00014.parquet`。未下载/解析该分片，不能将其标作数据端到端已验收。
- 各文件的 immutable commit、实际 payload SHA-256 与条数见 `sources_verified.json`；可用 `python validation/check_sources.py` 重做联网检查。
- 两个尝试的 ChatDoctor 魔搭路径返回 404。没有在默认可运行路径中引用它们，也没有静默换成其他医学数据集。medical 的采样和分析代码可通过已可访问的魔搭镜像配置启用。
- 执行 `docker build --check --platform linux/amd64 .`：通过指定的 `docker.1ms.run` 加载了基础镜像元数据，结果 **Check complete, no warnings found.** 这是 Dockerfile 构建规则检查，**不是完整 Docker 镜像构建或运行**。

## 第三轮：端到端逻辑与故障恢复复查

命令：继续使用锁定的 Transformers 4.51.3，执行 `python -m pytest -q tests`。

- 结果：**41 passed in 11.66s**，见 `round3_final.log`。
- 新增覆盖：非零退出和 profiler 错误不能被误报为成功；完成 shard 可恢复，文件损坏必须拒绝恢复；独立 counter 观测与矛盾 truth 合并后，恢复值仍由 counter 决定；ncu 命令包含真正的指令 counter、公开 range 过滤和 NVTX kernel rename 参数。
- 同时执行 Python compileall、Bash `-n` 语法检查。
- 完整 paper 配置改为每个 shard 一个请求，以限制 profiler 单份报告的内存占用；最终封装前另行核验了三个 preset 的配置边界和 Python/Bash 语法。
- 完整报告生成测试使用明确标记为 `TEST_FIXTURE_ONLY` 的固定输入，输出放在 pytest 临时目录；这些图表不属于硬件实验结果，不随代码包作为实测结论分发。

## 尚须在目标 WSL/RTX 3070 上完成

1. 构建完整 Linux x86_64 Docker 镜像并安装所列依赖。
2. `bash run.sh --doctor-only`：真实 CUDA、NVTX、GPU 指令 counter、空槽对照；必须恢复 `[1,3]`。
3. `bash run.sh --preset smoke`：真实 FP16 权重加载、8GB 显存验收、stock/none/full 输出验收、原始 ncu 报告导出/解析、完整属性分析。
4. 再运行 quick/paper，获取可报告的实验数字。没有 `COMPLETE.json` 不视作整套运行成功。

三轮软件校验能降低引入 bug 的风险，但不能证明任意驱动、虚拟化环境和 GPU 上都不存在问题。代码把关键硬件条件设计成必须通过的检查，不在不满足条件时伪造结果或回退成模拟观测。
