# docker.1ms.run 的 CUDA 清单 not found：启动与构建入口

错误发生在 Docker 获取基础镜像元数据阶段，GPU、模型和实验尚未启动。日志中缺失的 `da679129...` 是 CUDA 12.4.1 devel / Ubuntu 22.04 的多架构索引。

2026-11-07 检查时，该 tag 和索引均返回 HTTP 200，并声明 linux/amd64 子清单为 `sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749`；该子清单也返回 HTTP 200。检查期间匿名认证服务还出现过一次超时。这些证据说明当时检查节点能够读取清单，不能保证用户所在网络的镜像节点始终正常，也不能据此断言原标签被删除。清单检查没有下载全部 11 个镜像层。

原 `run.sh` 总是 `docker build --pull`，即便本地已有完整实验镜像，也再次访问基础镜像源。本补充包提供两个独立入口，不编辑原核心文件。

## 安装：只有 validation 一层

把 `real_channel_image_recovery.zip` 放到原来执行 `bash run.sh` 的目录。在这个能看到 `run.sh`、`cli.py`、`artifacts/` 的目录执行：

```bash
unzip -o real_channel_image_recovery.zip -d .
ls validation/build_image_1ms.sh validation/run_cached_image.sh
```

补充包内直接是 `validation/`，解压后不应产生额外的 `real_channel/`。

## 构建或直接复用已有镜像

```bash
bash validation/build_image_1ms.sh
```

有本地 `moe-real-channel:cu124-v1` 时直接返回，不 build、不 pull。没有时，使用独立的 `validation/Dockerfile.1ms_amd64` 构建：同一个 CUDA 12.4.1 devel / Ubuntu 22.04 镜像，显式指定 linux/amd64 并直接引用上述子清单；其余 Dockerfile 内容逐行保持一致。源仍然是 `docker.1ms.run`。

只绕过多架构索引解析，不承诺解决镜像源的匿名认证超时、层下载故障或全部网络问题。构建失败会明确退出，不会改用其他源或 CUDA 版本。apt 和 pip 的获取方式与原 Dockerfile 一致。

## 新建完整的 paper 实验，每分片 4 请求

```bash
set -o pipefail
bash validation/run_cached_image.sh --preset paper --shard-requests 4 \
  2>&1 | tee paper-launch.log
```

结果按原 CLI 保存到 `artifacts/<UTC时间_随机后缀>/`，启动输出会打印 `[run]` 的实际路径。全部完成后才生成 `COMPLETE.json`。

## 继续原来的实验

需要继续 `artifacts/20261025T222451Z_c60ddd` 时，选择以下命令，而不是再新建一份实验：

```bash
set -o pipefail
bash validation/run_cached_image.sh \
  --resume artifacts/20261025T222451Z_c60ddd \
  2>&1 | tee paper-resume.log
```

恢复沿用旧 config，不能把旧分片大小改成 4。原代码会继续检查代码指纹、输入、模型、GPU/驱动/ncu/运行时身份和已完成分片 hash。若原镜像已经丢失，重新构建的 ncu/运行时与旧记录不同，原校验会停止，不能绕过它把两个环境的数据混合。

已有 `validation/resume_guard.py` 时，也可继续使用它恢复原实验；它同样直接使用本地实验镜像，不 build/pull。

## 后续使用哪个启动入口

后续直接使用 `validation/run_cached_image.sh`，它带 `docker run --pull=never`。缺少本地实验镜像时会明确提示先运行构建入口，绝不隐式拉取。

原 `run.sh` 保持原样，因此它依旧会强制 build/pull。不要用旧入口重新触发同一问题。这样处理是为了保持旧实验的核心代码指纹。

## 检查范围

新增 shell 入口用模拟 Docker 子进程测试：覆盖本地镜像复用、缺失时明确失败、doctor/新建/resume 参数透传、按 amd64 子清单构建、已有镜像不重建，以及恢复 Dockerfile 与原依赖内容一致。开发机没有完成实际 Docker 镜像层下载、构建或 RTX 3070 GPU 测试。

所有新增文件位于 `validation/`；原 v4 指纹仍为 `83bf6a5cab9f49243ffbbd4a07b78141d76fa63583ae3a6a2d6ac5b81d2c4e54`。
