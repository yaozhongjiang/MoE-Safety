# docker.1ms.run CUDA manifest 拉取失败

这次失败发生于 Docker 构建的 FROM 元数据解析阶段，实验 Python 和 GPU 采集尚未启动。日志说明镜像源请求的某个 manifest digest 返回 not found；不能仅据此确定是源端缓存、上游变化还是平台索引问题。

原 `run.sh` 每次执行 `docker build --pull`，会强制尝试拉取引用的镜像。即使以前已经成功构建了 `moe-real-channel:cu124-v1`，这次也可能在访问源时失败。Docker 文档：[build --pull](https://docs.docker.com/reference/cli/docker/buildx/build/#pull)。

## 已经运行过本实验：先复用本机完整镜像

在 WSL Ubuntu 中，进入原来运行 `bash run.sh` 的目录，先检查：

```bash
docker image inspect moe-real-channel:cu124-v1 --format '{{.Id}}'
```

有正常 image ID 输出，表示可以直接使用。此次构建失败不代表旧镜像被删除。

把 `real_channel_cached_image_fix.zip` 放到这个能看到 `run.sh`、`cli.py` 的原目录，执行：

```bash
unzip -o real_channel_cached_image_fix.zip -d .
set -o pipefail
bash validation/run_cached_image.sh --preset paper --shard-requests 4 \
  2>&1 | tee paper-cached-launch.log
```

ZIP 根目录直接是 `validation/`，不会多出一层 real_channel。该命令新建完整 paper 实验，默认输出为 `artifacts/<UTC时间_后缀>/`，启动日志中的 `[run]` 会给出实际路径。这个入口使用原 CLI，所以输出名没有加 `paper_fast_` 前缀；实际分片大小仍是命令指定的 4。

如果目的是继续原实验，则使用以下命令，替换掉上面的新建命令：

```bash
bash validation/run_cached_image.sh \
  --resume artifacts/20261025T222451Z_c60ddd
```

恢复从原 config.json 读取参数，不会把旧实验改成分片 4。之前的有限驱动错误恢复工具也会复用现有镜像，可照常使用。

该入口不构建镜像，也不拉取镜像。`docker run --pull=never` 使镜像缺失时明确失败：[Docker pull policy](https://docs.docker.com/reference/cli/docker/container/run/#pull)。模型和数据仍由原实验代码处理，已有缓存照常复用；“不访问镜像仓库”不表示实验模型/数据下载也被禁用。

## 本机镜像不存在时

执行并保留输出：

```bash
docker image ls --digests
docker context show
```

核对当前 Docker context 是否就是之前运行实验的 daemon，查看原镜像是否换了 tag。若另一个 daemon 或原机器上仍有完整镜像，可以在该处用 `docker save -o moe-real-channel-cu124.tar moe-real-channel:cu124-v1` 导出，拷贝 tar 后在目标 daemon 执行 `docker load -i moe-real-channel-cu124.tar`。先确认空间足够，这个文件会较大。

如果没有任何完整镜像或备份，必须先恢复基础镜像的可获取性。可用以下单独的拉取测试区分构建与拉取：

```bash
docker pull --platform linux/amd64 docker.1ms.run/nvidia/cuda:12.4.1-devel-ubuntu22.04
```

它仍使用用户指定的源和原 CUDA 版本。显式平台不是源端 manifest 缺失的保证修复；若仍返回 not found，需要镜像源提供方检查该 tag/digest，或等待该源恢复。在无缓存且源不可用时，本补充脚本不能凭空生成镜像。

保留现有缓存及镜像，不执行 prune，不任意切换 CUDA 版本。更换运行时还可能使原实验的硬件身份校验拒绝恢复。

## 安装与验证

补充脚本放在 validation 下，不改变原核心指纹，仍为 `83bf6a5cab9f49243ffbbd4a07b78141d76fa63583ae3a6a2d6ac5b81d2c4e54`。使用假 Docker 可执行文件测试已有镜像、不存在镜像、daemon 错误、含空格路径、参数转发及 pull=never 契约；这不代表已在开发机测试 RTX3070/Docker daemon，也不代表远端镜像源已修好。
