# 2026-10-25：v3 修复 NVTX/kernel 组合名称解析

用户的 RTX 3070 / WSL 已成功导出 profiler CSV。该版本的 ncu 将名称打印为：

```text
slot_R000000_P000_L000_E000/void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, at::detail::Array<char *, (int)2>>(int, T2, T3)
```

v2 的 parser 要求整个名称严格等于 NVTX label，因此错误拒绝了该合法记录。v3 根据报告中实际出现的名称形态修复 parser。

## 行为

- 同时接受 `label`、`label/CUDA kernel` 和带已知外层范围的 `moecho/label/CUDA kernel`。
- 只解析位于名称开头的公开调度 label。CUDA kernel 的模板参数、逗号、空格及其他符号作为不透明后缀保留。
- 仍拒绝畸形坐标、任意前缀中夹带的 label、空 kernel 后缀和多个坐标造成的歧义。
- 同一个 launch ID 的不同指标行，必须保有一致的完整 kernel 名称；继续检查指标值、重复记录及全部 anchor。
- 不修改 GPU 采集、模型、路由、标签或性能计数器权限。

名称兼容回归使用了用户报告中的完整 CUDA kernel 字符串，并覆盖 long/wide CSV；指标数值是测试 fixture，不是 RTX 3070 实测结果。独立子进程导出契约测试也新增了带 kernel 后缀的 CSV。

针对性回归 **49 passed in 0.66s**，完整回归 **71 passed in 12.12s**；实际输出见 `nvtx_name_targeted.log`、`nvtx_name_regression.log`。运行环境仍为本机 CPU，使用 Transformers 4.51.3，以及本机 PyTorch 2.11.0；不声称已在本机完成 GPU 实验。

## 更新

v2 用户只需替换 `observer.py` 即可修复当前问题；完整 v3 包还包含新增测试与说明。Python 源码由容器 bind mount 加载，不需要重建依赖镜像。保留已有 `cache/` 和 `artifacts/`，更新后正常执行：

```bash
bash run.sh --doctor-only
```

若要跳过 Docker build，直接用已有镜像：

```bash
docker run --rm --gpus all --cap-add=SYS_ADMIN --shm-size=2g \
  -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
  -v "$PWD:/workspace" moe-real-channel:cu124-v1 --doctor-only
```

旧的 CSV 和 .ncu-rep 不需删除。当前失败目录发生在 doctor 阶段，不使用 `--resume`；重新预检将建立新目录。
