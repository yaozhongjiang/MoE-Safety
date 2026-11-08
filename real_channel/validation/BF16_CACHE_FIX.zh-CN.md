# v4：真实 Granite 权重的 FP16 NaN、Cache 和低精度求和修复

用户在 smoke 的 stock/adapter 输出校验阶段遇到 NaN。v4 没有放宽原有 `rtol=1e-2, atol=1e-2`，没有开启 `equal_nan`，没有用 `nan_to_num` 或其他方式替换无效值。

## 实际复现

本次不只使用随机小模型测试，还使用本机已有的完整 Granite-3.0-1B-A400M 权重做离线数值检查。开发机为 macOS CPU；读取的是用户本机先前已缓存的官方 checkpoint，没有新增 Hugging Face 下载。实验运行程序仍只通过 ModelScope 获取模型与数据。

环境：PyTorch 2.11.0、Transformers 4.51.3，CPU，真实缓存 checkpoint 的 config SHA-256 与具体路径均记录在 `real_model_numerics.json`。

- 原始 stock 模型采用 FP16、显式 DynamicCache：前四个位置的有限性为 `[true,false,false,false]`，后三个位置各有 49,152 个 NaN。这表明 NaN 在未安装 adapter 时就会发生，legacy tuple 警告不是它的必要条件。
- 使用原 checkpoint **重新加载 BF16**（不是把已舍入成 FP16 的参数再转换为 BF16）。两条固定诊断输入各取前 32 token：stock、none、full 均为有限值；四个 adapter 对照的最大绝对误差均为 **0.0**，greedy 输出一致；none/full logits 逐位相等。
- 实际输出见 `real_model_numerics.json` 和 `real_model_numerics.log`。这是数值验证，不是 GPU 信道保真度或属性攻击准确率结果。

可在任意已下载相同 checkpoint 的环境离线重做：

```bash
python validation/check_model_numerics.py --model-path /实际本地模型目录 --device cpu
```

该可选检查会加载完整权重两次，需要主机内存与运行时间。目标 RTX 3070 仍通过正常 smoke 做 CUDA 验收。

## 修复内容

1. 默认权重 dtype 从 FP16 改为 BF16，BF16/FP16 都是每参数 2 字节。精度明确写入 config、hardware 和 validation_gpu；GPU 不支持 BF16 时直接说明并失败，不做静默回退或 CPU offload。`--dtype float16` 仅供显式诊断，本模型已确认存在 FP16 非有限值风险。
2. 每个请求创建新的 DynamicCache，并显式传入单 token 的 cache_position。检查模型确实返回 Cache 实例；消除首次传 None 导致的 legacy tuple 往返与警告。参考 [锁定版本的官方模型实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/granitemoe/modeling_granitemoe.py)。
3. 适配器保留原模型的加权乘法与 index_add 归约。旧版逐专家 add_ 在低精度下可能逐步舍入，与 stock 的归约实现不同。full 仍实际计算每个专家；只有原始 top-k 的输出参加与 stock 相同的最终合并。
4. stock、none、full 在比较前以及采集请求结束后检查 logits 的 NaN/Inf；错误包含阶段、dtype 与异常位置，不输出提示词内容。独立 latency 测量不包含此诊断检查。
5. 模型加载时打印实际 torch/transformers 版本、设备和 dtype，并写入验证 JSON，便于识别运行环境。

## 软件回归

- 模型/数值针对性回归：**20 passed in 2.58s**，见 `bf16_targeted.log`。涵盖三个 seed、FP32/BF16/FP16 的真实专家调用计数、Cache 请求隔离与位置增长、缓存增量推理与 stock prefill 对照、stock NaN 必须先于 adapter 检查失败、BF16 不支持时的早期错误。
- Transformers 4.51.3 完整回归：**86 passed in 11.64s**，见 `bf16_regression.log`。
- Transformers 5.5.0 的模型/数值兼容性回归：**20 passed in 2.91s**，见 `bf16_transformers55.log`；生产依赖仍锁定 4.51.3。
- Python/Bash 语法、ZIP CRC 和解压后的逐文件 SHA-256 在重新封装时检查。

开发机无 NVIDIA GPU。本次真实权重 CPU 数值结果比之前的随机小模型回归更有针对性，但不代表已经完成目标 RTX 3070/CUDA/PyTorch 2.5.1 的实机验收。

## 更新与重跑

覆盖 v4 包中的代码后，正常运行 `bash run.sh --preset smoke`。依赖版本没有变，不需要重新安装 Python 包或重新下载模型权重。现有缓存和旧日志保留。

此前失败发生在验证阶段，不使用该目录执行 `--resume`，正常启动新运行。完整 paper 运行只在新的 smoke 成功后启动。
