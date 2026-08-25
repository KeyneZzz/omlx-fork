# DFlash2 单会话长上下文内存优化：技术总结与使用指导

- 日期：2026-08-20
- 适用环境：Mac M4 Pro 48GB（macOS，`iogpu.wired_limit_mb=40000`，GPU 上限 ≈39.06GB）
- 模型：Qwen3.8-27B-4bit（MLX 4bit）+ Qwen3.8-27B-DFlash2 draft
- 软件：oMLX app（`/Applications/oMLX.app`）+ 本 fork（`omlx-fork`），DFlash2 speculative decoding 引擎
- 配套文档：`docs/kv-prealloc-plan.md`（实现计划，含设计推导）

## 1. 概述

针对**单实例、单会话驻留服务**的使用模式，本 fork 在不改动 app 内 `dflash_mlx` 代码的前提下，
通过运行时 monkeypatch（`omlx/patches/dflash_kv_prealloc.py`）实现了 llama.cpp 式的
**KV 容量预分配**，把内存行为从"随上下文增长 + 不可控瞬态"改造为：

> **启动即固定底座（~26GB），运行中窄带波动，峰值封顶 ~35GB，全程零卸载。**

成果（48GB 机器，Qwen3.8-27B-4bit + DFlash2）：

| 指标 | 优化前（stock 增长式） | 优化后（容量 163840） |
|---|---|---|
| 稳定可用上下文 | ~100K（真峰值 42.6GB 超 GPU 上限） | **163K 实测通过** |
| 峰值内存 @163K | 不可达（中途触发卸载 / Metal OOM） | **34.99GB**（距上限 4GB） |
| 稳态内存 | 15.7GB 起步，+85KB/tok 增长 | 底座 ~26GB，+20~23KB/tok |
| 超容量请求 | 运行中爆内存 → 卸载重建（23min+） | **请求入口即拒（HTTP 400）** |
| 会话切换首轮峰值 | ~37.3GB | ~27.3GB |
| 输出速度（散文/代码） | 15 / 30 tok/s | **15.4 / 30.0 tok/s（零回退）** |

一句话：**用"预分配换确定性"**——启动时按声明的容量一次性分好 KV，运行期不再有增长拷贝，
内存画像可预测，剩余内存（~10GB+）可放心用于其他应用。

## 2. 问题与根因（优化前）

### 2.1 三个瞬态来源

稳态占用之上的瞬态峰值（100K 时实测 +15GB，内核 `phys_footprint_peak` 42.58GB）：

| 组件 | 机理 | 位置 |
|---|---|---|
| KV 增长拷贝 | `KVCache.update_and_fetch` 容量不足时 `mx.concatenate`，旧+新+结果三份数组短暂共存，释放后滞留 MLX 分配器池 | `mlx_lm/models/cache.py` |
| 分数物化 | head_dim=256 不在 MLX 融合 SDPA 支持集内，每个 prefill chunk 每层物化 `[heads, chunk, ctx]` 分数矩阵 | `dflash_mlx/engine/`（自定义 GQA SDPA） |
| 池滞留 | 已释放缓冲滞留 MLX cache pool，footprint 不回落 | 实测卸载重建后仍 +9.4GB |

### 2.2 连锁故障

峰值超过 39.06GB GPU wired 上限后：

1. 内存 enforcer 触发 → 排空请求后**卸载模型**；
2. L1 前缀快照随实例一起消失 → 下一轮全量重算（113K 冷重建实测 **23 分钟**）；
3. 更糟路径：Metal OOM（`kIOGPUCommandBufferCallbackErrorOutOfMemory`）直接打断生成。

即优化前服务在 ~100K 上下文附近**不可用**，且故障后恢复代价极大。

### 2.3 隐藏根因：快照逻辑切片导致容量预分配失效

实现预分配的第一版验证中发现（通过 realloc/序列化插桩日志）：

- 下一轮请求的 chat template 会**重新分词**上一轮的 assistant 回复，token 序列与生成时不完全一致；
- 于是 generation 快照（全前缀匹配）失效，命中走 **sidecar 边界切片路径**，切片产生的是
  **逻辑长度视图**（非物理容量数组）；
- 视图在下一次 `update_and_fetch` 时触发"增长"→ 为容量重新分配一整份 10GB 缓冲 →
  **两份容量缓冲共存**（实测 +11GB 异常）。

结论：预分配要成立，必须保证**物理容量数组在快照/hydrate/切片全链路零拷贝传递**。

## 3. 优化方案

### 3.1 设计原则

- 单会话语义下，同一时刻只需要**恰好一份**容量 KV 缓冲；
- 快照是 KV 跨请求延续的唯一载体（L1 开销 ~156MB，可忽略），因此把它改造成容量缓冲的
  零拷贝传递通道，而不是绕过它；
- 全部改动以**运行时补丁**形式挂在 fork 侧（app 内 `dflash_mlx` 只读），补丁在引擎
  start 时安装、teardown 时恢复，`dflash_kv_capacity_tokens=0`（默认）时完全走原逻辑，
  对其他引擎零影响。

### 3.2 KV 容量预分配（P1）

补丁 `mlx_lm.models.cache.KVCache.update_and_fetch`：

- 首次分配：直接 `mx.zeros` 到声明的容量（B, H, capacity, D），`mx.eval` 提交；
- 容量内：原地 slice 写入，零分配；
- 超容量：`RuntimeError`（正常情况下被 3.5 的结构性准入提前拦截）。

类级补丁同时覆盖 target 的 16 层 FA KV（64KB/tok）与 draft 全上下文 KV，一次到位。

### 3.3 快照零拷贝链（P2 / P3 + builder + sidecar 切片）

保证容量数组全链路按引用传递：

| 环节 | 补丁 |
|---|---|
| 序列化 | `codecs.serialize_target_cache` 的 adopt 路径（`clone=False`）直接取 `entry.keys/values` 全数组引用 |
| 水化 | `codecs.hydrate_target_cache` 接受 `shape[2] >= offset`，按引用收养 |
| prefill 快照 | `PrefixSnapshotBuilder.build` 在容量模式下强制 `adopt_cache_arrays=True` |
| sidecar 切片 | `slice_snapshot_at_sidecar_boundary` 不再物理切片 FA 数组，改为**保留全容量数组 + boundary 作为 offset**（同时重绑 `prefix_l1` 命名空间，因其按名导入） |
| draft 侧 | `FullContextDraftKVCache._grow` 同样增长直达容量 |

合成的运行链：请求 N 结束 → 快照零拷贝挂住容量缓冲 → 请求 N+1 空模板 hydrate 引用收养 →
所有 update 原地写。**会话内每一轮零重分配**（13 轮 bench 验证）。

### 3.4 L1 字节预算联动（防自逐出）

容量快照的物理字节数（~10.9GB）远超默认 L1 预算（8GiB），若不处理会**自我逐出** →
每轮全量重算。引擎在容量模式启用时自动抬高 L1 字节预算下限：

```python
l1_max_bytes = max(l1_max_bytes, capacity_tokens * 80 * 1024)  # 80KB/tok 覆盖 FA KV + GDN/draft 余量
```

### 3.5 结构性容量准入

`server.py` 在 completion/chat 入口先解析采样参数（含 `max_tokens`），preflight 检查：

```
prompt_tokens + max_tokens <= dflash_kv_capacity_tokens
```

超出即返回干净的 HTTP 400（`PrefillMemoryExceededError`）：

```
DFlash reserved KV capacity exceeded: request needs 206052 tokens, capacity is 163840
```

服务本身存活，不需要也不触发卸载。实测 prompt 6052 + max_tokens 200000 被正确拒绝。

### 3.6 会话切换释放

换会话（首轮 miss）时旧会话的容量缓冲已无复用价值，若不释放，新旧两份容量缓冲会在
新会话首配时叠加（实测首轮峰值 37.3GB）。补丁包装 `DFlashPrefixCache.lookup`：在容量模式下
检测到"记录的 miss + 存在 pinned 容量缓冲"时，先 `clear()` L1 并 `mx.clear_cache()`，
再进行新分配。实测切换首轮峰值降至 **27.2~27.3GB**，日志可见：

```
KV prealloc conversation switch: released N pinned capacity buffer(s) before fresh allocation
```

代价是切回旧会话需要全量重算 prefill——单会话模式下这是正确取舍。

### 3.7 prefill chunk 调小（2048 → 512）

`dflash_prefill_step_size=512` 把每 chunk 分数矩阵物化（`24 × chunk × ctx × 2B`）缩小 4 倍，
这是峰值从 ~38GB 收敛到 ~35GB 的主要来源之一；实测对 prefill 吞吐无明显伤害。

## 4. 实现载体

| 文件 | 内容 |
|---|---|
| `omlx/patches/dflash_kv_prealloc.py` | 全部 monkeypatch：P1/P2/P3、builder 强制 adopt、sidecar 全量切片、会话切换释放；install/restore 幂等，备份/恢复原函数 |
| `omlx/engine/dflash.py` | 配置接线、start 安装 / teardown 恢复、L1 预算下限、容量 preflight（`_preflight_kv_capacity_or_raise`）、准入信用（`_resident_cache_tokens`） |
| `omlx/model_settings.py` | 新配置字段（见下节） |
| `omlx/server.py` | completion 入口先解析 `max_tokens` 再 preflight |
| `tests/` | install/restore 幂等、切片语义、准入等 38+ 单测 |

启动日志确认（INFO 级）：

```
DFlash KV prealloc enabled: capacity=163840 tokens, target_kv_estimate=10.00GB, prefill_step=512, max_snapshot_tokens=32000
```

## 5. 验证结果（2026-08-20，48GB M4 Pro）

### 5.1 长会话爬坡（13 轮至 163,019 tokens）

- 稳态：26.16GB → 29.17GB（斜率 ~20-23KB/tok，全部来自分数池缓冲，无 KV 重分配）
- 峰值：27.86GB → **34.99GB**，全程距 39.06GB 上限 ≥4GB
- **零** enforcer 升级、零卸载、零 Metal OOM；会话内每轮零容量重分配
- 每轮增量 prefill（12.5K 新 token，L1 命中续算）104~234s，随 ctx 增长符合预期

### 5.2 会话切换（A→B）

切换后首轮峰值 27.21 / 27.32GB（优化前 37.3GB），释放日志按预期出现。

### 5.3 输出速度（优化后 vs 基线）

| 场景 | 基线 | 优化后 |
|---|---|---|
| 散文生成 | 15 / 17.4 tok/s | 15.2 / 15.6 tok/s |
| 代码生成 | 30 / 30.2 tok/s | 32.3 / 27.7 tok/s |
| TTFT | ~1s | 0.7~1.1s |

**速度无实质回退**（差异在轮间方差内）。

### 5.4 结构性拒绝

prompt 6052 + max_tokens 200000 → HTTP 400（消息见 3.5），服务存活，后续请求正常。

## 6. 内存画像与容量规划

优化后的内存模型（供规划剩余内存使用）：

```
总峰值 ≈ 底座 + ctx × 22KB（稳态斜率） + prefill 分数瞬态（≤4GB @163K）
底座   ≈ 模型权重 ~15.7GB + 预分配 target KV 10.0GB（163840 × 64KB）
```

容量档位参考（48GB 机器，上限 39.06GB）：

| dflash_kv_capacity_tokens | 底座 | 峰值（实测/推算） | 余量 |
|---|---|---|---|
| 131072 (128K) | ~24GB | ~31GB | ~8GB |
| **163840 (160K)** | **~26GB** | **~35GB（实测 34.99）** | **~4GB（推荐）** |
| ≥196608 | ~29GB+ | ~37GB+ | <2GB，不建议 |

> 36GB 机型建议 98304（96K）起步；改容量后需重启服务（预分配在首次建缓存时生效）。

## 7. 使用指导

### 7.1 配置（`~/.omlx/model_settings.json`，按模型条目）

```json
{
  "dflash_enabled": true,
  "dflash_draft_model": "/Users/juz/.omlx/models/Qwen3.8-27B-DFlash2",
  "dflash_draft_quant_enabled": true,
  "dflash_draft_quant_weight_bits": 4,
  "dflash_draft_quant_group_size": 64,
  "dflash_kv_capacity_tokens": 163840,
  "dflash_prefill_step_size": 512
}
```

| 字段 | 说明 | 默认 |
|---|---|---|
| `dflash_kv_capacity_tokens` | 预分配容量（token）；**0 = 关闭，完全走 stock 行为** | 0 |
| `dflash_prefill_step_size` | prefill chunk；None = 库默认 2048；单会话长上下文建议 512 | None |
| `dflash_max_snapshot_tokens` | prefill 快照 token 上限；None = 运行时默认 32000 | None |

`dflash_in_memory_cache_max_bytes` 无需手调：容量模式下引擎自动抬高到
`capacity × 80KB`（若配置值更低）。

### 7.2 安装与启动

**方式 A：独立安装（发布推荐，不依赖 oMLX.app）**

本 fork 是标准 pip 包（`omlx`），全部依赖在 pyproject/uv.lock 中声明并锁定，包括
git-pin 的 `dflash-mlx 0.1.10+omlx.5`、`mlx 0.32.0`、`mlx-lm 0.31.3`——与 app 内打包
版本完全一致（补丁验证所基于的版本）。目标机只需 macOS arm64 + Python 3.11~3.13：

```bash
uv venv ~/omlx-env
uv pip install --python ~/omlx-env/bin/python /path/to/omlx-fork
~/omlx-env/bin/omlx serve --model-dir /path/to/models --port 8000
```

2026-08-20 已实测全新 venv 安装（约 1.1GB）并启动：模型发现、HTTP 服务全部正常，
serve 路径对 `/Applications/oMLX.app` 零依赖（代码中对 app 路径的引用只出现在
集群启动、托管控件等非 serve 功能）。多实例隔离可用 `OMLX_BASE_PATH` 指定独立
配置目录。

**方式 B：复用 app 内 Python（当前开发机用法）**

```bash
R=/Applications/oMLX.app/Contents/Resources/Python
PYTHONHOME=$R/cpython-3.11 \
PYTHONPATH=/Users/juz/llama/omlx-fork:$R:$R/Python/framework-mlx-base/lib/python3.11/site-packages \
$R/cpython-3.11/bin/python3 -m omlx.server --port 8000
```

两种方式均等 `/health` ready（模型加载完成后底座即 ~26GB）。

### 7.3 客户端注意事项

- 请求体必须带 `"model"` 字段（fork API 校验，缺省 422）；
- **客户端不要继承服务端的 `PYTHONHOME/PYTHONPATH`**（会话内直接跑 python 客户端需
  `env -u PYTHONHOME -u PYTHONPATH python3 ...`，否则 `No module named 'encodings'`）；
- `max_tokens` 参与容量准入：`prompt + max_tokens > capacity` 会得到 400。客户端若
  习惯性传超大 `max_tokens`（如 200000），请改小或按需申请。

### 7.4 运维观察

| 关注点 | 手段 |
|---|---|
| 预分配是否生效 | 启动日志 `DFlash KV prealloc enabled: capacity=...` |
| 会话切换释放 | 日志 `KV prealloc conversation switch: released ...` |
| 意外容量重分配 | 日志 `realloc`（正常会话内不应出现；出现说明 token 分歧路径回归，需排查） |
| 内存 | `footprint` 采样盯服务进程；期望窄带（见第 6 节画像） |

## 8. 已知边界

- **冷启动全量 prefill 依旧慢**（计算瓶颈，~23min@113K）：预分配只保内存安全，不加速
  fresh prefill；跨会话切换回旧会话同样全量重算（3.6 的取舍）。
- **单会话假设**：并发多会话会频繁触发切换释放与重建，此优化不适用（多会话场景请保持
  `dflash_kv_capacity_tokens=0` 并依赖 stock 逐出机制）。
- 首轮（容量首配）峰值仍略高于后续轮（27.9 vs 稳态 26.2GB 量级），属一次性成本。
- 补丁通过反射定位 `dflash_mlx` 内部符号（如 `PrefixSnapshotBuilder`、
  `slice_snapshot_at_sidecar_boundary`），**app 升级后需回归验证**（单测 + 冒烟 bench）。

## 9. 内部发布建议

发布三件套，独立分发**已实测可行**（2026-08-20 全新 venv 验证，见 7.2）：

1. **omlx 本体**：本 fork 仓库（含 `uv.lock`）。接收方 `uv venv` + `uv pip install`
   即得独立 CLI 服务（`omlx serve`），无需安装 oMLX.app。
   - 安装期需可达 PyPI 与 GitHub（`mlx-lm` / `mlx-vlm` / `mlx-embeddings` /
     `dflash-mlx` 均为 git-pin）；接收方无外网时，用 `uv pip compile` + wheelhouse
     做离线包。
   - 自定义 Metal 内核（`omlx.custom_kernels.*`）默认不参与构建，dflash 路径
     用不到，安装无需 cmake/nanobind 工具链。
   - 包内 `omlx/eval`（62MB 评测数据）与 `omlx/admin`（24MB Web 管理界面）对纯
     API 使用非必需；如需剔除要同步验证 server 的路由挂载导入，建议首发不剪。
2. **模型与 draft 参数**：target 15GB（`mlx-community/Qwen3.8-27B-4bit`）+ draft
   3.6GB（`z-lab/Qwen3.8-27B-DFlash2`）+ 7.1 的 `model_settings.json` 条目。两者
   均为 HF 仓库快照：网络可达时让接收方自行下载，否则随包分发模型目录（放入
   `<model-dir>/` 并保留目录名）。
3. **使用指南**：本文档 + `docs/kv-prealloc-plan.md`。

其他发布要求：

- **版本锁定**：补丁以反射定位 `dflash_mlx` 内部符号，pyproject 已 pin
  `dflash-mlx@2eb169f`；升级任何依赖（尤其 dflash-mlx / mlx-lm / mlx）后必须跑
  `tests/`（38+ 单测）+ 一轮爬坡冒烟（`bench_omlx_agent_slope.py`）再放行。
- **机器规格分级**：48GB → 163840；36GB → 98304 起步。`iogpu.wired_limit_mb` 无需
  手工设置——omlx 启动时自动抬到内核允许上限并在日志给出建议值；但需提醒接收方
  机器内存要与容量档位匹配（第 6 节）。
- **发布前待办**：
  - 把 `dflash_kv_prealloc.py` 中每次请求的 INFO 级插桩日志降为 DEBUG；
  - fork 当前未提交改动（补丁模块、引擎接线、配置字段、server max_tokens 解析、
    单测、本文档与 plan 文档）打包成一次提交。
- **随附工具**：`bench_spec.py`（散文/代码速度）、`bench_omlx_agent_slope.py`（内存
  爬坡）作为验收基准随文档一起归档。
