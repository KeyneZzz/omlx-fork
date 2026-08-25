# Phase 2 实现计划：KV 预分配（llama.cpp 式固定内存画像）

目标：单实例单会话场景下，把 omlx-fork (dflash) 的内存行为从"随 ctx 增长 + 不可控瞬态"
改为**启动即固定底座 + 运行中窄带波动**，将 Qwen3.8-27B-4bit 的稳定可用 ctx 从 ~100K
推到 **160K**。

## 1. 根因回顾（已由 10 轮 bench + 代码证据确认）

稳态之上的瞬态（100K 时 +15GB，真峰值 42.6GB）由三部分构成：

| 组件 | 机理 | 位置 |
|---|---|---|
| KV 增长拷贝 | `KVCache.update_and_fetch` 容量不够时 `mx.concatenate`，旧+新+结果三份共存，释放后滞留分配器池 | `mlx_lm/models/cache.py:335-350` |
| 分数物化 | head_dim=256 无融合 SDPA，每 chunk 每层物化 `[24, chunk, ctx]` 分数矩阵（实测 ≈1×，softmax 输出复用池缓冲） | `dflash_mlx/engine/target_qwen_gdn.py` 分发逻辑 |
| 池滞留 | 释放缓冲滞留 MLX 缓存池，footprint 不回落 | turn 9 后实测 +9.4GB |

实际 prefill chunk = **2048**（`DEFAULT_RUNTIME_CONFIG.prefill_step_size`，omlx 未覆盖）。
实测稳态斜率 **85KB/tok**（FA KV 64KB + draft 侧 ~21KB）。

## 2. 设计总览

```
启动（模型加载 + 预分配）
  底座 = fp0 15.7GB + 160K × 85KB ≈ 29.6GB        ← 一次性，含 draft 全上下文 KV
运行窄带
  decode:  底座（q_len=1，分数 ~8MB/层，可忽略）
  prefill: 底座 + 24×512×ctx×2B（满 160K 时 +3.9GB）→ 封顶 ~33.5GB
  无增长拷贝、无卸载-重载、无 23 分钟重建；enforcer(soft 35.2) 不再触发
超容量请求
  prefill 结构性拒绝（干净 400，PrefillMemoryExceededError）
```

容量档位（chunk 512，ceiling 39.06GB）：

| C | 底座 | 峰值 | 余量 |
|---|---|---|---|
| 131072 | ~26.6GB | ~30GB | 9GB |
| **163840** | **~29.6GB** | **~33.5GB** | **5.5GB** |
| 196608 | ~32.1GB | ~36.8GB | 2.2GB（不建议） |

## 3. 改动清单

### 3.1 新模块 `omlx/patches/dflash_kv_prealloc.py`

遵循 `dflash_lifecycle.py` 的备份/恢复模式（install at start / restore at stop，
幂等，进程级一次）。三个 monkeypatch：

**P1 — `KVCache.update_and_fetch`（增长直达容量）**

原语义：容量不足时按 `step=256` 对齐增量 concatenate。
补丁语义：
- 首次分配：直接分配到 `CAPACITY`（`mx.zeros((B, H, cap, D))`，`mx.eval` 提交）
- 已在容量内：走原逻辑（原地 slice 写入 `keys[..., prev:offset, :] = keys`，零分配）
- `prev + n > CAPACITY`：抛 `RuntimeError("KV cache capacity exceeded")`
  （scheduler 的 prefill 错误路径会转成请求错误；preflight 结构检查会更早拦截）

类级补丁覆盖**所有** KVCache 实例：target 的 16 层 FA KV（64KB/tok）与 draft 的
全上下文 KV（~21KB/tok）一次到位，正好等于实测 85KB/tok 斜率。
补丁仅在 DFlash 引擎主模式期间生效（fallback/teardown 时恢复），避免影响其他引擎。

**P2 — `codecs.serialize_target_cache`（adopt 路径携带全容量数组）**

原语义：`entry.state` 返回 `keys[..., :offset, :]` 切片视图 → 快照 fa 数组是
exact-length → hydrate 后下一次 update 必然重新触发增长分支（codecs 注释自认）。
补丁语义：`clone=False`（generation 快照的 adopt 路径）时直接取 `entry.keys /
entry.values` 全数组引用（零拷贝），offset 仍记逻辑长度。
`clone=True`（prefill 快照）保持切片不变。

**P3 — `codecs.hydrate_target_cache`（接受 shape ≥ offset，零拷贝收养）**

原语义：`k.shape[2] != offset → raise "exact-length ... cannot adopt"`。
补丁语义：`k.shape[2] >= offset` 即可，`new_cache.keys = k; new_cache.offset = offset`
（引用收养，无拷贝、无新分配）。

三者合成的零拷贝链：

```
请求 N 结束: generation 快照 adopt 全容量缓冲（P2，零拷贝）
请求 N+1: make_cache 建空模板（keys=None，零成本）
          → hydrate 引用收养同一缓冲（P3，零拷贝）
          → 全部 update 原地写（P1，容量内零分配）
```

全程**恰好一份**容量缓冲，无 per-request 拷贝。

### 3.2 `omlx/engine/dflash.py` 接线

1. 新配置（`model_settings` getattr 模式，同 `dflash_in_memory_cache_max_bytes`）：
   - `dflash_kv_capacity_tokens: int = 0`（0 = 关闭预分配，默认不动生产行为）
   - `dflash_prefill_step_size: int | None = None`（None = 库默认 2048；单会话配 512）
2. `start()`：模型加载后调用 `install_dflash_kv_prealloc(capacity)`，
   记录日志（capacity、按 monitor 模型信息算出的预分配字节数）；
   两处 teardown（evict ~L780 / stop ~L863）调用 `restore_dflash_kv_prealloc()`。
3. `_build_runtime_context`：透传 `prefill_step_size`（签名已支持）。
4. `preflight_chat` / `preflight_completion`：结构性容量检查——
   `capacity > 0 且 num_tokens > capacity - 4096` 时 raise
   `PrefillMemoryExceededError`（复用现有 HTTP-400 映射），消息注明
   "exceeds reserved KV capacity"。
5. `prefix_cache_max_bytes` 联动：快照 nbytes 现在按物理数组计（~14GB@160K），
   需把 `dflash_in_memory_cache_max_bytes` 提到 `capacity_bytes × 1.2` 以上
   （配置侧设置；单会话下 L1 同时最多一份容量快照）。

### 3.3 运行配置（~/.omlx/model_settings.json）

```json
{
  "dflash_kv_capacity_tokens": 163840,
  "dflash_prefill_step_size": 512,
  "dflash_in_memory_cache_max_bytes": 20615843020
}
```
pinned-model 预加载（server 已支持 startup preload + /health 503 loading），
保证"启动即达底座"而不是首个请求才加载。

### 3.4 不改的东西（明确排除）

- `memory_monitor.estimate_prefill_peak_bytes`：现有估算（current + KV_new + SDPA）
  在预分配下依然成立——fresh 时 current 低、续写时 cached 抵扣——保守无害，保留为
  物理后备防线。准入修复（`_resident_cache_tokens`）保持不变。
- L1/sidecar 机制：不动（单会话下仅 156MB，是 KV 延续载体）。
- scheduler 周期 clear_cache（pool>1GB / 16 步）：保留，decode 期池很小，无冲突。
  请求边界**不**额外 clear（keep-flat 策略：行为最可预测）。

## 4. 风险与边界

| 风险 | 缓解 |
|---|---|
| draft 全上下文 KV 若不用 mlx_lm KVCache 类，P1 覆盖不到 → 残留 ~21KB/tok 增长 | bench 斜率验证；未覆盖则把补丁扩到 draft cache 类 |
| sidecar 路径（token 分歧时的降级命中）仍是切片视图 → 一次增长拷贝 | 罕见路径；发生时退化为单次 copy（~30ms），可接受 |
| 类级 P1 影响进程内其他 KVCache 使用者 | 仅主模式期间挂载；teardown 恢复（同 dflash_lifecycle 模式） |
| nbytes 按物理计后 L1 预算语义变化 | 3.2-5 的预算联动；后续可改 nbytes_breakdown 按逻辑长度计（refinement） |
| 预分配后 enforcer 若误触发（底座 29.6 < soft 35.2，理论不会） | 验证阶段盯 enforcer 日志 |
| 冷启动大会话 fresh prefill 依然慢（~23min@113K，计算瓶颈） | 预分配不加速 fresh prefill，只保内存安全；已知边界 |

## 5. 验证计划

1. **静态**：`py_compile`；install/restore 幂等单测（仿 `tests/test_decode_fairness.py` 风格）。
2. **冒烟**（1 轮）：重启 fork serve（新 model_settings + pinned 预载）→
   `/health` 等 loading 完成 → 确认加载后 footprint ≈ 29.6GB（**底座立现**）。
3. **窄带验证**（核心，~40min）：`bench_omlx_agent_slope.py 8000 12288×13 → 160K`：
   - 每轮"边际"应 ≈ 0（稳态平线 29.6GB）——这是预分配成立的直接信号
   - peak ≤ ~34GB，无 enforcer 事件、无 unload/reload 日志
   - decode 速度不回退（~30 tok/s）；prefill 轮耗时对比基线（chunk 512 vs 2048 的实测代价）
4. **结构性拒绝**：追加一轮到 ~170K → 期待干净 400（"exceeds reserved KV capacity"）。
5. **chunk A/B**（可选，信息补充）：`dflash_prefill_step_size=1024` 复跑 3 轮，
   对比 prefill 相位耗时与峰值，决定 512/1024 最终取值。

## 6. 成功标准

- 13 轮全部成功到 160K，稳态平线，峰值 ≤ 34GB
- 无 enforcer 升级、无模型卸载、无 Metal OOM
- decode 30 tok/s 保持；prefill 每轮耗时劣化 ≤ 10%
- 170K 请求被结构性拒绝且服务存活
