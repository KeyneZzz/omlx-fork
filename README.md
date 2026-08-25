<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX Fork:DFlash2 单会话内存优化</h1>

<p align="center"><b>基于 <a href="https://github.com/jundot/omlx">jundot/omlx</a>(0.6.3rc1,<code>146d2724</code>)的改造分支</b><br>
面向<b>单实例、单会话驻留服务</b>场景,把 DFlash2 speculative decoding 的内存行为<br>从「随上下文增长 + 不可控瞬态」改造为 llama.cpp 式<b>启动即固定、峰值封顶</b>。</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.11--3.13-green" alt="Python 3.11-3.13">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/base--commit-146d2724-orange" alt="Base 146d2724">
</p>

> **本仓库不是上游 oMLX 本身。** 上游项目的完整介绍(安装、功能、模型支持等)请看
> [README.zh.origin.md](README.zh.origin.md)(上游中文原版 README,原样保留)与
> [omlx.ai](https://omlx.ai)。本 README 只说明这个 fork 改了什么、为什么、怎么用。

## 解决什么问题

上游 DFlash2 引擎在长上下文单会话场景下,内存由三部分不可控成分主导:

| 成分 | 机理 | 位置 |
|---|---|---|
| KV 增长拷贝 | `KVCache.update_and_fetch` 容量不足时 `mx.concatenate`,旧 + 新 + 结果三份数组共存,释放后滞留分配器池 | `mlx_lm/models/cache.py` |
| 分数物化 | head_dim=256 无融合 SDPA,每 chunk 每层物化 `[24, chunk, ctx]` 分数矩阵 | `dflash_mlx` 分发逻辑 |
| 池滞留 | 释放的缓冲滞留 MLX 缓存池,footprint 不回落 | turn 边界实测 +9.4GB |

实测(Mac M4 Pro 48GB,Qwen3.8-27B-4bit):100K ctx 时稳态之上仍有 **+15GB 瞬态**,真峰值 42.6GB,稳定可用上下文只有 ~100K,并且随时可能触发卸载。

## 改动内容

| 模块 | 说明 |
|---|---|
| `omlx/patches/dflash_kv_prealloc.py`(新) | **KV 容量预分配**:运行时 monkeypatch `mlx_lm` 的 `KVCache`、dflash codecs 与 draft cache,首次使用即按 token 容量一次性分配,消除增长拷贝与池滞留。不改动 app 内 `dflash_mlx` 代码 |
| `omlx/patches/dflash_session_base.py`(新) | **跨会话系统前缀复用**:预热「系统提示 + 工具定义」精确边界快照,新会话从基点续 prefill 而非全量重算。仅 capacity 模式生效 |
| `omlx/engine/dflash.py` | capacity 模式接线;快照按物理容量记账(所有快照别名同一物理缓冲,账面开销不占真实内存);session base 预热/服务;load/evict 时安装与卸载补丁 |
| `omlx/server.py` | 采样参数先于 preflight 计算,`preflight_completion` 携带 `max_tokens`,内存预检把生成 token 也计入 |
| `omlx/scheduler.py` | 更激进的 `mx.clear_cache()`:间隔 512 → 16 步;清理预留 `limit/3`(≥2GB)→ `limit/16`(≥1GB) |
| `omlx/model_settings.py` / `model_profiles.py` | 新增 5 个 `dflash_*` 运维旋钮,有意不暴露给 profiles / admin UI(见下表) |

## 实测效果

Mac M4 Pro 48GB(macOS,`iogpu.wired_limit_mb=40000`,GPU 上限 ≈39.06GB),Qwen3.8-27B-4bit(MLX 4bit)+ Qwen3.8-27B-DFlash2 draft:

- 内存行为:~~随 ctx 增长 + 不可控瞬态~~ → **启动即固定底座 ~26GB,运行中窄带波动,峰值封顶 ~35GB**
- 稳定可用上下文:~100K → **160K(163840 tokens)**
- 全程零卸载

完整数据与推导见 [docs/dflash-single-session-memory.md](docs/dflash-single-session-memory.md)。

## 使用

### 配置(`~/.omlx/model_settings.json`,按模型条目)

```json
{
  "dflash_enabled": true,
  "dflash_draft_model": "/Users/<you>/.omlx/models/Qwen3.8-27B-DFlash2",
  "dflash_draft_quant_enabled": true,
  "dflash_draft_quant_weight_bits": 4,
  "dflash_draft_quant_group_size": 64,
  "dflash_kv_capacity_tokens": 163840,
  "dflash_prefill_step_size": 512
}
```

### 新增设置项

| 设置 | 默认 | 说明 |
|---|---|---|
| `dflash_kv_capacity_tokens` | `0` | 固定 KV 容量(token)。`0` = 原生增量增长,行为同上游 |
| `dflash_prefill_step_size` | `None` | DFlash prefill chunk 大小覆盖(`None` = dflash 默认) |
| `dflash_max_snapshot_tokens` | `None` | 前缀快照 token 上限(`None` = dflash 默认;`0` 禁用) |
| `dflash_session_base_reuse` | `True` | 跨会话复用共享系统前缀(仅 capacity 模式) |
| `dflash_session_base_min_tokens` | `1024` | 触发 session base 预热/服务的最小前缀长度,更短的直接重 prefill |

容量档位参考:**48GB 机器 → 163840;36GB → 98304 起步**。`iogpu.wired_limit_mb` 无需手工设置——omlx 启动时会自动抬到内核允许上限并在日志给出建议值,但机器内存需要与容量档位匹配。

安装/构建方式与上游一致,见 [README.zh.origin.md](README.zh.origin.md) 的安装章节。

## 兼容性与升级注意

- 补丁以反射定位 `dflash_mlx` 内部符号;`dflash-mlx` 已 pin 到
  [`jundot/dflash-mlx@b7f6255`](https://github.com/jundot/dflash-mlx)(v0.1.10+omlx.6)。
- 升级任何依赖(尤其 dflash-mlx / mlx-lm / mlx)后,必须先跑 `tests/`(38+ 单测)
  并做一轮长上下文爬坡冒烟,再放行。
- 与上游 DFlash2 新版本的合并流程,见 [docs/upstream-dflash2-merge-runbook.md](docs/upstream-dflash2-merge-runbook.md)
  (包括 capacity 关闭/开启双模式冒烟、`min_p`/`seed` 参数在 `b7f6255` 的行为核对等)。

## 文档

- [docs/dflash-single-session-memory.md](docs/dflash-single-session-memory.md) — 技术总结与使用指导(根因、10 轮 bench 证据、调参指导)
- [docs/kv-prealloc-plan.md](docs/kv-prealloc-plan.md) — Phase 2 实现计划(设计推导)
- [docs/upstream-dflash2-merge-runbook.md](docs/upstream-dflash2-merge-runbook.md) — 上游 DFlash2 合并 runbook

## 测试

```bash
pytest tests/test_dflash_kv_prealloc.py tests/test_dflash_session_base.py \
       tests/test_dflash_engine.py tests/test_dflash_prefill_memory_guard.py
```

## 与上游的关系

- 基准提交:`146d2724`(formula: bump to 0.6.3rc1),上游全部历史(2400+ 提交)完整保留,可随时 `git diff` 对比或 rebase 新上游。
- 上游原版中文 README 保留为 [README.zh.origin.md](README.zh.origin.md)。
- 感谢 [@jundot](https://github.com/jundot) 的 oMLX 与 DFlash2 相关上游工作。

## License

Apache-2.0,随上游([LICENSE](LICENSE))。
