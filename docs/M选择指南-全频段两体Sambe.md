# 全频段两体 Sambe：逐载频光子数（M）选择指南（t53）

> 2026-09-03 harddev。per_freq_M API 已就绪（evolve_sambe_pair，d2396c9）。

## 物理判据：r = Ω/ω（Rabi / 载频）

光子块截断误差 ~ r^(M+1) 量级（Floquet 收敛）：

| 频段 | r = Ω/ω | M 建议 | 截断误差（实测） |
|---|---|---|---|
| 光场（~300 THz） | ~1e-7 | **M=1** | ~1e-14（可忽略） |
| 微波（~17 GHz） | ~3e-4 | **M=1** | 5.75e-5（Ω=5.1MHz 实测） |
| 微波（强驱动） | ~3e-3（Ω=50MHz） | M=2 | 6.86e-4（仍 <1e-3） |

## 精度-性能选档表（22 态两体交换全过程，GPU RTX 4050）

| (M_opt, M_mw) | dim | GPU eigh/段 | h（mw 区） | 620ns 全程 | max\|Δpops\| 级 |
|---|---|---|---|---|---|
| [1,1] | 4356 | ~3.5s | 8ns（含 const 合并） | **~2.4 分钟**（46 eigh） | 微波 M1 5.75e-5 + h=8 冻结 4.8e-4 → <1e-3 ✓ |
| [1,1] | 4356 | ~3.5s | 2ns | ~10 分钟 | ~5e-5 |
| [1,2] | 7260 | ~14.8s | 8ns | ~11 分钟（估） | 微波 M2 更高精度 |
| [2,2] | ~12000+ | >30s | 8ns | ~30+ 分钟 | 更高阶（通常不必要） |

## per_freq_M 用法

```python
from src.core.evolve_sambe_pair import evolve_sambe_pair

# 光场 M=1 + 微波 M=1（推荐：620ns ~2.4 分钟，误差 <1e-3）
r = evolve_sambe_pair(states, fields_A, fields_B, psi0_A, psi0_B, t,
                      frozen_h_ns=8.0, per_freq_M=[1, 1])

# 微波要更高精度 → M=2（dim 7260，~11 分钟）
r = evolve_sambe_pair(..., frozen_h_ns=8.0, per_freq_M=[1, 2])

# 未指定 → 均匀 M 探针（历史行为）
r = evolve_sambe_pair(states, fields_A, fields_B, psi0_A, psi0_B, t)
```

per_freq_M 的 list 顺序 = 载频组顺序（_pair_groups 按出现顺序分组：A 侧光场组、B 侧微波组……）。

## const 段合并（620ns ~2.4 分钟的关键）

- K 恒定块（包络变化 < 1e-9 相对）→ 单 eigh 覆盖整块 + 块内传播向量化
- 交换全过程 [329-620]（光 square 平台-only）单 eigh 覆盖 291ns
- mw 高斯活跃区（σ=20）h=8ns 逐段（frozen 语义，误差 4.8e-4）
- 实测：46 eigh / 145.6s / 峰值显存 2.14GB（<6.4 ✓）

## GUI 接线（t56 已落地）

- `src/ui/evolve_worker.py` 的 **EvolveWorkerPair** 已支持 `per_freq_M`/`frozen_h_ns` 参数
  （method='sambe'/'exact' 分支透传到 evolve_sambe_pair；旧路径 pair_evolve 保留）
- 调用方（两体动态页）传 `per_freq_M=[1, 1]`、`frozen_h_ns=8.0` 即用全频段推荐档；
  不传 = None（引擎自动判据/历史行为，向后兼容）
- ui_selftest 57/57 不破坏；smoke：2 态小场景 method='sambe' + per_freq_M=[1,1] +
  frozen_h_ns=0.5 跑通（pops2d (31,4)）
- UI 面板控件（每载频组 M 微调框）为后续美化项——引擎与 worker 调用链已完整
