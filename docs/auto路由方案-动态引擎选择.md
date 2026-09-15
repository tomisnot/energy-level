# auto 智能路由方案：Sambe vs Hard（动态引擎选择）

> 2026-09-05 船长构思。目标：用户选 auto 时，根据配置的场与能级，自动判断
> 用 Sambe（全频段 frozen）还是 Hard（两场旋转系+硬积分）。本文档先定判据
> 与路由逻辑，评审通过后实现。

---

## 1. 两个引擎的能力边界（调研事实，均已验证）

| 能力 | **Sambe**（evolve_sambe frozen）| **Hard**（evolve_hard）|
|---|---|---|
| 变频/任意相位场 | ❌ 不支持（光子块需固定载频）| ✅ cos(φ_f(t)) 任意相位（t18）|
| 多光场 STIRAP/拉曼 | ✅ 无 RWA/无参考（t39 仲裁正确）| ❌ **跨流形链结构性不可用**（t34；同流形拍频 t10 ✅）|
| 光场 RWA 精度 | ✅ 无 RWA（dressed 全含）| ⚠️ RWA（近共振 0.5 振幅，反旋丢弃 1e-14）|
| 微波处理 | ✅ 光子块 M=2 | ✅ 硬积分（无 RWA）|
| 远失谐强对 | ⚠️ 需 AC 化（t61 修，进行中）| ✅ RWA 自动丢弃（1e-14）|
| 拍频在 Sambe 冻结 | ⚠️ 各载频光子块可处理，但两场 Sambe 显式拒绝（L448）| ✅ |
| 速度（22 态 620ns） | frozen ~分钟级（eigh/段）| ~2s（numba DOP853）|
| 单原子两体 | 均有 | 均有 |

**关键不对称**：
1. **变频/任意波形只有 Hard 能做**（sambe 光子块周期性假设）——这是硬路由
2. **STIRAP 类跨流形链（5S↔6P↔60S 多光场）hard 结构性不可用**：rot_pick 每光场
   独立取低流形旋转（5S+ω1、6P+ω2）→ 光1 耦合对旋转系残差 = −295.7 THz（RWA 完全
   失效，非精度问题，t33/t34 + 2026-09-06 实测复核）→ **auto 硬路由 sambe/混合**
3. **速度差 ~100×**：用户若追求快（交互/扫描）应偏 hard；追求精确应偏 sambe

---

## 2. 路由判据（草案，逐条评审）

### 判据 A：存在任意波形/变频场 → Hard（硬性）
- 检测：waves 中有 `ArbitraryWave`（kind='arb'）或任何 phase_fn 变频
- Sambe 无法构造光子块 → 必选 hard
- 若场景还有多个普通光场 → hard 的拍频路径处理（t18/t10 已支持）

### 判据 B：光场是否需高精度 dressed（无 RWA）
- 检测：光场强度高（Rabi/失谐 > 阈值？）或场数多且共享流形（STIRAP）
- 原则：RWA 误差 ~ (Ω/Δ)² 与 (Ω/ω)²——**强光近共振 → sambe**（无 RWA）
- 弱光远失谐 → hard 够（RWA 1e-14 可忽略）
- **已实测（2026-09-06，scripts/tmp/_route_stirap_data.py）**：STIRAP.json 双光场
  光1 (5S↔6P) 旋转系残差 −295.722 THz → hard RWA 完全失效——跨流形链 STIRAP
  **结构性硬路由 sambe**（非 <1e-3 精度偏好问题，无放宽空间）

### 判据 C：微波 vs 光场主导
- 纯微波（无光场）→ hard（快且精确，微波无 RWA）
- 光场主导（STIRAP/拉曼）→ sambe（无 RWA）——**已实测为结构性硬路由**（见 B）
- 混合（光+微波）→ 看光场强度（B）；强光近共振 → 混合引擎（t62，微波当场）

### 判据 D：两体维数可行性（sambe 特有）
- sambe 两体 M 按 t58 r 判据 + t61 AC 化后 M=1 可行（dim 4356）
- 若某场景 AC 化后仍维数爆炸（多强对）→ hard（2420 维）

### 判据 E：用户精度/速度偏好（GUI 提供）
- auto 需用户给一个旋钮：`精度优先`（→sambe）/ `速度优先`（→hard）/ `auto 智能`
- 否则 auto 默认按物理判据（A-D）

---

## 3. 路由逻辑（草案）

```
def route_auto(states, waves, t_grid, preference='auto'):
    1. 若有 ArbitraryWave/变频 → return 'hard'          (判据 A)
    2. 若无光场（纯微波）→ return 'hard'                 (判据 C)
    3. 光场强度分析：每光场 max|Rabi|、对近共振失谐
       - 全弱光（Rabi/Δ << 1）→ return 'hard'（RWA 够，快）(判据 B)
       - 有强光近共振 → 候选 sambe（无 RWA）            (判据 B)
    4. sambe 可行性检查：
       - 两体维数（per_freq_M + AC 化后）> 上限 → 降 hard (判据 D)
    5. 候选集 {sambe, hard}：
       - 若 preference='精度' → sambe
       - '速度' → hard
       - 'auto' → 按物理（强光/多光场 STIRAP → sambe；否则 hard）
    6. 返回 (method, reason)
```

---

## 4. 实测补充的数据（状态：1 已实测，2/3 归档）

1. ~~**hard 在 STIRAP（双光场高斯）的误差**~~ → **已实测（2026-09-06）**：
   STIRAP.json（光1 713.28 THz 5S↔6P + 光2 295.72 THz 6P↔60S）route_auto 无 cache/
   有 cache/两体 三情形均路由 sambe ✓；build_two_field rot_pick：5S 全 +713.282 THz、
   6P 全 +295.723 THz → 光1 耦合对旋转系残差 **−295.722 THz**（光2 对 −196.6 MHz 正常）
   → **hard 对跨流形链 RWA 完全失效，结构性硬路由 sambe**（脚本 scripts/tmp/_route_stirap_data.py）
2. **hard 在强光近共振（大 Rabi）的 RWA 误差**：vs sambe——归档于 t39 仲裁
   （ip4/hard 同源 RWA；STIRAP 光2 强脉冲区 ip4 中段误差 2.2e-2，末态收敛 9e-4）
3. **变频场景唯一性确认**：ArbitraryWave → hard（sambe 光子块需固定载频，无隐藏路径）

---

## 5. GUI 集成

- method_cb 加 'auto' 项（默认 auto 或保留 hard？评审）
- auto 时：route_auto 返回 method + reason 显示（"auto→hard：变频场"等）
- 保存/恢复：auto 是新档位，需序列化兼容（旧档映射已有）
- 两体页同步（两体 auto 路由同判据 + D 维数检查）

---

## 6. 评审点（请用户/队长定）

1. **判据 B 的"强光"阈值**怎么定：物理量（Ω/Δ）还是让用户精度旋钮决定？
2. **auto 是否需要用户偏好旋钮**（精度/速度/智能），还是纯物理判断？
3. **hard 在 STIRAP 实测**是否要先做（决定 STIRAP 是否硬路由 sambe）？
4. GUI method 列表顺序/默认值（auto 放哪、默认谁）
