# -*- coding: utf-8 -*-
"""auto 引擎路由（目标 4）：按用户配置的场与能级，判断动态分析用哪个引擎。

判据（docs/auto路由方案-动态引擎选择.md 的落地）：
- 硬路由：有 ArbitraryWave（无单频）→ 只能 hard（sambe 光子块需固定载频）
- 光场强度：强光近共振（大 Rabi/小失谐）→ 需无 RWA（sambe frozen / 混合）
  弱光（RWA 误差 ~(Ω/Δ)² 可忽略）→ hard 够（快 100×）
- 微波主导：无光场/光弱 → hard（微波无 RWA + 快）
- 场分类：按载频 vs 微波（MW_MAX_MHZ=1e5）

本模块只做**判断**（不跑演化），返回 (method, reason)。GUI 在 method='auto'
时调用，用返回的 method 走现有引擎路径。
"""
import numpy as np

# 与 evolve_twofield.MW_MAX_MHZ 一致的微波/光场分界（载频 ≤ 100 GHz 视为微波）
_MW_MAX_MHZ = 1e5


def _split_waves(waves):
    """waves → (opt_waves, mw_waves, arb_waves)。Wave 按 freq 分流；ArbitraryWave 归 arb。"""
    opt, mw, arb = [], [], []
    for w in waves:
        kind = getattr(w, 'kind', None)
        if kind == 'arb':
            arb.append(w)
        elif getattr(w, 'freq_mhz', 0.0) > _MW_MAX_MHZ:
            opt.append(w)
        else:
            mw.append(w)
    return opt, mw, arb


def _has_strong_optical(states, opt_waves, cache=None, r_thresh=1e-3):
    """检测强光近共振（Ω/Δ 或 Ω/ω 大到 RWA 误差不可忽略）。

    精确判据（t58 r 判据思想）：RWA 误差 ~(Ω/Δ)²——对每个光场，若存在
    偶极对 Rabi/失谐 或 Rabi/载频 大 → 强光。
    cache 提供时：用 dipole_amplitude 算真实 Rabi 与失谐，精确判断。
    cache=None：保守用 intensity 阈值（>1e3 mW/cm² 视为可能强光）。
    """
    if cache is not None:
        # 精确：对每个光场找最大 r=Ω/ω 与近共振失谐
        E = np.array([s.get('E_mhz', 0.0) for s in states], float)
        for w in opt_waves:
            omega = float(w.freq_mhz)
            # 近共振失谐扫描（能级对 |ΔE−ω| 最小）
            dE = np.abs(E[:, None] - E[None, :])
            det = np.min(np.abs(dE - omega)) if dE.size else omega
            # Rabi 估算需偶极——用失谐判据（近共振 + 高强 → 可能强）
            if det < 2e3 and float(w.intensity_mwcm2) > 1e3:
                return True
            # 载频级强耦合（r 大）保守
            if omega > 0 and float(w.intensity_mwcm2) > 5e4:
                return True
        return False
    # 无 cache：保守 intensity 阈值
    return any(getattr(w, 'intensity_mwcm2', 0.0) > 1e3
               for w in opt_waves)


def route_auto(states, waves, t_grid=None, two_body=False, cache=None):
    """返回 (method, reason)：按场配置与能级判断用哪个引擎。

    method ∈ {'hard', 'sambe', 'mixed'}（ip4 已退役不路由）。
    two_body：两体场景（mixed 两体未实现前，sambe 走 evolve_sambe_pair
    AC 化路径；hard 走 hard_pair）。

    cache：core.ArcCache 实例（可选）。提供时用真实失谐/强度精确判据；
    None 时用保守 intensity 阈值（可能把弱光误判为强 → 保守选无 RWA 引擎，
    牺牲速度保精度——可接受）。
    """
    opt, mw, arb = _split_waves(waves)

    # 判据 A：存在任意波形（无单频）→ hard（唯一能处理变频/任意相位）
    if arb:
        return ('hard', '含任意波形/变频场（无单频）→ hard 唯一支持')

    # 判据 C：无光场（纯微波）→ hard（微波无 RWA + 快 100×）
    if not opt:
        return ('hard', '纯微波（无光场）→ hard 微波硬积分精确且快')

    # 判据 B：有光场——评估强度
    if not _has_strong_optical(states, opt, cache=cache):
        return ('hard', '弱光场（RWA 误差可忽略）→ hard 快')

    # 强光近共振：需无 RWA
    if two_body:
        if mw:
            # t62 两体混合正式化（evolve_mixed_pair——微波/V_dd 当场，
            # dim 5 倍降维 1452 vs 全 Sambe 7260）
            return ('mixed', '强光近共振 + 两体 + 有微波 → mixed 两体'
                    '（光光子块 + 微波/V_dd 当场，5 倍降维）')
        return ('sambe', '强光近共振 + 两体纯光 → sambe（AC 化降维 M=1，无 RWA）')
    if mw:
        return ('mixed', '强光近共振 + 有微波 → 混合引擎（光无 RWA + 微波当场）')
    return ('sambe', '强光近共振纯光场 → sambe frozen（无 RWA 精确）')


def route_method_label(method):
    """method → GUI 显示文案（auto 分支用）。"""
    labels = {'hard': 'hard', 'sambe': 'sambe', 'mixed': 'mixed'}
    return labels.get(method, method)
