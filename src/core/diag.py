from .static_atom import MW_MAX_MHZ
from .sambe import _M_MAX
# -*- coding: utf-8 -*-
"""诊断摘要/模型卡的**唯一真值源**（P5：一套代码，两个渲染器）。

诊断快照本身由**引擎**产出，契约写在 `solver_inspector` 顶部。
**目前只有 `evolve_mixed` / `evolve_mixed_pair` 产出**（`out['diag']`，
`diag=True` 时；sambe / hard / two_field / rwa / reference 都不产出）——
`summarize(None, {'method': …})` 的消息会点名这件事。

本模块把 GUI 检视器、MCP `get_diag`、模型卡三处摘要收敛成一处，都调
`summarize()` 与 `warnings()`。**不改变 diag 快照本身**（那是引擎的产物）。
"""
import numpy as np


def _g(d, *path, default=None):
    """安全取嵌套字段。"""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _pair_split(prs):
    """pairs 段 → (n_microwave, n_optical_vdd)。

    ⚠ 2026-09-14 修（自检实测暴露）：`summarize/warnings` 原来读
    `prs['n_microwave']`，但**契约里根本没有这个键**（见 `solver_inspector`
    顶部的 diag 结构契约：`pairs: {a, b, v, dD, is_mw, w}`）⇒ 它**永远是 None**，
    于是"微波候选对为 0"的告警**从不触发**（不是误报，是哑掉）。
    现在按契约键 `is_mw` 现算（**新增非契约键时优先用它**，保持向后兼容）。
    """
    if prs.get('n_microwave') is not None:
        return prs.get('n_microwave'), prs.get('n_optical_vdd')
    is_mw = prs.get('is_mw')
    if is_mw is None:
        return None, None
    try:
        arr = np.asarray(is_mw).ravel()
        n_mw = int((arr == 1).sum())
        return n_mw, int(arr.size - n_mw)
    except Exception:                              # noqa: BLE001
        return None, None


def _solver_steps(solv):
    """solver 段 → 步数（契约键 `t_steps`/`dt` 优先；`n_steps` 仅为兼容）。"""
    if solv.get('n_steps') is not None:
        return solv.get('n_steps')
    for k in ('t_steps', 'dt'):
        arr = solv.get(k)
        if arr is not None:
            try:
                return int(np.asarray(arr).ravel().size)
            except Exception:                      # noqa: BLE001
                continue
    return None


def _omega_ratio(prs, cfg=None):
    """逐对 Ω/ω 的**最大值**（RWA 适用性判据）——唯一实现。

    2026-09-15 修（核查报告 B5）：原来 `summarize` 与 `warnings` 各算一遍，
    且分母取 `cfg['max_freq_mhz']`（配置里**最大**载频）。光+微波混合场景下
    分母被光频（≈2.96e8 MHz）主导 ⇒ 比值恒 ≪1 ⇒ "已出 RWA 适用区"这条
    **几乎永不触发**——而它恰恰是为强驱动越界准备的。
    现在按**逐对**算（契约里 `pairs` 自带 `w`，用它当分母），取最大；
    无逐对数据时回退旧口径（保守但不再主导）。
    """
    try:
        import numpy as _np
        ws = prs.get('w')
        vs = prs.get('v')
        if ws is not None and vs is not None:
            num = _np.abs(_np.asarray(vs, dtype=complex)).ravel()
            den = _np.abs(_np.asarray(ws, dtype=float)).ravel()
            rr = [float(a) / float(b) for a, b in zip(num, den) if float(b) > 0]
            if rr:
                return max(rr)
    except Exception:                              # noqa: BLE001
        pass
    cfg = dict(cfg or {})
    vmax, om = prs.get('max_abs_v_mhz'), cfg.get('max_freq_mhz')
    if vmax and om:
        try:
            return float(vmax) / float(om)
        except ZeroDivisionError:
            return None
    return None


def summarize(diag, cfg=None):
    """diag 快照 → 规范化摘要（GUI 摘要头 / get_diag / 模型卡共用）。

    cfg（可选）：{'basis_n':…, 'has_mw':bool, 'has_opt':bool, 'method':str}
    —— 用于**判定式警告**（缺数据时不猜）；`method` 给定时，无快照的消息会点名。
    """
    cfg = dict(cfg or {})
    if not isinstance(diag, dict):
        m = cfg.get('method')
        who = 'method=%s ' % m if m else '本引擎 '
        return {'msg': '%s不产出诊断快照——诊断当前只有 mixed 引擎产出'
                       '（传 diag=True 也不会改变这一点）' % who,
                'warnings': []}
    meta = diag.get('meta') or {}
    solv = diag.get('solver') or {}
    prs = diag.get('pairs') or {}
    acz = diag.get('acize') or {}
    pho = diag.get('photon') or {}
    dim = meta.get('dim')
    vmax = prs.get('max_abs_v_mhz')
    # 载波频率判据：**逐对** Ω/ω（见 _omega_ratio；不再用"最大载频"当分母）
    ratio = _omega_ratio(prs, cfg)
    ratio = None if ratio is None else round(float(ratio), 4)
    out = {
        'method': meta.get('method'), 'dim': dim, 'N': meta.get('N'),
        'P': pho.get('P', meta.get('P')), 'M': pho.get('Ms', meta.get('M')),
        'n_pairs': prs.get('n', meta.get('n_pairs')),
        'n_microwave': _pair_split(prs)[0],
        'n_optical_vdd': _pair_split(prs)[1],
        'keep': meta.get('keep'), 'drop_cap': meta.get('n_drop_cap'),
        'drop_phase': meta.get('n_drop_phase'),
        'acized_states': acz.get('n_shifted'),
        'max_abs_v_mhz': vmax,
        'max_omega_over_omega': ratio,
        'tol': meta.get('tol'), 'max_step': meta.get('max_step'),
        't_end_ns': meta.get('t_end'), 'r_crit': meta.get('r_crit'),
        'mw_cap_mhz': meta.get('mw_cap'),
        'n_steps': _solver_steps(solv), 'nfev': solv.get('nfev'),
        'norm_dev': solv.get('norm_dev'),
        'basis_n': cfg.get('basis_n'),
    }
    out['warnings'] = warnings(diag, cfg)
    return out


def warnings(diag, cfg=None):
    """判定式警告（**只在数据确实给出时才判**，避免 E2 那类误报）。

    规则：
      1. 微波候选对为 0 **且配置里确有微波波形** → 疑缺中间态/频率不匹配；
      2. maxΩ/ω > 1 → 已出 RWA 适用区（物理存疑、数值未必错）；
      3. 光子块 M 达上限（截断误差可能 > tol）；
      4. 有对因成本窗被剔除（等于改了模型，必须交代）。
    """
    cfg = dict(cfg or {})
    w = []
    if not isinstance(diag, dict):
        return w
    prs = diag.get('pairs') or {}
    meta = diag.get('meta') or {}
    pho = diag.get('photon') or {}
    n_mw = _pair_split(prs)[0]
    # 语义：cfg 明确说"没有微波波形"时不报（修 E2 误报）；
    # cfg 未提供（未知）时保守报警——宁可多提示，不可漏掉真问题
    if n_mw is not None and int(n_mw) == 0 and cfg.get('has_mw', True):
        w.append('微波候选对为 0：基组可能缺中间态（多光子通道需要真实中间态）'
                 '或频率与能级间隔不匹配 —— 该波不会产生任何转移')
    ratio = _omega_ratio(prs, cfg)
    if ratio is not None and ratio > 1.0:
        w.append('maxΩ/ω=%.2f > 1：已出 RWA 适用区（逐对取最大；物理存疑，'
                 '数值未必错）' % ratio)
    Ms = pho.get('Ms') or meta.get('M') or []
    try:
        m_max = _M_MAX
        if Ms and m_max and max(int(x) for x in Ms) >= int(m_max):
            w.append('光子块 M 达上限 %s：截断误差可能 > tol' % m_max)
    except Exception:                              # noqa: BLE001
        pass
    if int(meta.get('n_drop_cap') or 0) > 0:
        w.append('有 %s 对因成本窗被剔除（等于改了模型，结论需交代）'
                 % meta.get('n_drop_cap'))
    # D1（2026-09-15）：光场对因 BW_OPT 被丢弃——原来完全静默（同族机制却有 loud
    # 统计）。BW_OPT 名义上是"双峰分离器"（共振对 ~0 vs 反旋 ~2ω），但它同时会
    # 删掉任何落在 20 GHz~2ω 之间的光场对，必须交代。
    if int(meta.get('n_drop_bw') or 0) > 0:
        w.append('有 %s 对光场耦合因 BW_OPT(%.0f MHz) 被剔除'
                 '（最大 |v|=%.3g MHz；等于改了模型，结论需交代）'
                 % (meta.get('n_drop_bw'), float(meta.get('bw') or 0.0),
                    float(meta.get('max_drop_bw_abs_v_mhz') or 0.0)))
    return w


def config_hints(reg_read):
    """从注册表读三条判据所需的配置线索（GUI/服务端都可传自己的 reader）。

    reg_read(path) -> 值（缺失应抛异常）。
    """
    cfg = {}
    try:
        cfg['basis_n'] = None
    except Exception:                              # noqa: BLE001
        pass
    waves = []
    for a in ('single', 'A', 'B'):
        try:
            waves += list(reg_read('dynamic.waves_' + a) or [])
        except Exception:                          # noqa: BLE001
            continue
    freqs = [float(w.get('freq_mhz') or 0.0) for w in waves]
    cfg['max_freq_mhz'] = max(freqs) if freqs else None
    try:
        mw_max = float(MW_MAX_MHZ)
    except Exception:                              # noqa: BLE001
        mw_max = 1e5
    cfg['has_mw'] = any(0.0 < f < mw_max for f in freqs)
    cfg['has_opt'] = any(f >= mw_max for f in freqs)
    return cfg

# ── 重型诊断（dim×dim 稠密 H 快照）的**唯一**尺寸策略 ──────────────
# 2026-09-12 实测：dim=8112 时单项 = 8112²×16 B = 1004 MiB，而快照要取 5 个
# 时刻 → 5×1 GB，直接分配失败。策略：轻量层永远采；重型层按尺寸自动决定。
H_SNAPSHOT_MAX_MB = 256.0


def h_snapshot_ok(dim):
    """重型 H 快照是否可行（按 dim² × 16 B 估）。"""
    try:
        dim = int(dim)
    except (TypeError, ValueError):
        return False
    return dim * dim * 16.0 / (1024.0 ** 2) <= H_SNAPSHOT_MAX_MB


def h_snapshot_note(dim):
    """跳过时的说明（进 diag，供 GUI/AI 显示，不再静默）。"""
    mb = (int(dim) ** 2) * 16.0 / (1024.0 ** 2)
    return {'too_large': True, 'skipped': True, 'dim': int(dim),
            'term_mb_each': round(mb, 1),
            'limit_mb': H_SNAPSHOT_MAX_MB,
            'note': 'H 快照跳过：稠密分项每项约 %.0f MB（上限 %.0f MB）；'
                    '诊断的配对/步长/扇区不受影响' % (mb, H_SNAPSHOT_MAX_MB)}
