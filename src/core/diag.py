# -*- coding: utf-8 -*-
"""诊断摘要/模型卡的**唯一真值源**（P5：一套代码，两个渲染器）。

背景（2026-09-12）：`diag` 快照本身早已统一（`evolve_mixed[_pair]` 产出，
契约写在 `solver_inspector` 顶部），但**摘要的组装有三套**——GUI 检视器
`load_diag`、MCP `get_diag`（`server._diag_summary`）、以及新加的模型卡
（`server._res_summary` 内联块）→ 数字会分叉（模型卡里 `n_microwave=None`
而 `get_diag` 有值，就是分叉的现场）。

本模块把这些摘要收敛成一处：GUI、`get_diag`、模型卡都调 `summarize()`
与 `warnings()`。**不改变 diag 快照本身**（那是引擎的产物）。
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


def summarize(diag, cfg=None):
    """diag 快照 → 规范化摘要（GUI 摘要头 / get_diag / 模型卡共用）。

    cfg（可选）：{'basis_n':…, 'has_mw':bool, 'has_opt':bool}
    —— 用于**判定式警告**（缺数据时不猜）。
    """
    cfg = dict(cfg or {})
    if not isinstance(diag, dict):
        return {'msg': '无诊断快照（diag=True 才有）', 'warnings': []}
    meta = diag.get('meta') or {}
    solv = diag.get('solver') or {}
    prs = diag.get('pairs') or {}
    acz = diag.get('acize') or {}
    pho = diag.get('photon') or {}
    dim = meta.get('dim')
    vmax = prs.get('max_abs_v_mhz')
    # 载波频率：取配置里最大波频（用于 maxΩ/ω 判定；无配置则跳过）
    om = cfg.get('max_freq_mhz')
    ratio = None
    if vmax and om:
        try:
            ratio = round(float(vmax) / float(om), 4)
        except ZeroDivisionError:
            ratio = None
    out = {
        'method': meta.get('method'), 'dim': dim, 'N': meta.get('N'),
        'P': pho.get('P', meta.get('P')), 'M': pho.get('Ms', meta.get('M')),
        'n_pairs': prs.get('n', meta.get('n_pairs')),
        'n_microwave': prs.get('n_microwave'),
        'n_optical_vdd': prs.get('n_optical_vdd'),
        'keep': meta.get('keep'), 'drop_cap': meta.get('n_drop_cap'),
        'drop_phase': meta.get('n_drop_phase'),
        'acized_states': acz.get('n_shifted'),
        'max_abs_v_mhz': vmax,
        'max_omega_over_omega': ratio,
        'tol': meta.get('tol'), 'max_step': meta.get('max_step'),
        't_end_ns': meta.get('t_end'), 'r_crit': meta.get('r_crit'),
        'mw_cap_mhz': meta.get('mw_cap'),
        'n_steps': solv.get('n_steps'), 'nfev': solv.get('nfev'),
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
    n_mw = prs.get('n_microwave')
    # 语义：cfg 明确说"没有微波波形"时不报（修 E2 误报）；
    # cfg 未提供（未知）时保守报警——宁可多提示，不可漏掉真问题
    if n_mw is not None and int(n_mw) == 0 and cfg.get('has_mw', True):
        w.append('微波候选对为 0：基组可能缺中间态（多光子通道需要真实中间态）'
                 '或频率与能级间隔不匹配 —— 该波不会产生任何转移')
    ratio = None
    vmax = prs.get('max_abs_v_mhz')
    om = cfg.get('max_freq_mhz')
    if vmax and om:
        ratio = float(vmax) / float(om)
        if ratio > 1.0:
            w.append('maxΩ/ω=%.2f > 1：已出 RWA 适用区（物理存疑，数值未必错）'
                     % ratio)
    Ms = pho.get('Ms') or meta.get('M') or []
    try:
        from . import params as _P
        m_max = _P.get('_M_MAX')
        if Ms and m_max and max(int(x) for x in Ms) >= int(m_max):
            w.append('光子块 M 达上限 %s：截断误差可能 > tol' % m_max)
    except Exception:                              # noqa: BLE001
        pass
    if int(meta.get('n_drop_cap') or 0) > 0:
        w.append('有 %s 对因成本窗被剔除（等于改了模型，结论需交代）'
                 % meta.get('n_drop_cap'))
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
        from . import params as _P
        mw_max = float(_P.get('MW_MAX_MHZ'))
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
    return {'skipped': True, 'dim': int(dim), 'term_mb_each': round(mb, 1),
            'limit_mb': H_SNAPSHOT_MAX_MB,
            'note': 'H 快照跳过：稠密分项每项约 %.0f MB（上限 %.0f MB）；'
                    '诊断的配对/步长/扇区不受影响' % (mb, H_SNAPSHOT_MAX_MB)}
