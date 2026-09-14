# -*- coding: utf-8 -*-
"""freeze 参数优化 —— 黑盒求值器（给 AutoTuner 项目调用）。

把"Rb-87 里德堡原子 + freeze 微波脉冲"的含时演化包装成一个黑盒函数：
    输入：脉冲参数 (频率, 光强, 偏振)      ← 你要优化的东西
    输出：若干标量指标（对比度 / 58G 收率 / 60S 保留）  ← 你的目标与约束

用法（单进程，处理一批点；并行由你按点切分到多进程）：
    python scripts/freeze_blackbox.py --points pts.json --out res.json
    python scripts/freeze_blackbox.py --points pts.json --out res.json --fast
    python scripts/freeze_blackbox.py --points pts.json --out res.json --box 5 0.2
      ↑ --box dω_MHz dI_rel：在每个点周围取 (ω±dω, I×(1±dI)) 的 3×3 抖动盒，
        额外返回最坏情况的 C_dual（"平坦性"口径）

输入 pts.json（list，每项一个候选点）：
    [{"id": 0, "freq_mhz": 7380.0, "intensity": 1.0,
      "alpha_deg": 0.0, "theta_deg": 0.0, "eps_deg": 0.0},
     {"id": 1, "freq_mhz": 7320.0, "intensity": 0.5,
      "alpha_deg": 0.0, "theta_deg": 0.0, "eps_deg": 45.0}]   # 45° → 纯 σ+
    · 光学偏振**只用三个连续角**表示（覆盖全部偏振态，含纯偏振）：
        alpha_deg ∈ [0, 90]    传播方向与 z(B) 的夹角
        theta_deg ∈ [0, 180)   椭圆主轴方位角（周期 180°）
        eps_deg   ∈ [-45, +45] 椭偏角（0=线偏，±45=圆偏）
      特例（可当搜索初值/锚点）：线偏⊥B=(0,0,0)；σ+=(0,0,45)；σ-=(0,0,-45)；
      纯 π(E‖z)=(90,0,0)
    · 可选 "fwhm_ns"（默认 40.0）、"center_ns"（默认 3σ = 51.0）；建议固定

输出 res.json（list，每项一条）：
    {"id":0, "ok":true,
     "C_m05":…, "C_p15":…, "C_dual":min(…),   ← 目标：C_dual 越大越好
     "S_ret":…,                                ← 约束：60S 保留（不应被打走）
     "kick_m05":…, "kick_p15":…,               ← 58G 收率（踢出效率）
     "surv_m05":…, "surv_p15":…,               ← 60P 残留
     "C_dual_worst":…, "S_ret_worst":…         ← 仅 --box 时给（抖动盒最坏值）
    }
说明：一次求值 = 3 次单体含时演化（初态 60S / 60P m=-0.5 / 60P m=+1.5）。
      **衡量的是脉冲打完之后的稳定末态**（窗口 150 ns，脉冲中心 51 ns、FWHM 40 ns）。
"""
import argparse
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np                                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arc import Rubidium87                                       # noqa: E402
from src.core import static_atom as core                         # noqa: E402
from src.core.atomic_levels import make_levels                   # noqa: E402
from src.core.waveforms import Wave, GaussianEnv                 # noqa: E402
from src.core.field_build import build_fields_exact              # noqa: E402
from src.core.evolve_mixed import evolve_mixed                   # noqa: E402


def _patch_arc_literature_dme():
    """ARC 并发修复：_readLiteratureValues 对共享 sqlite 无条件 DROP+重建。

    8 个 worker 并行时，A 进程 DROP 的瞬间 B 进程 SELECT ->
    'OperationalError: no such table: literatureDME'（实测随机吃掉 ~3% 评估点）。
    文献 DME 是静态数据：表已存在且非空即无需重建。守卫掉 DROP，消除竞争。
    冷启动首次建表仍走原逻辑（此时无并发读者，或读者命中异常由 stagger 错峰）。
    """
    try:
        from arc.alkali_atom_functions import AlkaliAtom
    except Exception:                                            # noqa: BLE001
        return
    if getattr(AlkaliAtom, '_dme_guarded', False):
        return
    _orig = AlkaliAtom._readLiteratureValues

    def guarded(self):
        try:
            c = self.conn.cursor()
            c.execute("SELECT COUNT(*) FROM literatureDME")
            if c.fetchone()[0] > 0:
                return                      # 表已建好且非空：跳过 DROP+重建
        except Exception:                                    # noqa: BLE001
            pass                            # 表不存在/空/锁 -> 回退原始建表逻辑
        return _orig(self)

    AlkaliAtom._readLiteratureValues = guarded
    AlkaliAtom._dme_guarded = True


_patch_arc_literature_dme()

B_GAUSS = 21.7
# 标准基组 = "W=2 壳层"（61 组 / N=394）。见 docs/freeze优化-AutoTuner交接.md
RUNGS = [(60, 1), (59, 2), (58, 3), (58, 4)]
T_SNAP = 150.0            # 演化窗口（ns）：>120 ns 已无场，取值即"打完的稳定末态"


def standard_groups(W=2):
    seen, out = set(), []
    for (nr, lr) in RUNGS:
        for lp in (lr - 1, lr + 1):
            if lp < 0:
                continue
            for n in range(max(1, nr - W), nr + W + 1):
                for jp in (lp - 0.5, lp + 0.5):
                    if jp < 0.5 or (n, lp, jp) in seen:
                        continue
                    seen.add((n, lp, jp))
                    out.append((n, lp, jp))
    return out


_CACHE = {}


def get_basis(W=2):
    if ('states_W', W) not in _CACHE:
        cache = core.ArcCache(Rubidium87())
        specs = []
        for (n, l, j) in standard_groups(W):
            for m in np.arange(-j, j + 1e-9, 1.0):
                specs.append(dict(n=n, l=l, j=j, kind='mJ', m=float(m),
                                  E_mhz=float(core._sublevel_energy(
                                      cache, n, l, j, m, B_GAUSS))))
        states = make_levels(cache, specs)
        _CACHE['cache'] = cache
        _CACHE[('states_W', W)] = states
        _CACHE[('idx_W', W)] = {(s['n'], s['l'], float(s['j']),
                                 round(float(s['m']), 2)): i
                                for i, s in enumerate(states)}
        _CACHE[('fam_W', W)] = {
            '60S': [i for i, s in enumerate(states)
                    if (s['n'], s['l']) == (60, 0)],
            '60P': [i for i, s in enumerate(states)
                    if (s['n'], s['l']) == (60, 1)],
            '58G': [i for i, s in enumerate(states)
                    if (s['n'], s['l']) == (58, 4)],
            '59D': [i for i, s in enumerate(states)
                    if (s['n'], s['l']) == (59, 2)],
            '58F': [i for i, s in enumerate(states)
                    if (s['n'], s['l']) == (58, 3)],
            '57D': [i for i, s in enumerate(states)
                    if (s['n'], s['l']) == (57, 2)],
        }
    return (_CACHE['cache'], _CACHE[('states_W', W)], _CACHE[('idx_W', W)],
            _CACHE[('fam_W', W)])


INITS = {
    'ret': ((60, 0, 0.5, 0.5), '60S_{1/2} m_J=+0.5'),
    'sm': ((60, 1, 1.5, -0.5), '60P_{3/2} m_J=-0.5'),
    'sp': ((60, 1, 1.5, 1.5), '60P_{3/2} m_J=+1.5'),
}


POL_ANGLES = {'线偏⊥B': (0.0, 0.0, 0.0), 'σ+': (0.0, 0.0, 45.0),
              'σ-': (0.0, 0.0, -45.0), 'π': (90.0, 0.0, 0.0)}


def eval_one(freq_mhz, intensity, alpha_deg=0.0, theta_deg=0.0, eps_deg=0.0,
             fwhm_ns=40.0, center_ns=None, fast=False, W=2):
    """单点求值 → 指标 dict（3 次演化）。

    **偏振 = 三个连续角**（唯一入口，全部偏振态都覆盖，含纯偏振）：
        alpha_deg : 传播方向与量化轴 z(B) 的夹角，[0, 90]
        theta_deg : 偏振椭圆主轴方位角，[0, 180)（周期 180°）
        eps_deg   : 椭偏角，[-45, +45]（0=线偏，±45=圆偏）
        特例：线偏⊥B=(0,0,0)；σ+=(0,0,45)；σ-=(0,0,-45)；纯 π=(90,0,0)
    引擎侧以 pol='椭圆偏' + elliptical 三元组传入（这是任意偏振的唯一通道）。
    """
    for nm, v, lo, hi in (('alpha_deg', alpha_deg, 0.0, 90.0),
                          ('theta_deg', theta_deg, 0.0, 180.0),
                          ('eps_deg', eps_deg, -45.0, 45.0)):
        if not (lo - 1e-9 <= float(v) <= hi + 1e-9):
            raise ValueError('%s=%r 超出物理区间 [%g, %g]（注意 θ 周期 180°，'
                             '不要用负值或 >180 的等价写法）' % (nm, v, lo, hi))
    ell = dict(alpha_deg=float(alpha_deg), theta_deg=float(theta_deg),
               eps_deg=float(eps_deg))
    cache, states, IDX, FAM = get_basis(W)
    sig = float(fwhm_ns) / 2.3548
    cen = float(center_ns) if center_ns is not None else 3.0 * sig
    w = Wave('f', float(freq_mhz), GaussianEnv(cen, sig, 1.0), pol='椭圆偏',
             intensity_mwcm2=float(intensity), elliptical=ell)
    fld = build_fields_exact(cache, states, [w])
    npts = 301 if fast else 1501
    tol = 1e-6 if fast else 1e-7
    t = np.linspace(0.0, T_SNAP, npts)
    out = {'own': {}, 'fam': {}}
    for tag, (key, _lab) in INITS.items():
        i0 = IDX[key]
        psi0 = np.zeros(len(states), complex)
        psi0[i0] = 1.0
        p = np.real(evolve_mixed(states, fld, psi0, t, tol=tol)['pops'])[-1]
        out['own'][tag] = float(p[i0])
        out['fam'][tag] = {k: float(p[v].sum()) for k, v in FAM.items()}
    r = {'S_ret': out['own']['ret'],
         'surv_m05': out['own']['sm'], 'surv_p15': out['own']['sp'],
         'kick_m05': out['fam']['sm']['58G'],
         'kick_p15': out['fam']['sp']['58G'],
         'leak_57D_ret': out['fam']['ret']['57D'],
         'C_m05': out['own']['ret'] - out['own']['sm'],
         'C_p15': out['own']['ret'] - out['own']['sp']}
    r['C_dual'] = min(r['C_m05'], r['C_p15'])
    return r


def eval_point(pt, fast=False, box=None, W=2):
    # 旧接口（pol/ell）硬拒绝：否则会被静默忽略 → pol="π" 变成线偏⊥B，白搜一轮
    for legacy in ('pol', 'polarization', 'ell'):
        if legacy in pt:
            raise ValueError(
                '字段 %r 已废弃：偏振现在只用三个连续角 alpha_deg / theta_deg / '
                'eps_deg 表示（等价写法：pol="π" → (90,0,0)；"σ+" → (0,0,45)；'
                '"σ-" → (0,0,-45)；"线偏⊥B/椭圆偏(0,0,0)" → (0,0,0)）' % legacy)
    kw = dict(freq_mhz=pt['freq_mhz'], intensity=pt['intensity'],
              alpha_deg=pt.get('alpha_deg', 0.0),
              theta_deg=pt.get('theta_deg', 0.0),
              eps_deg=pt.get('eps_deg', 0.0),
              fwhm_ns=pt.get('fwhm_ns', 40.0), center_ns=pt.get('center_ns'),
              fast=fast, W=W)
    res = eval_one(**kw)
    res['ok'] = True
    if box:
        dw, di = float(box[0]), float(box[1])
        worst_c, worst_s = None, None
        for d_om in (-dw, 0.0, dw):
            for d_i in (-di, 0.0, di):
                if d_om == 0.0 and d_i == 0.0:
                    continue
                q = eval_one(freq_mhz=pt['freq_mhz'] + d_om,
                             intensity=pt['intensity'] * (1.0 + d_i),
                             **{k: v for k, v in kw.items()
                                if k not in ('freq_mhz', 'intensity')})
                worst_c = q['C_dual'] if worst_c is None else min(worst_c,
                                                                  q['C_dual'])
                worst_s = q['S_ret'] if worst_s is None else min(worst_s,
                                                                 q['S_ret'])
        res['C_dual_worst'] = min(res['C_dual'],
                                  worst_c if worst_c is not None
                                  else res['C_dual'])
        res['S_ret_worst'] = min(res['S_ret'],
                                 worst_s if worst_s is not None
                                 else res['S_ret'])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fast', action='store_true',
                    help='npts=301/tol=1e-6（约为标准设置 1/3 成本，'
                         '结果与标准设置差 ~1e-6，见交接文档 §5）')
    ap.add_argument('--box', nargs=2, type=float, default=None,
                    metavar=('dOMEGA_MHZ', 'dI_REL'),
                    help='抖动盒：返回最坏情况 C_dual_worst / S_ret_worst')
    ap.add_argument('--basis', type=int, default=2,
                    help='壳层宽度 W：2=标准配置（61 组 / N=394，已证收敛）；'
                         '3=再加一层（83 组 / N=538），用于 I>1 时的复核')
    args = ap.parse_args()
    with open(args.points, encoding='utf-8') as f:
        pts = json.load(f)
    if isinstance(pts, dict):
        pts = [pts]
    t0 = time.time()
    out = []
    for k, pt in enumerate(pts):
        try:
            r = eval_point(pt, fast=args.fast, box=args.box, W=args.basis)
            r['id'] = pt.get('id', k)
        except Exception as e:                                   # noqa: BLE001
            r = {'id': pt.get('id', k), 'ok': False,
                 'err': '%s: %s' % (type(e).__name__, str(e)[:160])}
        out.append(r)
        print('[%d/%d] id=%s C_dual=%.4f C(-0.5)=%.4f C(+1.5)=%.4f '
              'S_ret=%.4f kick=%.4f/%.4f  (%.0fs, %.1fs/点)'
              % (k + 1, len(pts), r.get('id'),
                 r.get('C_dual', float('nan')), r.get('C_m05', float('nan')),
                 r.get('C_p15', float('nan')), r.get('S_ret', float('nan')),
                 r.get('kick_m05', float('nan')), r.get('kick_p15',
                                                        float('nan')),
                 time.time() - t0, (time.time() - t0) / (k + 1)), flush=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('结果 → %s（%d 点，总 %.0fs）' % (args.out, len(out),
                                            time.time() - t0))


if __name__ == '__main__':
    main()
