# -*- coding: utf-8 -*-
"""耗散引擎兼容性调查 · 实测探针（2026-09-19，只读调查不改 src）。

逐项复现《耗散引擎兼容性调查》文档问题（H1–H9）。每项打印
REPRO=问题复现 / PASS=问题不存在（附数值）。运行：
    python checks/dissip_compat_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from arc import Rubidium87
from src.core import core
from src.core.atomic_levels import make_levels, square
from src.core.evolve_util import decay, auto_decays, _single_loss_rates
from src.core.sambe import evolve_sambe, evolve_sambe_pair
from src.core.evolve_mixed import evolve_mixed
from src.core.field_build import build_fields_exact, real_field
from src.core.waveforms import Wave

cache = core.ArcCache(Rubidium87())
res = []


def say(tag, name, problem, detail):
    """problem=True → 复现（记一条问题）。"""
    res.append((tag, name, bool(problem)))
    print(f'[{ "REPRO" if problem else "PASS "}] {tag} {name}: {detail}',
          flush=True)


def mk_states():
    """60S(m=+0.5) + 6P3/2(m=+1.5)（Δm=+1 允许 σ+ 近共振驱动；6P 短寿命）。"""
    return make_levels(cache, [
        dict(n=60, l=0, j=0.5, m=0.5,
             E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, 0.5, 0.0)),
        dict(n=6, l=1, j=1.5, m=1.5,
             E_mhz=core._sublevel_energy(cache, 6, 1, 1.5, 1.5, 0.0)),
    ])


st = mk_states()
t = np.linspace(0, 50, 11)
dec = [decay(1, 8.464, None)]           # 6P 自发辐射（落点=丢失）
w = Wave('微波', 7000.0, square(0, 1e12), pol='π', intensity_mwcm2=10.0)
flds = build_fields_exact(cache, st, [w])

# =====================================================================
# H1 · sambe 单原子 + decays → evolve_reference：E_shift 被静默丢弃
# =====================================================================
print('--- H1: evolve_sambe(decays, E_shift) 丢 E_shift ---')
# 关键判定（避免近共振微波撑爆 sambe 光子块 M 探针）：
#   ① 含耗散 ⇒ 走 reference-bdf（快），传/不传 E_shift 结果应逐位相同（丢弃）；
#   ② evolve_reference 签名根本无 E_shift 参数（结构性丢弃）；
#   ③ "E_shift 本应改变物理"用纯 numpy 二能级对照（不触发引擎慢路径）。
import inspect
from src.core.evolve_reference import evolve_reference as _evref
st_h1 = make_levels(cache, [
    dict(n=60, l=0, j=0.5, m=0.5,
         E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, 0.5, 0.0)),
    dict(n=60, l=1, j=1.5, m=0.5,
         E_mhz=core._sublevel_energy(cache, 60, 1, 1.5, 0.5, 0.0))])
nu_mw = abs(st_h1[1]['E_mhz'] - st_h1[0]['E_mhz'])
w_mw = Wave('微波', nu_mw + 3000.0, square(0, 1e12), pol='π',
            intensity_mwcm2=1.0)          # 弱+失谐 ⇒ M=1，快
flds_mw = build_fields_exact(cache, st_h1, [w_mw])
dec_h1 = [decay(1, 8.464, None)]
r_d_no = evolve_sambe(st_h1, flds_mw, 0, t, decays=dec_h1, E_shift=None)
r_d_sh = evolve_sambe(st_h1, flds_mw, 0, t, decays=dec_h1, E_shift=[0.0, 500.0])
d_bad = np.abs(r_d_sh['pops'] - r_d_no['pops']).max()
sig_has = 'E_shift' in inspect.signature(_evref).parameters
# numpy 对照：同哈密顿量加/不加 500MHz 对角位移，末态布居应显著不同
C = 2 * np.pi * 1e-3
Om = float(np.abs(np.asarray(flds_mw[0]['A'], float)).max())
H0 = np.array([[0.0, Om], [Om, 3000.0]])           # 失谐 3000
Hs = H0 + np.diag([0.0, 500.0])
from scipy.linalg import expm
p0 = expm(-1j * C * H0 * 50)[1, 0] ** 2
p1 = expm(-1j * C * Hs * 50)[1, 0] ** 2
d_ctl = abs(p1 - p0)
say('H1a', '含耗散时 500MHz E_shift 被丢弃（静默改物理）',
    d_bad < 1e-12 and not sig_has,
    f'max|Δpops|={d_bad:.1e}（=0 丢弃）；evolve_reference 无 E_shift 形参='
    f'{not sig_has}；backend={r_d_no.get("backend")}')
say('H1b', 'E_shift 本应改变物理（numpy 对照，非问题）',
    not (d_ctl > 1e-6),
    f'加位移前后末态布居差={d_ctl:.3f}（>0 ⇒ 参数有物理意义，丢弃即错）')

# =====================================================================
# H2 · 转参考路径丢 error_report（键契约不一致）
# =====================================================================
print('--- H2: 返回契约 ---')
r_no = evolve_sambe(st_h1, flds_mw, 0, t, decays=None)   # 无耗散→sambe 本体
has_norm = 'error_report' in r_no
has_ref = 'error_report' in r_d_no
say('H2', 'error_report：无耗散有 / 含耗散转参考后无', has_norm and not has_ref,
    f'无耗散={has_norm} 含耗散={has_ref}（消费端 res["error_report"] 会 KeyError）')

# =====================================================================
# H3 · GUI 两体自定义档：同一通道同时塞 A、B（无法单侧）
# =====================================================================
print('--- H3: 两体通道对称性（读 ui_dynamic 源码）---')
ui = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'src', 'ui', 'ui_dynamic.py')
txt = open(ui, encoding='utf-8').read()
sym = ('outA.append(ch)' in txt and 'outB.append(ch)' in txt)
say('H3', '两体自定义档把同一通道同时给 A 和 B（想只衰 A 做不到）', sym,
    f'同循环 outA+outB.append={sym}（auto 档两侧同表尚属物理正确，'
    f'但自定义档用户明确只想给一侧时无入口）')

# =====================================================================
# H4 · 两体非丢失通道（显式落点）全路径拒绝
# =====================================================================
print('--- H4: 两体显式落点拒绝 ---')
try:
    evolve_sambe_pair(st, [], [], 0, 0, t, decays_A=[decay(1, 8.464, 0)],
                      decays_B=None)
    say('H4', 'sambe 两体显式落点被拒', False, '竟然没抛异常！')
except NotImplementedError as e:
    say('H4', 'sambe 两体显式落点被拒（能力缺口，非 bug）', True,
        f'NotImplementedError: {e}')
except Exception as e:
    say('H4', 'sambe 两体显式落点(其他异常)', True,
        f'{type(e).__name__}: {e}')

# =====================================================================
# H5 · hard 报错文案指向 GUI 不存在的 'lindblad' 档
# =====================================================================
print('--- H5: 报错文案 ---')
try:
    _single_loss_rates(st, [decay(1, 8.464, 0)])
    say('H5', 'hard 显式落点被拒', False, '没抛异常')
except NotImplementedError as e:
    txt5 = str(e)
    say('H5', 'hard 报错文案含"单原子 lindblad 引擎"（GUI 方法下拉无此档）',
        'lindblad 引擎' in txt5, f'文案: {txt5}')

# =====================================================================
# H6 · auto_decays 里德堡排除 + 黑体温度 GUI 无入口
# =====================================================================
print('--- H6: auto_decays 过滤 ---')
st_rb = make_levels(cache, [
    dict(n=60, l=0, j=0.5, m=0.5,
         E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, 0.5, 0.0))])
d_rb = auto_decays(cache, st_rb)
say('H6', '60S 被 τ≤100µs 阈值排除（室温黑体跃迁无界面入口）',
    len(d_rb) == 0, f'通道数={len(d_rb)}（里德堡 ~230µs 被滤；'
    'auto_decays(temperature=) 支持但 GUI/MCP 未接线）')

# =====================================================================
# H7 · mixed 单原子 + 耗散可跑（对照，非问题）
# =====================================================================
print('--- H7: mixed 耗散 ---')
res_m = evolve_mixed(st, flds, 1, t, decays=dec)
lm = res_m.get('loss')
say('H7', 'mixed 单原子 + 耗散可跑，loss 有值（对照，非问题）',
    not (lm is not None and lm[-1] > 0),
    f'loss(T)={lm[-1]:.4f}（mixed 正常；sambe 则换后端 reference-bdf）')

# =====================================================================
# H8 · F3：real_field scatter 公式在里德堡寻址几何下取反
# =====================================================================
print('--- H8: real_field scatter（F3）---')
st_r = make_levels(cache, [
    dict(n=60, l=0, j=0.5, m=0.5,
         E_mhz=core._sublevel_energy(cache, 60, 0, 0.5, 0.5, 0.0)),
    dict(n=6, l=1, j=1.5, m=1.5,
         E_mhz=core._sublevel_energy(cache, 6, 1, 1.5, 1.5, 0.0))])
nu = st_r[0]['E_mhz'] - st_r[1]['E_mhz']
d = core.dipole_amplitude(cache, st_r[0], st_r[1], +1)
I = core.intensity_mwcm2_from_rabi(d, 100.0, 1.0)
for dg in (5000.0, 20000.0):
    _, _, sc = real_field(cache, st_r, nu + dg, I, pol='σ+',
                          return_scatter=True)
    correct = 100.0 ** 2 * 8.464 / (4 * dg * dg)      # 期望 γ(60S) MHz
    say(f'H8@{dg:.0f}', 'scatter[60S] 恒 0（损失记到 6P、乘错 Γ）',
        sc[0] < 1e-9 and sc[1] > 0,
        f'sc[60S]={sc[0]:.1e} sc[6P]={sc[1]:.1e} 期望 γ(60S)={correct:.1e}')

n_repro = sum(1 for _, _, p in res if p)
print(f'\n== 复现 {n_repro} 项问题 / 共 {len(res)} 项探针 ==')
