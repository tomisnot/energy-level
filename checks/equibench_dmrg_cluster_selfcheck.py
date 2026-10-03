# -*- coding: utf-8 -*-
"""任意点云 DMRG（自写 autoMPO）标杆：多腿/无周期/逐键复耦合 ≡ 引擎 H ≡ ED。

补的缺口：旧 `dmrg.py` 只会 **均匀 2-leg 圆柱**（靠 TeNPy 晶格的平移不变近邻对），
任意点云（多腿、无周期、逐键 V/J/A、逐位 δ/Ω）进不去。现用「N 格点单胞 + 逐项
`add_coupling(n=0)`」自写 autoMPO ⇒ 任意原子对都能表达。

四路对齐（同 `equibench_dmrg` 的纪律）：
  自写 MPO 的 ExactDiag 谱 **逐元素**（按排序置换）≡ 引擎 `build_H_sparse` ≡ 独立 kron
  DMRG E0 / ⟨n_i⟩ ≡ 稠密 ED ≡ Lanczos。
其中特别钉住一个真实踩过的坑：**蛇形反向行使原子对下标颠倒，若只换下标不换算符位置，
复 J 的 σ⁺σ⁻ 会被共轭错**（实测偏差 2e-1，谱看着"差不多"却全错）。
无 tenpy 时整组 SKIP。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

from checks.checklib import Checker                              # noqa: E402
from src.core.manybody import assemble as asm                    # noqa: E402
from src.core.manybody import dmrg as dm                          # noqa: E402
from src.core.manybody import equilibrium as eq                  # noqa: E402
from src.core.manybody import eq_route as R                      # noqa: E402

C = Checker('equibench_dmrg_cluster')

try:
    import tenpy  # noqa: F401
    HAS_TENPY = True
except Exception:
    HAS_TENPY = False


# --------------------------------------------------------------- 造例工具
def calib_of(atoms, pairs, terms=('I1', 'I2', 'I3')):
    return dict(template='ising', terms=list(terms), atoms=atoms, pairs=pairs,
                dropped=[], dropped_amp={}, single_basis=[])


def pair(i, j, V, J, A):
    return dict(i=i, j=j, V=complex(V), J=complex(J), A=complex(A),
                raw_V=complex(V), raw_J=complex(J), raw_A=complex(A),
                delta={}, info={})


def grid(rows, cols, sp=6.0, jitter=0.0, seed=7):
    """行=腿（y 方向），列=链（x 方向）。jitter 要远小于 sp 才保得住分行。"""
    rng = np.random.default_rng(seed)
    atoms, k = [], 0
    for r in range(rows):
        for c in range(cols):
            x = sp * c + (jitter * rng.standard_normal() if jitter else 0.0)
            y = sp * r + (jitter * rng.standard_normal() if jitter else 0.0)
            atoms.append(dict(id=k, pos=(x, y, 0.0)))
            k += 1
    return atoms


def eng_dense(cal, Om, dl):
    return np.asarray(asm.build_H_sparse(cal, Om, dl).todense(), dtype=complex)


def perm_array(order):
    """m(MPS 序，site0=MSB) → s(引擎原子序，原子0=MSB)。"""
    n = len(order)
    ms = np.arange(1 << n, dtype=np.int64)
    s = np.zeros(1 << n, dtype=np.int64)
    for k in range(n):
        b = (ms >> (n - 1 - k)) & 1
        s |= b << (n - 1 - int(order[k]))
    return s


def nn_bonds(atoms, rows, cols, sp=6.0):
    """格点最近邻键：逐键**互不相同**的复 V/J/A（平移不变都打破，逼出逐键通路）。

    只发 n=len(atoms) 以内的原子（支持“矩形被截掉尾巴”的非矩形点云）。
    """
    n = len(atoms)
    pairs = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i >= n:
                continue
            j1 = i + 1
            if c + 1 < cols and j1 < n:
                pairs.append(pair(i, j1, 1.4 - 0.3 * c + 0.05 * r,
                                  0.2 + 0.1j * c, 0.05j * (c + 1) - 0.01 * r))
            j2 = i + cols
            if r + 1 < rows and j2 < n:
                pairs.append(pair(i, j2, 0.9 + 0.2 * r,
                                  -0.15j * (r + 1), 0.03 * (r + c)))
    return pairs


# ================= 1) 排序与项生成（纯函数，不跑 DMRG） =================
if not HAS_TENPY:
    C.skip('任意点云 DMRG（自写 autoMPO）', 'tenpy 不可用')
    sys.stdout.flush()
    os._exit(0 if C.summary() else 1)

a_34 = grid(3, 4)
ord34, info34 = dm.cylinder_order([a['pos'] for a in a_34])
# 3 腿×4 列（原子号 = r*4+c）⇒ 梯级优先顺序 = 列优先、腿最快
_exp_order = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
_nn = nn_bonds(a_34, 3, 4)
_rng = max(abs(info34['perm'][p['j']] - info34['perm'][p['i']]) for p in _nn)
subs_o = [
    ('3×4 网格 → 3 腿、每腿 4 位（%s/%s）' % (info34['n_legs'], info34['leg_len']),
     info34['n_legs'] == 3 and info34['leg_len'] == [4, 4, 4]),
    ('order 是合法置换（覆盖 0..11 各一次）',
     sorted(ord34) == list(range(12))),
    ('梯级优先顺序确切为 %s' % _exp_order, ord34 == _exp_order),
    ('最近邻键的最大程 = 腿数（%d ≤ 3，与链长无关）；蛇形会涨到 2L−1' % _rng,
     _rng <= 3),
    ('全部同 y（单腿链）→ 1 腿', dm.cylinder_order([(6.0 * i, 0.0, 0.0)
                                                   for i in range(5)])[1]['n_legs'] == 1),
    ('轻微抖动（≪腿距）仍分对行', dm.cylinder_order(
        [a['pos'] for a in grid(3, 3, jitter=0.05)])[1]['n_legs'] == 3),
]
# terms 门控：只开 I1 ⇒ 只应有 Nhat-Nhat 项
c_i1 = calib_of(a_34, nn_bonds(a_34, 3, 4), terms=('I1',))
_o, _ons, coups_i1, _inf = dm.terms_from_calib(c_i1, [1.0] * 12, [0.0] * 12)
c_all = calib_of(a_34, nn_bonds(a_34, 3, 4))
_o2, _ons2, coups_all, _inf2 = dm.terms_from_calib(c_all, [1.0] * 12, [0.0] * 12)
subs_o.append(('terms=(I1,) 只出 Nhat·Nhat 项（%d/%d 条）'
               % (len(coups_i1), len(coups_all)),
               all(opA == 'Nhat' and opB == 'Nhat' for (_c, _a, opA, _b, opB) in coups_i1)
               and len(coups_i1) < len(coups_all)))
# 每条耦合的 MPS 下标必须已归一 a<=b（autoMPO 的前提）
subs_o.append(('couplings 全部 a ≤ b（已按 MPS 下标归一）',
               all(a <= b for (_c, a, _o1, b, _o2_) in coups_all)))
C.group('排序与项生成：cylinder_order / terms_from_calib', subs_o)


# ================= 2) 自写 MPO ≡ 引擎 H（置换等价，逐元素） =================
CASES = [
    ('3leg×3 复耦合', grid(3, 3), 3, 3),
    ('2leg 非矩形(9原子)', grid(2, 5)[:9], 2, 5),
    ('4leg×2 微抖', grid(4, 2, jitter=0.05), 4, 2),
]
subs_m = []
for name, atoms, rows, cols in CASES:
    n = len(atoms)
    Om = list(0.9 + 0.25 * np.arange(n))
    dl = list(-0.4 + 0.3 * np.arange(n))
    cal = calib_of(atoms, nn_bonds(atoms, rows, cols))
    order, onsite, coups, info = dm.terms_from_calib(cal, Om, dl)
    Hm, nm = dm.rydberg_cluster_full_hamiltonian(onsite, coups)
    He = eng_dense(cal, Om, dl)
    p = perm_array(order)
    err = float(np.abs(Hm - He[np.ix_(p, p)]).max())
    herm = float(np.abs(Hm - Hm.conj().T).max())
    subs_m.append(('%s：MPO ≡ 引擎 H（逐元素，err=%.1e，n=%d，腿=%d）'
                   % (name, err, n, info['n_legs']), err < 1e-10 and herm < 1e-12))

# —— 专钉"复 J 交换陷阱"：只留 J（含复部）且必须有反向行 ⇒ 旧写法在此必错
atoms_j = grid(2, 4)
pairs_j = []
for r in range(2):
    for c in range(3):
        pairs_j.append(pair(r * 4 + c, r * 4 + c + 1, 0.0, 0.3 + 0.4j, 0.0))
for c in range(4):
    pairs_j.append(pair(c, 4 + c, 0.0, -0.2j * (c + 1), 0.0))
cal_j = calib_of(atoms_j, pairs_j, terms=('I2',))
oj, ons_j, cp_j, inf_j = dm.terms_from_calib(cal_j, [1.1] * 8, [-0.2] * 8)
Hj, _ = dm.rydberg_cluster_full_hamiltonian(ons_j, cp_j)
poj = perm_array(oj)
Ej = eng_dense(cal_j, [1.1] * 8, [-0.2] * 8)
err_j = float(np.abs(Hj - Ej[np.ix_(poj, poj)]).max())
subs_m.append(('**复 J 专例**（只有 σ⁺σ⁻、含反向行）：置换后逐元素 err=%.1e'
               '（旧"只换下标"写法此处分叉到 2e-1）' % err_j, err_j < 1e-12))
C.group('自写 autoMPO ≡ 引擎 build_H_sparse（按排序置换，逐元素）', subs_m)


# ================= 3) DMRG ≡ ED ≡ Lanczos（含 ⟨n_i⟩） =================
subs_d = []
for name, atoms, rows, cols in [('3leg×3 (n=9)', grid(3, 3), 3, 3),
                                ('2leg 非矩形 (n=9)', grid(2, 5)[:9], 2, 5)]:
    n = len(atoms)
    Om = list(0.9 + 0.25 * np.arange(n))
    dl = list(-0.4 + 0.3 * np.arange(n))
    cal = calib_of(atoms, nn_bonds(atoms, rows, cols))
    Hs = asm.build_H_sparse(cal, Om, dl)
    gs = eq.ground_state(Hs, dense_max=1 << n)          # 只算一次（旧写法算了三遍）
    E_ed = gs['E0']
    n_ed = eq.bit_matrix(n, 1 << n) @ (np.abs(gs['vec']) ** 2)
    r = dm.dmrg_rydberg_cluster(cal, Om, dl, chi_max=48, truncerr=1e-8,
                                max_sweeps=30)
    subs_d.append(('%s：DMRG E0 ≡ 稠密 ED（Δ=%.2e，MPO 键维=%d）'
                   % (name, abs(r['E0'] - E_ed), r['mpo_bond_dim']),
                   abs(r['E0'] - E_ed) < 1e-8))
    subs_d.append(('%s：DMRG ⟨n_i⟩ ≡ ED（max Δ=%.2e）'
                   % (name, np.abs(r['n_atom'] - n_ed).max()),
                   np.abs(r['n_atom'] - n_ed).max() < 1e-7))
    rl = R.solve_equilibrium(dict(calib=cal, Om=Om, delta=dl, n_sites=n),
                             want='gs', backend='dmrg', chi_max=48)
    subs_d.append(('%s：经 eq_route(backend=dmrg，**无 ladder**) 一致'
                   % name, abs(rl['E0'] - E_ed) < 1e-8
                   and rl['backend'] == 'dmrg' and len(rl['n_atom']) == n))
    r_lz = R.solve_equilibrium(dict(calib=cal, Om=Om, delta=dl, n_sites=n),
                               want='gs', backend='lanczos')
    subs_d.append(('%s：三方一致 DMRG ≡ ED ≡ Lanczos' % name,
                   abs(r_lz['E0'] - E_ed) < 1e-8 and abs(rl['E0'] - r_lz['E0']) < 1e-8))
C.group('DMRG ≡ 稠密 ED ≡ Lanczos（任意点云，含逐原子 ⟨n_i⟩）', subs_d)

# ---- 大一点的 2D 柱交叉验（截断到近邻 ⇒ 梯级程=腿数，DMRG 才划得来）----
# n=12 时稠密 ED 还能算（4096 个基矢），故拿 **三法**对齐；更大尺寸（n=16 需 68 GB
# 稠密、n=20 连稀疏装配都撞内存）不在判据里跑——那部分只当性能叙事，不占测试时间。
a_big = grid(3, 4)
pairs_big = nn_bonds(a_big, 3, 4)
cal_big = calib_of(a_big, pairs_big)
Om_big = [1.0] * 12
dl_big = [-0.3] * 12
_o_big, _ons_big, _cp_big, _info_big = dm.terms_from_calib(cal_big, Om_big, dl_big)
_rng_big = max(abs(b - a) for (_c, a, _x, b, _y) in _cp_big)
_r_d = dm.dmrg_rydberg_cluster(cal_big, Om_big, dl_big, chi_max=48,
                               truncerr=1e-8, max_sweeps=30)
E_ed_big = eq.ground_state(asm.build_H_sparse(cal_big, Om_big, dl_big),
                           dense_max=1 << 12)['E0']
C.group('三法交叉：3×4 (n=12) DMRG ≡ 稠密 ED ≡ 引擎（梯级优先排序生效）', [
    ('DMRG ≡ 稠密 ED（Δ=%.2e < 1e-6）' % abs(_r_d['E0'] - E_ed_big),
     abs(_r_d['E0'] - E_ed_big) < 1e-6),
    ('梯级优先排序：最大程 == 腿数（%d）' % _rng_big, _rng_big == 3),
    ('MPO 键维有界（%d ≤ 16，长程键不多）' % _r_d['mpo_bond_dim'],
     _r_d['mpo_bond_dim'] <= 16),
])


# ================= 4) 路由决策 + 变分收敛 =================
subs_r = [
    ('route: dmrg_ok=True 大尺寸→dmrg',
     R.route_equilibrium(20, 'gs', gpu=False, dmrg_ok=True) == 'dmrg'),
    ('route: dmrg_ok=False 大尺寸→lanczos',
     R.route_equilibrium(20, 'gs', gpu=False, dmrg_ok=False) == 'lanczos'),
    ('route: 不传 dmrg_ok 时仍跟随 is_ladder（向后兼容）',
     R.route_equilibrium(20, 'gs', gpu=False, is_ladder=True) == 'dmrg'
     and R.route_equilibrium(20, 'gs', gpu=False) == 'lanczos'),
    ('route: 小尺寸 dmrg_ok=True 仍走 ed（不被抢走）',
     R.route_equilibrium(8, 'gs', gpu=False, dmrg_ok=True) == 'ed'),
]
C.group('路由决策：dmrg_ok 新参数（不破坏 is_ladder 语义）', subs_r)

# 变分性：chi 增大 E0 不升，且向 ED 收敛


def _guard_raises():
    """故意用小 chi + 紧 truncerr 且不放宽守卫 → TeNPy 必须**报错**而非静默降级。"""
    try:
        dm.dmrg_rydberg_cluster(cal_c, Om_c, dl_c, chi_max=4, truncerr=1e-12)
        return False
    except Exception:                                    # noqa: BLE001
        return True

atoms_c = grid(2, 5)
nc = 10
Om_c = [1.0] * nc
dl_c = [-0.3] * nc
cal_c = calib_of(atoms_c, nn_bonds(atoms_c, 2, 5))
E_ed_c = eq.ground_state(asm.build_H_sparse(cal_c, Om_c, dl_c),
                         dense_max=1 << nc)['E0']
E4 = dm.dmrg_rydberg_cluster(cal_c, Om_c, dl_c, chi_max=8, truncerr=1e-6,
                             max_sweeps=20, max_trunc_err=None)
E_ch4 = E4['E0']
E16r = dm.dmrg_rydberg_cluster(cal_c, Om_c, dl_c, chi_max=16, truncerr=1e-6,
                               max_sweeps=20, max_trunc_err=None)
E16_trunc = E16r['truncerr']
E_ch16 = E16r['E0']
E_ch64 = dm.dmrg_rydberg_cluster(cal_c, Om_c, dl_c, chi_max=64, truncerr=1e-8,
                                 max_sweeps=30)['E0']
C.group('键维收敛：chi↑ ⇒ E0 单调不升、误差估计下降、趋 ED', [
    ('chi 8(%g) ≥ chi 16(%g) ≥ chi 64(%g)（变分序）'
     % (E_ch4, E_ch16, E_ch64),
     E_ch4 >= E_ch16 - 1e-10 and E_ch16 >= E_ch64 - 1e-10),
    ('截断误差随 chi 降（%s → %s）'
     % (E4['truncerr'], E16_trunc), E4['truncerr'] > E16_trunc),
    ('chi=64 已达 ED（Δ=%.2e < 1e-7）' % abs(E_ch64 - E_ed_c),
     abs(E_ch64 - E_ed_c) < 1e-7),
    ('小 chi 硬卡 truncerr 会被 TeNPy 拦住（不静默降级）',
     _guard_raises()),
])

# ================= 6) 产品接线：eq_engine 能把 dmrg 调起来（旧桩 _geom_is_ladder≡False） =========
from src.core.manybody import eq_engine as EE                    # noqa: E402


def _spec(cal, Om, dl, method, dmrg_ok):
    n = len(cal['atoms'])
    return dict(version=1, nbody='many', equilibrium=True, two_body=False,
                want='gs', method=method, method_in=method, betas=None,
                calib=cal, n_atoms=n, dim=1 << n, Om=list(Om), delta=list(dl),
                gamma=None, terms=list(cal['terms']), template='ising', kpm={},
                dmrg=dict(chi_max=48, truncerr=1e-8, max_sweeps=30),
                states=[{'label': 'atom%d' % a['id']} for a in cal['atoms']],
                labels=['atom%d' % a['id'] for a in cal['atoms']],
                warnings=[], ed_max_dim=1 << 20, dmrg_ok=dmrg_ok)


# 规整网格（NN 键、跨切割少）→ _dmrg_ok 应为 True；全对全稠密（跨切割 ∼n²/8）→ 应 False
a_ok = grid(2, 4)
cal_ok = calib_of(a_ok, nn_bonds(a_ok, 2, 4))
_ok_om, _ok_dl = [1.0] * 8, [-0.3] * 8
flag_ok = EE._dmrg_ok(cal_ok, _ok_om, _ok_dl)
# 反例：所有原子对都键上（长程库仑级别密度）⇒ auto 不该选 DMRG
pairs_all = [pair(i, j, 0.6, 0.02j, 0.0)
             for i in range(8) for j in range(i + 1, 8)]
cal_dense = calib_of(grid(2, 4), pairs_all)
flag_dense = EE._dmrg_ok(cal_dense, _ok_om, _ok_dl)
# 单腿（所有同 y）也不是“柱”，不该走 DMRG（交给 ed/lanczos）
chain_cal = calib_of(grid(1, 6), nn_bonds(grid(1, 6), 1, 6))
flag_chain = EE._dmrg_ok(chain_cal, [1.0] * 6, [-0.3] * 6)

res_prod = EE.run_equilibrium(_spec(cal_ok, _ok_om, _ok_dl, 'dmrg', True))
E_ed_ok = eq.ground_state(asm.build_H_sparse(cal_ok, _ok_om, _ok_dl),
                          dense_max=1 << 8)['E0']
res_auto = EE.run_equilibrium(_spec(cal_ok, _ok_om, _ok_dl, 'auto', True))
# 旧行为：dim=256 ≤ dense_max ⇒ auto 应仍走 ed（DMRG 不抢小尺寸）
C.group('产品接线：eq_engine.run_equilibrium → dmrg（任意点云）', [
    ('_dmrg_ok：规整 2×4 梯子（跨切割≤4）→ True', flag_ok is True),
    ('_dmrg_ok：28 键全对全（跨切割~16）→ False（不自动上 DMRG）',
     flag_dense is False),
    ('_dmrg_ok：单腿链（n_legs=1）→ False', flag_chain is False),
    ('显式 method=dmrg → Result(backend=dmrg, E0 ≡ ED, 契约键齐)',
     res_prod['backend'] == 'dmrg' and abs(res_prod['E0'] - E_ed_ok) < 1e-8
     and {'want', 'method', 'n_sites', 'E0', 'n_atom', 'states'} <= set(res_prod)),
    ('method=dmrg 时 order/n_legs 随 Result 透出（可审计排序）',
     res_prod.get('n_legs') == 2 and len(res_prod.get('order')) == 8),
    ('auto + 小尺寸（dim=256）仍走 ed，不被 DMRG 抢走',
     res_auto['backend'] == 'ed' and abs(res_auto['E0'] - E_ed_ok) < 1e-9),
    ('Result 携 method=want（分派键）且 dmrg 结果含 chi',
     res_prod['method'] == 'gs' and res_prod.get('chi', 0) >= 1),
])

sys.stdout.flush()
os._exit(0 if C.summary() else 1)
