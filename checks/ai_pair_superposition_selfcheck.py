# -*- coding: utf-8 -*-
"""自检：两体纠缠初态接口（T1 / docs/叠加态初态-接口设计.md §2.2）。

物理动因：`|60P_A,60S_B⟩ ↔ |60S_A,60P_B⟩` 的交换结果是**纠缠态**，
逐原子直积 |ψ_A⟩⊗|ψ_B⟩ 写不出来 → 必须有 `set_pair_superposition` +
`psi0_pair` 入参这条通道（freeze 验证的前置）。

## 使用者（物理侧）2026-09-12 三态对拍结论（已互验，勿重复怀疑）
配置：3 组基（60S+60P_3/2+6P_3/2）→ N=22、两体 dim=484、n_pairs=50、P=1、
R=12 µm/θ=30°、mixed、T=400 ns、**无波形**（纯 V_dd 交换）：

| 初态 | P_PS(0) | P_SP(0) | d(t)=P_PS−P_SP |
|---|---|---|---|
| 直积 \|P_A S_B⟩ | 1.0000 | 0.0000 | +1.0000 → −0.83（400 ns） |
| (\|PS⟩+i\|SP⟩)/√2 | 0.5000 | 0.5000 | 0 → +0.99 |
| (\|PS⟩+\|SP⟩)/√2 | 0.5000 | 0.5000 | **恒 0（精确暗态，逐点一个 bit 都不动）** |

判决量：`d1²+d2²` 在 t=0 精确 = 1.0000，其亏缺与布居漏出**同步**；
`d3(t)=0.00000±0.00000`。→ `psi_pair` 确实成为初态、**相对相位 i 被忠实保留**、
两体接口与逐原子直积在物理上自洽。本自检 E4/E5 就是这条判据的锚。

## ⚠ 方法论警告（使用者实测踩过的坑，别再踩）
**两体交换的判据不要用"单频余弦网格拟合"**：对 (\|PS⟩+i\|SP⟩) 那组，拟合会给出
**6.67 MHz / B=0.45 / φ=−93.4°** 这种"看起来完美方相"的结果，而**原始序列里根本
没有 6.67 MHz 分量**（真实行为是慢的非正弦演化：交换半周期 ≈ 415 ns → V≈0.5 MHz；
直积组抖动周期 ~50 ns、幅度仅 0.04）。拟合会锁到假主频并给出自洽假象。
→ **用不依赖函数形式的判决量**：`d1²+d2²≈1` 与 `d3≡0`。

覆盖（headless 为主，GUI 段内存不足自动 SKIP）：
  A. build_pair_sup_spec（纯函数）：归一前范数 / 占据谱 / mag+phase 形态 /
     裸 pair 形态 / 四类非法输入报错
  B. _as_pair_vector：索引映射 k=iA*N+iB **逐位**、归一、相对相位、
     未归一幅度、缺项报错（点名缺失项）、非 pair_sup 形态报错
  C. pair_psi0_vector：单分量 spec ≡ 逐原子直积（逐位）；psi0_pair 优先
  D. 引擎签名接线：evolve_engine / *_pair 都有 psi0_pair；n_body=1 拒绝
  E. **真引擎**（mixed，合成基组 + 解析 V_dd，20 ns 交换窗）：
     纠缠初态 50/50、布居守恒、两体子空间封闭、确有交换摆动、
     单分量 spec 与直积路径**逐位一致**、psi0_pair 优先于 psi0_A/B
  F. AI/GUI 面（offscreen 构窗；MemoryError → SKIP）：dynamic.psi_pair 在册、
     set_pair_superposition 往返、面板 _psi0 取到、set_initial_state 清除它、
     基组变更失效告警、param 快照/回填往返
  G. CodeAct 命名空间注入 set_pair_superposition

用法：python checks/ai_pair_superposition_selfcheck.py
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.ai.sim_state import build_pair_sup_spec                 # noqa: E402
from src.core.evolve_util import (_as_pair_vector,               # noqa: E402
                                  pair_psi0_vector)

C = Checker('ai_pair_superposition')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError as e:
        return str(e)
    return None


# 合成基组（N=5——刻意避开 _as_vector 的"恰好 4 态"边角，见接口设计 §六）
_Q = [(60, 0, 0.5, +0.5), (60, 0, 0.5, -0.5), (60, 1, 1.5, +1.5),
      (60, 1, 1.5, +0.5), (60, 1, 1.5, -0.5)]


def fake_states():
    return [dict(n=n, l=l, j=j, m=m, kind='mJ', mj=m, mF=None,
                 E_mhz=0.0, label='L%d' % i)
            for i, (n, l, j, m) in enumerate(_Q)]


S = fake_states()
N = len(S)
K1 = 4 * N + 0        # |60P(m=-0.5)_A, 60S(m=+0.5)_B⟩
K2 = 0 * N + 4        # |60S(m=+0.5)_A, 60P(m=-0.5)_B⟩


def quad(i):
    return [int(_Q[i][0]), int(_Q[i][1]), float(_Q[i][2]), float(_Q[i][3])]


# ---------------------------------------------------------------
# A. build_pair_sup_spec（纯函数）
# ---------------------------------------------------------------
print('A) build_pair_sup_spec（纯函数：归一前范数 / 占据谱 / 非法输入）')
spec, info = build_pair_sup_spec([
    {'pair': [quad(4), quad(0)], 'amp': [1.0, 0.0]},
    {'pair': [quad(0), quad(4)], 'amp': [0.0, 1.0]}])
C.check('A1 spec 形态 = [pair_sup, [10 元行…]]',
        spec[0] == 'pair_sup' and len(spec[1]) == 2
        and all(len(r) == 10 for r in spec[1]), str(spec)[:80])
C.check('A2 归一前范数 = √2（不在此归一，归一交给引擎）',
        abs(info['norm_before'] - 2 ** 0.5) < 1e-12,
        str(info['norm_before']))
C.check('A3 占据谱 50/50',
        all(abs(p[8] - 0.5) < 1e-9 for p in info['populations']),
        str([p[8] for p in info['populations']]))
C.check('A4 分量顺序 = 输入的 A/B 语义（P_A,S_B 与 S_A,P_B 各一行）',
        spec[1][0][:8] == quad(4) + quad(0)
        and spec[1][1][:8] == quad(0) + quad(4),
        str(spec[1]))
# mag/phase 形态 ≡ amp 形态
s2, i2 = build_pair_sup_spec([{'pair': [quad(0), quad(4)], 'mag': 1.0,
                               'phase_deg': 90.0}])
C.check('A5 mag+phase_deg 与 amp 等价（90° → 0+1i）',
        abs(s2[1][0][8]) < 1e-12 and abs(s2[1][0][9] - 1.0) < 1e-12)
s3, i3 = build_pair_sup_spec([{'pair': [quad(0), quad(4)]}])
C.check('A6 缺 amp → 默认 1（同单体 build_sup_spec 语义）',
        abs(s3[1][0][8] - 1.0) < 1e-12 and abs(i3['norm_before'] - 1.0) < 1e-12)
C.check('A7 空分量 → 报错', _raises(build_pair_sup_spec, []) is not None)
C.check('A8 pair 不是 [[…],[…]] → 报错',
        _raises(build_pair_sup_spec,
                [{'pair': [quad(0)], 'amp': [1, 0]}]) is not None)
C.check('A9 pair 项不是 [n,l,j,m] → 报错',
        _raises(build_pair_sup_spec,
                [{'pair': [[60, 0, 0.5], quad(4)], 'amp': [1, 0]}]) is not None)
C.check('A10 幅度全零 → 报错',
        _raises(build_pair_sup_spec,
                [{'pair': [quad(0), quad(4)], 'amp': [0.0, 0.0]}])
        is not None)

# ---------------------------------------------------------------
# B. _as_pair_vector：索引映射 / 归一 / 相位 / 缺项
# ---------------------------------------------------------------
print('B) _as_pair_vector：索引映射 k=iA*N+iB、归一、相位、报错')
v = _as_pair_vector(S, ['pair_sup', [quad(4) + quad(0) + [1.0, 0.0]]])
C.check('B1 单分量 → k=iA*N+iB 处为 1（k=%d）' % K1,
        abs(v[K1] - 1.0) < 1e-15 and abs(np.linalg.norm(v) - 1.0) < 1e-15
        and abs(np.abs(v).sum() - 1.0) < 1e-15,
        'argmax=%d' % int(np.argmax(np.abs(v))))
v2 = _as_pair_vector(S, ['pair_sup', [quad(4) + quad(0) + [1.0, 0.0],
                                      quad(0) + quad(4) + [0.0, 1.0]]])
C.check('B2 纠缠叠加：两分量各 1/√2 且**相对相位 = i**',
        abs(abs(v2[K1]) - 2 ** -0.5) < 1e-15
        and abs(v2[K1] - 2 ** -0.5) < 1e-12
        and abs(v2[K2] - 1j * 2 ** -0.5) < 1e-12,
        'v[k1]=%s v[k2]=%s' % (v2[K1], v2[K2]))
C.check('B3 Σ|ψ|² = 1（自动归一）',
        abs(np.linalg.norm(v2) - 1.0) < 1e-15)
v3 = _as_pair_vector(S, ['pair_sup', [quad(4) + quad(0) + [3.0, 0.0],
                                      quad(0) + quad(4) + [4.0, 0.0]]])
C.check('B4 未归一幅度 (3,4) → 3/5、4/5',
        abs(abs(v3[K1]) - 0.6) < 1e-12 and abs(abs(v3[K2]) - 0.8) < 1e-12,
        '%.6f / %.6f' % (abs(v3[K1]), abs(v3[K2])))
msg = _raises(_as_pair_vector, S, ['pair_sup', [[60, 0, 0.5, 9.5]
                                               + quad(0) + [1.0, 0.0]]])
C.check('B5 分量不在基组 → 报错并点名缺失项',
        msg is not None and '9.5' in msg, str(msg))
C.check('B6 非 pair_sup 形态 → 报错',
        _raises(_as_pair_vector, S, ['sup', []]) is not None)
C.check('B7 空分量 → 报错（不是静默全零）',
        _raises(_as_pair_vector, S, ['pair_sup', []]) is not None)
C.check('B8 幅度全零 → 报错',
        _raises(_as_pair_vector, S, ['pair_sup',
                                     [quad(0) + quad(4) + [0.0, 0.0]]])
        is not None)

# ---------------------------------------------------------------
# C. pair_psi0_vector：与逐原子直积的等价性 / 优先级
# ---------------------------------------------------------------
print('C) pair_psi0_vector ≡ 逐原子直积（单分量时逐位）')
prod = pair_psi0_vector(S, 4, 0)
one = _as_pair_vector(S, ['pair_sup', [quad(4) + quad(0) + [1.0, 0.0]]])
C.check('C1 单分量 pair spec ≡ |4⟩⊗|0⟩（逐位）',
        float(np.max(np.abs(prod - one))) == 0.0,
        'max|Δ|=%.3g' % float(np.max(np.abs(prod - one))))
a = pair_psi0_vector(S, 0, 4)
b = pair_psi0_vector(S, 4, 0)
C.check('C2 直积对换 ≠ 同一向量（确认 k 索引方向没搞反）',
        K1 != K2 and int(np.argmax(np.abs(a))) != int(np.argmax(np.abs(b)))
        and float(np.linalg.norm(a - b)) > 1.0,
        'argmax %d vs %d' % (int(np.argmax(np.abs(a))),
                             int(np.argmax(np.abs(b)))))
pri = pair_psi0_vector(S, 0, 0, ['pair_sup', [quad(4) + quad(0) + [1.0, 0.0]]])
C.check('C3 psi0_pair 优先于 psi0_A/B（逐位 = 单分量 spec）',
        float(np.max(np.abs(pri - one))) == 0.0)
C.check('C4 两体初态缺参 → 报错（不静默取 0 号态）',
        _raises(pair_psi0_vector, S, 0, None) is not None)

# ---------------------------------------------------------------
# D. 引擎签名接线
# ---------------------------------------------------------------
print('D) 引擎签名：psi0_pair 贯穿三引擎 + 统一入口')
from src.core.evolve_engine import evolve_engine                 # noqa: E402
from src.core.evolve_mixed import evolve_mixed_pair              # noqa: E402
from src.core.hard import evolve_hard_pair                       # noqa: E402
from src.core.sambe import evolve_sambe_pair                     # noqa: E402

for fn, nm in ((evolve_engine, 'evolve_engine'),
               (evolve_mixed_pair, 'evolve_mixed_pair'),
               (evolve_hard_pair, 'evolve_hard_pair'),
               (evolve_sambe_pair, 'evolve_sambe_pair')):
    C.check('D1 %s 有 psi0_pair 入参' % nm,
            'psi0_pair' in inspect.signature(fn).parameters)
msg = _raises(evolve_engine, 'mixed', S, psi0_pair=one, t_grid=np.linspace(0, 1, 2),
              n_body=1)
C.check('D2 n_body=1 + psi0_pair → 明确报错（不许静默忽略）',
        msg is not None and 'n_body' in msg, str(msg))
# GUI 接线（源码级哨兵：worker/面板确实把 spec 传下去）
_ui = open(os.path.join(ROOT, 'src', 'ui', 'evolve_worker.py'),
           encoding='utf-8').read()
_dy = open(os.path.join(ROOT, 'src', 'ui', 'ui_dynamic.py'),
           encoding='utf-8').read()
_pm = open(os.path.join(ROOT, 'src', 'pm_setup.py'), encoding='utf-8').read()
C.check('D3 worker 三引擎分支都带 psi0_pair',
        _ui.count('psi0_pair') >= 4, '出现 %d 次' % _ui.count('psi0_pair'))
C.check('D4 两体运行路径把 spec 交给 worker',
        'psi0_pair=psi0_pair' in _dy and '_pair_psi0_spec' in _dy)
C.check('D5 dynamic.psi_pair 在册（pm_setup）',
        "'dynamic.psi_pair'" in _pm)

# ---------------------------------------------------------------
# E. 真引擎：解析 V_dd 的 20 ns 交换窗（合成基组，不需要 ARC）
# ---------------------------------------------------------------
print('E) 真引擎（mixed，合成基组 + 解析 V_dd 50 MHz）')
import scipy.sparse as sp                                        # noqa: E402

V0 = 50.0                                   # MHz：交换率 → 周期 20 ns
V = np.zeros((N * N, N * N))
V[K1, K2] = V[K2, K1] = V0
Vsp = sp.csr_matrix(V)
t = np.linspace(0.0, 10.0, 11)              # 半周期：应看到完全交换
spec_sym, _ = build_pair_sup_spec([          # (|k1⟩+|k2⟩)/√2 —— 交换**暗态**
    {'pair': [quad(4), quad(0)], 'amp': [1.0, 0.0]},
    {'pair': [quad(0), quad(4)], 'amp': [1.0, 0.0]}])
spec_ph, _ = build_pair_sup_spec([           # (|k1⟩+i|k2⟩)/√2 —— 有相对相位
    {'pair': [quad(4), quad(0)], 'amp': [1.0, 0.0]},
    {'pair': [quad(0), quad(4)], 'amp': [0.0, 1.0]}])
spec_one = ['pair_sup', [quad(4) + quad(0) + [1.0, 0.0]]]
kw = dict(fields_A=None, fields_B=None, t_grid=t, n_body=2, vdd_matrix=Vsp)
r_sym = evolve_engine('mixed', S, psi0_A=0, psi0_B=0, psi0_pair=spec_sym, **kw)
r_ph = evolve_engine('mixed', S, psi0_A=0, psi0_B=0, psi0_pair=spec_ph, **kw)
r_prod = evolve_engine('mixed', S, psi0_A=4, psi0_B=0, **kw)
r_one = evolve_engine('mixed', S, psi0_A=1, psi0_B=2, psi0_pair=spec_one, **kw)
ps = r_sym['pops_2d']
pp = r_ph['pops_2d']
C.check('E1 纠缠初态：两个分量都是 0.5（逐位）',
        abs(ps[0, K1] - 0.5) < 1e-12 and abs(ps[0, K2] - 0.5) < 1e-12,
        'P=%.12f/%.12f' % (ps[0, K1], ps[0, K2]))
C.check('E2 布居守恒（norm_dev < 1e-6）',
        max(r_sym['error_report']['max_norm_dev'],
            r_ph['error_report']['max_norm_dev']) < 1e-6,
        '%.2e / %.2e' % (r_sym['error_report']['max_norm_dev'],
                         r_ph['error_report']['max_norm_dev']))
C.check('E3 两体子空间封闭：P(k1)+P(k2) ≡ 1（对称与相位两例）',
        float(np.max(np.abs(ps[:, K1] + ps[:, K2] - 1.0))) < 1e-6
        and float(np.max(np.abs(pp[:, K1] + pp[:, K2] - 1.0))) < 1e-6)
# 对称组合 (|k1⟩+|k2⟩)/√2 是交换哈密顿量的**本征态（暗态）**：只吃全局相位、
# 布居不动 —— 这一条同时证明"相对相位真的进了引擎"（若 spec 被当纯态处理，
# 下一条的摆动就不可能同时成立）。
C.check('E4 对称态 = 交换暗态：P(k1) 恒 0.5（摆幅 < 1e-6 数值泄漏）',
        float(np.ptp(ps[:, K1])) < 1e-6,
        'ptp=%.2e' % float(np.ptp(ps[:, K1])))
C.check('E5 相对相位 π/2 → 确有交换（P(k1) 摆幅 > 0.1）',
        float(np.ptp(pp[:, K1])) > 0.1,
        'ptp=%.4f' % float(np.ptp(pp[:, K1])))
C.check('E6 纠缠初态轨迹 ≠ 直积初态轨迹（coherence 真的进了引擎）',
        float(np.max(np.abs(pp - r_prod['pops_2d']))) > 0.1,
        'max|Δ|=%.4f' % float(np.max(np.abs(pp - r_prod['pops_2d']))))
C.check('E7 单分量 spec ≡ 逐原子直积（逐位一致，且 psi0_pair 优先于 A/B=1,2）',
        float(np.max(np.abs(r_one['pops_2d'] - r_prod['pops_2d']))) == 0.0,
        'max|Δ|=%.3g' % float(np.max(np.abs(r_one['pops_2d']
                                            - r_prod['pops_2d']))))
C.check('E8 直积路径的初态是纯态（P(k1)=1）且它会交换',
        abs(r_prod['pops_2d'][0, K1] - 1.0) < 1e-12
        and float(np.ptp(r_prod['pops_2d'][:, K1])) > 0.1,
        'ptp=%.4f' % float(np.ptp(r_prod['pops_2d'][:, K1])))

# ---------------------------------------------------------------
# F. AI/GUI 面（offscreen 构窗；内存不足 → SKIP 不是 FAIL）
# ---------------------------------------------------------------
print('F) AI/GUI 面：dynamic.psi_pair 登记 + 往返（构窗失败则 SKIP）')
_gui = True
try:
    from PyQt5.QtWidgets import QApplication                     # noqa: E402
    import app_entry as M                                        # noqa: E402
    _app = QApplication.instance() or QApplication(sys.argv)
    try:
        win = M.MainWindow()
    except MemoryError as e:
        print('  SKIP 内存不足（软件可能正跑着）：%s' % e)
        _gui = False
except Exception as e:                                           # noqa: BLE001
    print('  SKIP 构窗不可用：%s' % e)
    _gui = False

if _gui:
    from src.ai.sim_state import SimState                        # noqa: E402
    st = SimState(win)
    C.check('F1 dynamic.psi_pair 在册（本进程已加载新代码）',
            'dynamic.psi_pair' in win.param_reg._bindings)
    st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])
    lv = st.list_levels()['levels']
    have = {(x['n'], x['l'], x['j'], x['m']) for x in lv}

    def lvl(n, l, j, m):
        return [n, l, j, m] if (n, l, j, m) in have else None

    pa, pb = lvl(60, 1, 1.5, -0.5), lvl(60, 0, 0.5, +0.5)
    C.check('F2 基组含 60P_3/2 m=-0.5 与 60S_1/2 m=+0.5（交换对）',
            pa is not None and pb is not None)
    if pa and pb:
        st.set_param('dynamic.mode', 0)
        r = st.set_pair_superposition([
            {'pair': [pa, pb], 'amp': [1.0, 0.0]},
            {'pair': [pb, pa], 'amp': [0.0, 1.0]}])
        C.check('F3 set_pair_superposition ok（含 mode=0 提示）',
                r.get('ok') and r.get('mode_warning'), str(r)[:120])
        spec_now = win.param_reg.read('dynamic.psi_pair')
        C.check('F4 写进 dynamic.psi_pair 且形态完整',
                isinstance(spec_now, list) and spec_now[0] == 'pair_sup'
                and len(spec_now[1]) == 2, str(spec_now)[:100])
        C.check('F5 面板 _pair_psi0_spec() 取到同一 spec（运行期通道）',
                win.dynamic._pair_psi0_spec() == spec_now)
        C.check('F6 get_config 暴露 psi_pair',
                st.get_config()['config'].get('psi_pair') == spec_now)
        # 逐原子初态要把两体 spec 清掉（防静默覆盖）
        r2 = st.set_initial_state('A', 60, 0, 0.5, 0.5)
        C.check('F7 set_initial_state 清除两体 spec（pair_spec_cleared）',
                r2.get('ok') and r2.get('pair_spec_cleared')
                and not win.param_reg.read('dynamic.psi_pair'), str(r2)[:120])
        # 快照 / 回填（v3：登记册即存档）
        win.param_reg.set('dynamic.psi_pair', spec_now, source='ai')
        snap = win.param_reg.collect()
        C.check('F8 快照含 dynamic.psi_pair',
                snap.get('dynamic.psi_pair') == spec_now)
        st.set_pair_superposition([])                    # 清空
        C.check('F9 传空列表 = 清除（回逐原子直积）',
                win.param_reg.read('dynamic.psi_pair') is None)
        win.param_reg.load({'dynamic.psi_pair': spec_now},
                           apply_external=True)
        C.check('F10 快照回填可恢复 spec（存档/undo 同路）',
                win.param_reg.read('dynamic.psi_pair') == spec_now)
        # 非法分量：报错 + 列出缺失项
        bad = st.set_pair_superposition([{'pair': [pa, [60, 0, 0.5, 9.5]],
                                          'amp': [1.0, 0.0]}])
        C.check('F11 分量不在基组 → ok=False 且列出缺失项',
                bad.get('ok') is False and bad.get('missing')
                and bad['missing'][0][3] == 9.5, str(bad)[:120])
        # 基组变更使 spec 失效 → psi_warnings 点名（A8 的教训）
        win.param_reg.set('dynamic.psi_pair', spec_now, source='ai')
        g = st.set_groups([(60, 0, 0.5), (60, 1, 1.5)])   # 去掉 6P（无关项）
        C.check('F12 组变更时 psi_warnings 汇报两体 spec 状态',
                isinstance(g.get('psi_warnings'), list))
        win.param_reg.set('dynamic.psi_pair', None, source='ai')

# ---------------------------------------------------------------
# G. CodeAct 命名空间
# ---------------------------------------------------------------
print('G) CodeAct 命名空间注入 set_pair_superposition')


class _S:
    def get_state(self, scope=None):
        return {}

    def summary(self):
        return {}

    def set_param(self, *a, **k):
        return {}

    def add_level_group(self, *a, **k):
        return {}

    def remove_level_group(self, *a, **k):
        return {}

    def set_waves(self, *a, **k):
        return {}

    def to_dict(self):
        return {}

    def set_pair_superposition(self, *a, **k):
        return {'ok': True}


from src.ai import codeact                                       # noqa: E402

ns = codeact._make_namespace(_S(), '.', {'seq': [0]}, lambda j: {},
                             lambda j: {}, lambda j, s=False: {})
C.check('G1 命名空间含 set_pair_superposition',
        'set_pair_superposition' in ns and ns['set_pair_superposition']()['ok'])

ok = C.summary()
sys.exit(0 if ok else 1)
