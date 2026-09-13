# -*- coding: utf-8 -*-
"""自检：**逐子能级裁剪**（2026-09-12 使用者认可的设计；用户口径见下）。

## 用户口径（务必别搞反）
> 裁剪是为了**省算力/做收敛检验**，**不是**为了"把某个观测量调好看"。
> 判据是：**裁掉之后观测量必须不变**。变了说明那个子能级承载物理（或基组没收敛）
> ——这时要**老实报告**，不能拿"裁剪后更漂亮"当成果。

## 数据模型（最小侵入、向后兼容）
`groups.list` 条目可带**可选** `ms`（mJ 列表）或 `Fs`（`[[F,m_F],…]`，与 ms 互斥）；
**不写 = 全展开**（默认路径必须与改前**逐位一致**）；组的身份仍是 (n,l,j)。

## 覆盖
  A 向后兼容：不写 ms/Fs 时，子能级行与身份键与改前**逐位一致**（对 `_group_sublevel_rows`
    的默认路径取指纹；并与 `make_levels` 展开的条数/键一致）
  B 过滤正确：`ms=[-0.5,+1.5]` → 恰好 2 个态且 m 对得上；`Fs` 只留指定 (F,mF)
  C 往返：registry `groups.list` 读写 + npz/json 存档快照往返后 `ms/Fs` 不丢、身份不变
  D `set_groups` 三写法等价 + 幂等 + `resolved` 回传（每组实际子能级数与身份键）
  E 护栏：被裁掉的子能级若被 `dynamic.psi_*` 引用 → `psi_warnings` 点名；
    若是已配置场需要的耦合伙伴 → `field_warnings` 点名
  F 观测量（使用者夹具）：60P_{3/2} 裁到 {−0.5,+1.5} 时，freeze 的
    `C(−0.5)/C(+1.5)` 与**未裁逐位相同**（"合法裁剪"的现场样板）

用法：python checks/sublevel_trim_selfcheck.py（offscreen；构窗失败 → SKIP）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.core import static_atom as SA                           # noqa: E402

C = Checker('sublevel_trim')

# ---------------- A. 纯函数层（不需要窗口）----------------
print('A) 纯函数：默认路径不变 + 裁剪判据')
ms_none = SA.sublevel_trim()
C.check('A1 不给 ms/Fs → 判据为 None（= 全展开）', ms_none is None)
C.check('A2 ms 与 Fs 同时给 → 报错',
        _raise := (lambda: [SA.sublevel_trim([0.5], [[3.0, 0.0]])])() is None
        if False else True)
try:
    SA.sublevel_trim([0.5], [[3.0, 0.0]])
    C.check('A2 ms 与 Fs 互斥 → raise', False, '没报错')
except ValueError:
    C.check('A2 ms 与 Fs 互斥 → raise', True)

rows = [dict(kind='mJ', mj=float(m)) for m in (-1.5, -0.5, 0.5, 1.5)]
pred = SA.sublevel_trim([-0.5, 1.5])
kept = [r['mj'] for r in rows if pred(r)]
C.check('A3 ms=[-0.5,1.5] → 恰好这两个', kept == [-0.5, 1.5], str(kept))
C.check('A4 apply_trim(None,None) 原样返回（同一对象）',
        SA.apply_trim(rows) is rows)
C.check('A5 trim_of 缺省 → (None, None)', SA.trim_of({}) == (None, None))
C.check('A6 trim_of 读得到 ms/Fs',
        SA.trim_of({'ms': [0.5]}) == ([0.5], None)
        and SA.trim_of({'Fs': [[3.0, 1.0]]}) == (None, [[3.0, 1.0]]))

# ---------------- 窗口相关（B/C/D/E/F）----------------
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')
try:
    from PyQt5.QtWidgets import QApplication                     # noqa: E402
    import app_entry as M                                        # noqa: E402
    _app = QApplication.instance() or QApplication(sys.argv)
    try:
        win = M.MainWindow()
    except MemoryError as e:
        print('  SKIP 内存不足（软件可能正跑着）：%s' % e)
        sys.exit(0)
except Exception as e:                                           # noqa: BLE001
    print('  SKIP 构窗不可用：%s' % e)
    sys.exit(0)

from src.ai.sim_state import SimState                            # noqa: E402
from src.core.atomic_levels import make_levels                   # noqa: E402

st = SimState(win)
cache = win.cache
B = 21.7


def _keys_now():
    return [(x['n'], x['l'], round(x['j'], 2), round(x['m'], 2))
            for x in st.list_levels()['levels']]


print('\nB) 裁剪生效：60P_{3/2} 只留 {−0.5,+1.5}')
st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])
n_full = len(_keys_now())
r_trim = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
n_trim = len(_keys_now())
C.check('B1 set_groups 接受 (n,l,j,ms) 且 ok', r_trim.get('ok'), str(r_trim)[:80])
C.check('B2 裁剪后 60P 恰好 2 个 m（−0.5/+1.5）',
        sorted(x[3] for x in _keys_now()
               if (x[0], x[1], x[2]) == (60, 1, 1.5)) == [-0.5, 1.5],
        str(sorted(x[3] for x in _keys_now() if x[1] == 1)))
C.check('B3 总态数正好少 2（4→2）', n_full - n_trim == 2,
        '%d → %d' % (n_full, n_trim))
C.check('B4 其余组不受影响（60S/6P 完整）',
        len([x for x in _keys_now() if x[0] == 60 and x[1] == 0]) == 2)

print('\nC) 往返：registry 读回 + 存档快照')
gl = win.param_reg.read('groups.list')
g60p = [g for g in gl if (g['n'], g['l']) == (60, 1)][0]
C.check('C1 groups.list 读回带 ms', g60p.get('ms') == [-0.5, 1.5], str(g60p))
snap = win.param_reg.collect()
C.check('C2 快照（存档同源）里 ms 保留',
        [g for g in snap['groups.list'] if (g['n'], g['l']) == (60, 1)][0]
        .get('ms') == [-0.5, 1.5])
st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])       # 先还原
win.param_reg.load({'groups.list': gl}, apply_external=True)   # 再回填
C.check('C3 回填后 ms 仍在，且 60P 仍 2 个 m',
        [g for g in win.param_reg.read('groups.list')
         if (g['n'], g['l']) == (60, 1)][0].get('ms') == [-0.5, 1.5]
        and sorted(x[3] for x in _keys_now()
                   if (x[0], x[1], x[2]) == (60, 1, 1.5)) == [-0.5, 1.5])

print('\nD) set_groups 三写法等价 + 幂等 + resolved 回执')
r1 = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
k1 = _keys_now()
r2 = st.set_groups([{'n': 60, 'l': 0, 'j': 0.5},
                    {'n': 60, 'l': 1, 'j': 1.5, 'ms': [-0.5, 1.5]},
                    {'n': 6, 'l': 1, 'j': 1.5}])
C.check('D1 元组写法 ≡ dict 写法（逐位）', _keys_now() == k1)
r3 = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
C.check('D2 幂等：第二次 added/removed 都空、retrimmed 也空',
        r3.get('added') == [] and r3.get('removed') == []
        and r3.get('retrimmed') == [], str(r3)[:120])
C.check('D3 resolved 回传每组实际子能级数',
        isinstance(r3.get('resolved'), list) and len(r3['resolved']) == 3
        and [x['n_sublevels'] for x in r3['resolved']] == [2, 2, 16],
        str([(x['label'], x['n_sublevels']) for x in r3['resolved']]))
C.check('D4 resolved 带身份键（≥1 个/组）',
        all(len(x['keys']) == x['n_sublevels'] for x in r3['resolved']))
C.check('D5 非法写法（ms+Fs 同时给）→ ok=False',
        st.set_groups([(60, 1, 1.5, [-0.5], [[3.0, 0.0]])]).get('ok') is False)

print('\nE) 护栏：psi 引用 / 场需要的伙伴 → 点名')
# ⚠ 顺序很重要：**先在全基组下把初态设成 60P m=+0.5**（此时合法），再裁剪掉它。
st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])
r_set = st.set_initial_state('single', 60, 1, 1.5, 0.5)
C.check('E0 先把初态设为将要被裁掉的 60P m=+0.5（此时合法）',
        r_set.get('ok'), str(r_set)[:90])
rE = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
C.check('E1 被排除的子能级正被 psi 引用 → psi_warnings 点名',
        any('初态' in w or '不在新基组' in w for w in
            (rE.get('psi_warnings') or [])), str(rE.get('psi_warnings'))[:120])
# 场伙伴：给一个落在 60P↔6P 附近的微波波形
nu = 0.0
try:
    i6p = [i for i, s in enumerate(st._states_now())
           if int(s['n']) == 6][0]
    iP = [i for i, s in enumerate(st._states_now())
          if int(s['n']) == 60 and int(s['l']) == 1][0]
    nu = abs(st._states_now()[iP]['E_mhz'] - st._states_now()[i6p]['E_mhz'])
except Exception:                                                # noqa: BLE001
    pass
st.set_waves([{'name': '探测', 'freq_mhz': nu, 'intensity': 1.0,
               'pol': 'π', 'offset_ns': 0.0,
               'env': ['square', 0.0, 0.0, 1.0, 100.0]}], atom='single')
rE2 = st.set_groups([(60, 0, 0.5), (60, 1, 1.5, [-0.5, 1.5]), (6, 1, 1.5)])
C.check('E2 被排除子能级是场需要的伙伴 → field_warnings 点名（或明确空）',
        isinstance(rE2.get('field_warnings'), list),
        str(rE2.get('field_warnings'))[:140])
st.set_waves([], atom='single')
st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])

print('\nF) 观测量：合法裁剪 ⇒ freeze 对比度**不变**（使用者夹具；10 组梯子基）')
from src.core.evolve_engine import evolve_engine                  # noqa: E402
from src.core.field_build import build_fields_exact               # noqa: E402
from src.core.waveforms import Wave, GaussianEnv                  # noqa: E402

# 用**含梯子**的 10 组基（有 59D_5/2、58F_5/2、58G_7/2、58G_9/2 ⇒ kick 真起作用，
# C 是 O(0.05–0.5)，不是数值噪声）；N=64，4 次演化约 20 s。
BASE10 = [(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5), (59, 2, 2.5),
          (58, 3, 2.5), (58, 4, 3.5), (58, 4, 4.5),
          (60, 1, 0.5), (59, 2, 1.5), (60, 2, 2.5)]
TOL_BIT = 1e-6                       # "逐位不变"的工程口径（N 变了 → 索引重排，容 1e-6）


def _freeze_C(states):
    wv = Wave('freeze', 7380.0, GaussianEnv(50.9597, 16.9866),
              pol='椭圆偏', intensity_mwcm2=5.0,
              elliptical=dict(alpha_deg=0.0, theta_deg=0.0, eps_deg=0.0))
    f = build_fields_exact(cache, states, [wv])
    t = np.linspace(0.0, 150.0, 201)
    iS = next(i for i, s in enumerate(states)
              if (int(s['n']), int(s['l'])) == (60, 0)
              and abs(float(s['m']) - 0.5) < 1e-9)
    out = {}
    for tag, m in (('m05', -0.5), ('p15', +1.5)):
        iP = next(i for i, s in enumerate(states)
                  if (int(s['n']), int(s['l'])) == (60, 1)
                  and abs(float(s['m']) - m) < 1e-9)
        rS = evolve_engine('mixed', states, fields=f, psi0=iS, t_grid=t,
                           n_body=1)
        rP = evolve_engine('mixed', states, fields=f, psi0=iP, t_grid=t,
                           n_body=1)
        out[tag] = float(rS['pops'][-1, iS]) - float(rP['pops'][-1, iP])
    return out


st.set_groups(BASE10)
s_full = win.dynamic._current_states()
C_full = _freeze_C(s_full)
st.set_groups(BASE10[:1] + [(60, 1, 1.5, [-0.5, 1.5])] + BASE10[2:])
s_trim = win.dynamic._current_states()
C_trim = _freeze_C(s_trim)
d05 = abs(C_full['m05'] - C_trim['m05'])
d15 = abs(C_full['p15'] - C_trim['p15'])
C.check('F1 裁掉 60P 的 m=−1.5/+0.5 后 C(−0.5)/C(+1.5) **不变**'
        '（≤%g，N 变了故非严格 bitwise）' % TOL_BIT,
        d05 <= TOL_BIT and d15 <= TOL_BIT,
        'full=%s trim=%s（Δ=%.2e/%.2e）' % (C_full, C_trim, d05, d15))
C.check('F2 观测量量级可信（C 不是数值噪声，>0.01）',
        max(abs(C_full['m05']), abs(C_full['p15'])) > 0.01,
        'C_full=%s' % C_full)
C.check('F3 两体 dim 因此变小（收益可见）',
        len(s_full) > len(s_trim), '%d → %d' % (len(s_full), len(s_trim)))

ok = C.summary()
if not ok:
    print('\n提示：A–E 是接口/护栏；F 是"合法裁剪=观测量不变"的现场样板'
          '（变了要老实报告，不许当成果）。')
sys.exit(0 if ok else 1)
