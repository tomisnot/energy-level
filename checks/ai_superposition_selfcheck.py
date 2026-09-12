# -*- coding: utf-8 -*-
"""自检：**逐原子叠加态初态**是持久真值（#3/#5 裁决 A，2026-09-12）。

背景（两个叠在一起的缺陷，都实测过）：
  1. **引擎侧格式错配**：`SimState.build_sup_spec` 存 `['sup', [[n,l,j,m,re,im],…]]`，
     而 `evolve_util._as_vector` 的 sup 分支只认旧形态 `[[spec, amp],…]` —— 规范形态的
     `item[0]` 是主量子数 n（60）→ 被当**索引** → IndexError。"叠加态初态"因此
     从未在引擎里真正跑通。
  2. **GUI 侧通道丢失**：`dynamic.psi_*` 的 spec 只进 `_pending_psi_keys`，
     `find_state_index` 匹配不到 → 运行期读 `psi_combo.currentIndex()` → **静默回落 0 号态**。
裁决 A 要求：① 运行期读不到可用 spec **必须报错**（禁止静默回落 0 号态）；
② GUI 下拉必须有**显式回落显示**（"叠加态（自定义 n 项）"）。

覆盖：
  A. 引擎侧：规范形态 / 旧形态 / 简写 [n,l,j,m] 三种写法逐位等价；单分量 ≡ 索引；
     相对相位保留；非法写法报错；缺项报错
  B. GUI（offscreen；内存不足 SKIP）：写 spec → 登记项读回是 **spec 而非 combo 键**；
     下拉出现"叠加态（自定义…）"哨兵项；`_psi0_for` 在运行期给出 spec（不是索引）；
     人重选具体能级 → spec 被清（显式让位）；快照往返；基组失效 → psi_warnings
  C. 硬要求①：无 spec 且下拉无可用能级 → `_psi0_for` **抛错**（不静默取 0 或 -1）；
     失效 spec → 引擎侧 `_as_vector` 抛错并点名

用法：python checks/ai_superposition_selfcheck.py（GUI 段建议 QT_QPA_PLATFORM=offscreen）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402
from src.ai.sim_state import build_sup_spec                      # noqa: E402
from src.core.evolve_util import _as_vector                      # noqa: E402

C = Checker('ai_superposition')

_Q = [(60, 0, 0.5, +0.5), (60, 0, 0.5, -0.5), (60, 1, 1.5, +1.5),
      (60, 1, 1.5, +0.5), (60, 1, 1.5, -0.5)]
S = [dict(n=n, l=l, j=j, m=m, kind='mJ', mj=m, mF=None, E_mhz=0.0,
          label='L%d' % i) for i, (n, l, j, m) in enumerate(_Q)]
N = len(S)


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except (ValueError, IndexError) as e:
        return '%s: %s' % (type(e).__name__, e)
    return None


# ---------------------------------------------------------------
# A. 引擎侧：三种写法等价（这是修格式错配的核心）
# ---------------------------------------------------------------
print('A) 引擎侧 _as_vector：规范形态 / 旧形态 / 简写')
spec, info = build_sup_spec([{'state': [60, 0, 0.5, +0.5]},
                             {'state': [60, 1, 1.5, -0.5]}])
C.check('A0 build_sup_spec 形态 = [sup, [[n,l,j,m,re,im],…]]',
        spec[0] == 'sup' and len(spec[1]) == 2
        and all(len(r) == 6 for r in spec[1]), str(spec)[:70])
v_canon = _as_vector(S, spec)                      # 规范形态（修前 → IndexError）
v_old = _as_vector(S, ['sup', [[[60, 0, 0.5, 0.5], [1.0, 0.0]],
                               [[60, 1, 1.5, -0.5], [1.0, 0.0]]]])
v_short = _as_vector(S, ['sup', [[60, 0, 0.5, 0.5], [60, 1, 1.5, -0.5]]])
C.check('A1 规范形态可用（修前 IndexError）',
        float(np.linalg.norm(v_canon) - 1.0) < 1e-15)
C.check('A2 规范形态 ≡ 旧形态 [(spec, amp)]（逐位）',
        float(np.max(np.abs(v_canon - v_old))) == 0.0)
C.check('A3 规范形态 ≡ 简写 [n,l,j,m]（幅度默认 1，逐位）',
        float(np.max(np.abs(v_canon - v_short))) == 0.0)
v_one = _as_vector(S, ['sup', [[60, 1, 1.5, -0.5, 1.0, 0.0]]])
C.check('A4 单分量叠加 ≡ 该能级的索引初态（逐位）',
        float(np.max(np.abs(v_one - _as_vector(S, 4)))) == 0.0)
v_ph = _as_vector(S, ['sup', [[60, 0, 0.5, 0.5, 1.0, 0.0],
                              [60, 1, 1.5, -0.5, 0.0, 1.0]]])
C.check('A5 相对相位保留（v[4] = i/√2，4 = 60P_3/2 m=-0.5）',
        abs(v_ph[4] - 1j * 2 ** -0.5) < 1e-12, str(v_ph[4]))
C.check('A6 Σ|ψ|²=1（自动归一）',
        abs(np.linalg.norm(v_ph) - 1.0) < 1e-15)
msg = _raises(_as_vector, S, ['sup', [[60, 0, 0.5, 9.5, 1.0, 0.0]]])
C.check('A7 分量不在基组 → 报错并点名',
        msg is not None and '9.5' in msg, str(msg))
C.check('A8 非法分量写法 → 报错（不是静默跳过）',
        _raises(_as_vector, S, ['sup', [['x', 'y']]]) is not None)
C.check('A9 空分量 → 报错',
        _raises(_as_vector, S, ['sup', []]) is not None)

# ---------------------------------------------------------------
# B/C. GUI + AI 面（offscreen 构窗；内存不足 SKIP）
# ---------------------------------------------------------------
print('B) GUI/AI 面：持久真值 + 显式回落显示 + 硬要求①')
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
    st.set_groups([(60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)])
    lv = st.list_levels()['levels']
    have = {(x['n'], x['l'], x['j'], x['m']) for x in lv}
    pa = [60, 1, 1.5, -0.5] if (60, 1, 1.5, -0.5) in have else None
    pb = [60, 0, 0.5, +0.5] if (60, 0, 0.5, +0.5) in have else None
    C.check('B0 基组含 60P_3/2 m=-0.5 与 60S_1/2 m=+0.5',
            pa is not None and pb is not None)
    if pa and pb:
        r = st.set_superposition('A', [
            {'state': pa, 'amp': [1.0, 0.0]},
            {'state': pb, 'amp': [0.0, 1.0]}])
        C.check('B1 set_superposition ok', r.get('ok'), str(r)[:110])
        stored = win.param_reg.read('dynamic.psi_A')
        C.check('B2 登记项读回 = **spec**（不是 combo 键）',
                isinstance(stored, list) and stored[0] == 'sup'
                and len(stored[1]) == 2, str(stored)[:80])
        txts = [win.dynamic.psi_combo_A.itemText(i)
                for i in range(win.dynamic.psi_combo_A.count())]
        cur = win.dynamic.psi_combo_A.currentText()
        C.check('B3 下拉**显式显示**"叠加态（自定义…）"（硬要求②）',
                '叠加态' in cur and cur in txts, cur)
        C.check('B4 运行期 _psi0_for 给的是 spec（不是索引）',
                win.dynamic._psi0_for('A') == stored)
        # 快照往返（存档/undo 同路）
        snap = win.param_reg.collect()
        C.check('B5 快照含 spec', snap.get('dynamic.psi_A') == stored)
        st.set_initial_state('A', 60, 0, 0.5, 0.5)               # 人/AI 重选能级
        C.check('B6 设具体能级 → spec 让位（读回变成身份键）',
                not (isinstance(win.param_reg.read('dynamic.psi_A'), list)
                     and win.param_reg.read('dynamic.psi_A')[0] == 'sup'))
        win.param_reg.load({'dynamic.psi_A': stored}, apply_external=True)
        C.check('B7 快照回填可恢复 spec', win.param_reg.read('dynamic.psi_A')
                == stored)
        C.check('B8 回填后下拉又是哨兵项（显示与真值一致）',
                '叠加态' in win.dynamic.psi_combo_A.currentText())
        # 硬要求①：无 spec 且下拉选不到基组内能级 → 抛错（禁止静默 0/-1）
        win.param_reg.set('dynamic.psi_A', None, source='ai')
        _saved = getattr(win.dynamic, '_states', None)
        win.dynamic._states = []
        msg2 = _raises(win.dynamic._psi0_for, 'A')
        C.check('B9 硬要求①：无 spec 且基组空 → _psi0_for 抛错（不静默取 0/-1）',
                msg2 is not None, str(msg2)[:90])
        win.dynamic._states = _saved
        # 硬要求①续：失效 spec（分量不在基组）→ 引擎侧报错点名声
        win.param_reg.set('dynamic.psi_A',
                          ['sup', [[60, 1, 1.5, 9.5, 1.0, 0.0]]], source='ai')
        msg3 = _raises(_as_vector, win.dynamic._states,
                       win.dynamic._psi0_for('A'))
        C.check('B10 失效 spec → 运行路径抛错并点名（不静默从别的态起跑）',
                msg3 is not None and '9.5' in msg3, str(msg3)[:90])
        # 基组变更使叠加态失效 → psi_warnings 点名（A8 的教训）
        g = st.set_groups([(60, 0, 0.5)])
        C.check('B11 组变更 → psi_warnings 点名叠加态失效',
                any('叠加态' in w for w in (g.get('psi_warnings') or [])),
                str(g.get('psi_warnings'))[:120])
        win.param_reg.set('dynamic.psi_A', None, source='ai')

ok = C.summary()
sys.exit(0 if ok else 1)
