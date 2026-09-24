# -*- coding: utf-8 -*-
"""solver_inspector 自检（B-2）：god-mode 检视器脱离已删的 core.params/pm_setup，
改走 registry —— engine.* 只读展示、solve.* 可写（经 choke point 落 session）。

  A. set_god_mode(True) 打开检视器、6 tab 建起、param tab 从 registry 出参数（非空）。
  B. **点1 不变量**：engine.* 单元全只读、solve.* 单元可写。
  C. solve.* 编辑 → _on_param_edit → app_bridge.write → 会话真有值（人机同路径 I5）。

offscreen；os._exit 绕过 teardown 噪声。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')
os.environ.setdefault('RE0_V2_DATA_ROOT', tempfile.mkdtemp(prefix='guiinsp_'))

from PyQt5.QtWidgets import QApplication                  # noqa: E402
from checks.checklib import Checker                        # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
C = Checker('gui_inspector')

import app_entry                                          # noqa: E402

w = app_entry.MainWindow()
d = w.dynamic
_insp, _err = None, None
try:
    d.set_god_mode(True)                 # 打开检视器（建 6 tab，含 param tab）
    _insp = d._inspector
except Exception as e:                                    # noqa: BLE001
    import traceback
    _err = '%s: %s' % (type(e).__name__, e)
    traceback.print_exc()

_edits = getattr(_insp, '_e_edits', None) or {}
_recs = getattr(_insp, '_e_recs', None) or {}
_eng = [n for n in _edits if (_recs.get(n) or {}).get('family') == 'engine']
_solv = [n for n in _edits if (_recs.get(n) or {}).get('family') == 'solve']

C.group('A 检视器打开、param tab 从 registry 建（脱离 core.params/pm_setup）', [
    ('set_god_mode(True) 无异常（%s）' % (_err or 'ok'), _err is None and _insp is not None),
    ('param 单元非空（registry 出了参数）', len(_edits) > 0),
    ('含 engine.* 内部常量', len(_eng) > 0),
    ('含 solve.* 运行级参数', len(_solv) > 0),
])
C.group('B 点1 不变量：engine.* 只读 / solve.* 可写', [
    ('engine.* 单元全只读', len(_eng) > 0
     and all(_edits[n].isReadOnly() for n in _eng)),
    ('solve.* 单元可写', len(_solv) > 0
     and all(not _edits[n].isReadOnly() for n in _solv)),
])

# C：写一个 solve.*（MAX_STEP）→ 经 choke point 落 session
_k = 'solve.MAX_STEP'
_wrote = None
if _k in _edits:
    _edits[_k].setText('1e-6')
    try:
        _insp._on_param_edit(_k)
        _wrote = w.app_bridge.read(_k)
    except Exception as e:                                # noqa: BLE001
        _wrote = 'ERR:%s' % e
C.group('C solve.* 编辑经 choke point 落 session（人机同路径）', [
    ('solve.MAX_STEP 在参数页', _k in _edits),
    ('写后会话读到 1e-6（%r）' % (_wrote,), _wrote == 1e-6),
])

# ---------------- D. P0-7：4 个精度旋钮接上（此前是死控件） ----------------
# 背景：`_bind_solver_knobs()` 定义后**无调用点** ⇒ 处理链/求解页那 4 个 QLineEdit
# 是"可输入、不生效、不报错"的死控件（D5/D6）。现接上：只读 + 回显生效值 + 联动。
_knob_attrs = ('god_rcrit', 'god_perfreq_m', 'god_tol', 'god_maxstep')
_knobs = {a: getattr(_insp, a, None) for a in _knob_attrs}
_shown = {a: (e.text() if e is not None else None) for a, e in _knobs.items()}
_all_ro = all(e is not None and e.isReadOnly() for e in _knobs.values())
# 联动：写 solve.MAX_STEP → refresh_param_display → god_maxstep 跟着变（读回面）
try:
    w.app_bridge.write('solve.MAX_STEP', 1e-6, reason='判据：旋钮回显联动', actor='gui')
    _insp.refresh_param_display()
    _linked = _knobs['god_maxstep'].text() == '1e-06'
except Exception as e:                                    # noqa: BLE001
    _linked = 'ERR:%s' % e
C.group('D P0-7 精度旋钮已接上（只读回显 + 与 solve.* 联动）', [
    ('4 个旋钮都存在且只读（不再假装可改）', _all_ro),
    ('绑定计数 == 4（此前恒 0：函数无调用点）',
     getattr(_insp, '_solver_knobs_bound', 0) == 4),
    ('engine 常量回显 registry 默认（r_crit=%r / tol=%r）'
     % (_shown['god_rcrit'], _shown['god_tol']),
     _shown['god_rcrit'] == '0.0135' and _shown['god_tol'] == '1e-08'),
    ('写 solve.MAX_STEP 后旋钮跟着变（读回面 D6，%r）' % (_linked,),
     _linked is True),
])

# ---------------- E. 旧参数覆写管线残骸已删（反向断言，防回流） ----------------
# 这些名字属已删的"god-mode 参数覆写管线"（god_split/god_write/engine_overrides 同族）。
# 留着的后果：看起来有 API，其实无人调用（P0-7 的"死接线会骗人"）。
_DEAD_API = ('set_god_params', 'god_values', 'param_overrides_valid',
             'param_overrides', 'param_errors', 'reset_param_overrides',
             '_mirror_write', '_sync_mirrors', '_LEGACY_KW2PATH')
_left = [n for n in _DEAD_API if hasattr(_insp, n)]
C.group('E 旧覆写管线死 API 已删（防回流）', [
    ('%d 个死名全不存在（残留 %s）' % (len(_DEAD_API), _left or '无'),
     not _left),
    ('param_edited 信号已删（从不 emit/从不 connect）',
     not hasattr(_insp, 'param_edited')),
])

_ok = C.summary()
sys.stdout.flush()
os._exit(0 if _ok else 1)
