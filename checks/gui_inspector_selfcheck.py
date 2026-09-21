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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')

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

_ok = C.summary()
sys.stdout.flush()
os._exit(0 if _ok else 1)
