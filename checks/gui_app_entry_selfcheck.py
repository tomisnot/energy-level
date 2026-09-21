# -*- coding: utf-8 -*-
"""app_entry MainWindow offscreen 实例化自检（B-4）——GUI 在新架构下总装起来的端到端守卫。

《回归主仓库》切换判据的核心一条：GUI 全量、可装配。验 MainWindow() 能 offscreen 实例化
——AppBridge 注入 engine + 建 cache/session、Binder 绑面板控件、各 panel（能级组/场/动态/
频谱/扫描）装配齐活，且关键部件在位（app_bridge/binder/cache/dynamic/field）。

已彻底脱离旧框架：无 pm_setup/param_reg/archive/src.ai（app_entry 迁移见 B-4）。

收尾用 os._exit：offscreen Qt + matplotlib 的解释器 teardown 会刷 QFontDatabase 警告并非零
退出（与实例化成败无关），故断言后直接 os._exit 绕过噪声 teardown。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')

from PyQt5.QtWidgets import QApplication                  # noqa: E402
from checks.checklib import Checker                        # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
C = Checker('gui_app_entry')

import app_entry                                          # noqa: E402

_w = None
_err = None
try:
    _w = app_entry.MainWindow()
except Exception as e:                                    # noqa: BLE001
    import traceback
    _err = '%s: %s' % (type(e).__name__, e)
    traceback.print_exc()

C.group('MainWindow offscreen 实例化（新架构总装，脱离 pm_setup/archive/src.ai）', [
    ('实例化无异常（%s）' % (_err or 'ok'), _err is None and _w is not None),
    ('有 app_bridge（mecha 门面：Bus/Authority/Session/Ops/Jobs）',
     _w is not None and hasattr(_w, 'app_bridge')),
    ('有 binder（面板控件 ↔ session）', _w is not None and hasattr(_w, 'binder')),
    ('cache 由注入引擎建（非 None）', _w is not None and getattr(_w, 'cache', None) is not None),
    ('dynamic + field 面板装配齐', _w is not None
     and hasattr(_w, 'dynamic') and hasattr(_w, 'field')),
    ('session 可读（effective 是 dict）', _w is not None
     and isinstance(_w.app_bridge.effective(), dict)),
])

_ok = C.summary()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0 if _ok else 1)          # 绕过 offscreen teardown 噪声（见模块 docstring）
