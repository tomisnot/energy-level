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
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')
# MainWindow 装配写临时数据根（单跑也成立，不污染仓内权威 journal）
os.environ.setdefault('RE0_V2_DATA_ROOT', tempfile.mkdtemp(prefix='guientry_'))

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

# ---------------- B. P1-7：「窗口 / 光子块」接线（此前静默失能） ----------------
# 用户裁决 2026-09-22：这两个值**只对静态引擎生效**、对动态引擎不应生效。
# 本组验四件事：① 值真的进了静态页（改旋钮→AC 位移变）；② 默认值与引擎默认**逐位一致**
# （零侵入）；③ registry 键的**入册/未入册两条分支都如实反映现状**（无需再改 GUI）；
# ④ 动态侧源码**不读**这两个键（"不生效"是结构性事实，不是靠约定）。
#
# ⚠ **本组对"键是否已入册"免疫**（2026-09-22 教训）：物理侧随时可能入册/改名，若断言
# "现在必须未入册"（或反之），别人一入册就把本判据弄红——那是**过期断言**，不是回归。
# 正确口径：**用一个人为不存在的键做反例**去验"未入册分支"，用**真实键**去验"入册分支"，
# 并断言 `registered` **等于**"registry 里到底有没有这个键"（如实反映，不预设）。
import os as _os                                            # noqa: E402
from src.core import core as _core                          # noqa: E402

_G = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
_f = abs(_w.cache.E_mhz(60, 1, 1.5) - _w.cache.E_mhz(60, 0, 0.5)) * 1e6
_CFG = {'enabled': True, 'intensity': 1.0e4, 'frequency_hz': _f,
        'polarization': 'π'}


def _max_ac(**kw):
    rows = _core.build_level_rows(_w.cache, 0.0, _G, _CFG, _G, **kw)
    vals = [abs(float(r.get('ac') or 0.0)) for r in rows]
    return max(vals) if vals else 0.0


def _in_registry(key):
    """该键当前是否真的在 registry 契约里（唯一事实来源，不预设）。"""
    try:
        _w.app_bridge.describe(key)
        return True
    except Exception:                                    # noqa: BLE001
        return False


_FIELD = type(_w.field)
_ac_default = _max_ac()                                  # 不传参 = 引擎默认
_ac_explicit = _max_ac(n_photon=4, sos_n_window=3)       # 控件默认值
_ac_loose = _max_ac(n_photon=1, sos_n_window=1)          # 改小 → 精度变
_kw0 = dict(_w.field.ac_kwargs())
_keys = tuple(k for _a, k in _FIELD.AC_PRECISION_KEYS)
_registered_now = bool(getattr(_w.field, '_ac_prec_registered', None))
_registry_says = all(_in_registry(k) for k in _keys)

# ③a 未入册分支：**人为构造一个不存在的键**（不依赖任何真实键的当前状态）
_saved = _FIELD.AC_PRECISION_KEYS
try:
    _FIELD.AC_PRECISION_KEYS = (('window_spin', 'field.__no_such_key__'),
                                ('n_photon_spin', 'field.__no_such_key2__'))
    _w.field._bind_ac_precision()
    _unreg_flag = bool(getattr(_w.field, '_ac_prec_registered', None))
    _unreg_tip = _w.field.window_spin.toolTip()
finally:
    _FIELD.AC_PRECISION_KEYS = _saved
    _w.field._bind_ac_precision()                        # 还原为真实键
    _w.field._ac_prec_connected = False                  # 允许下面重连一次

# ③b 入册分支：真实键（若物理侧已入册则必须走 registry；未入册则必须走本地）
_w.field.window_spin.setValue(2)
_win_key = _w.app_bridge.read('field.window')
_events = [e for e in _w.app_bridge.activity(0)
           if e.get('target') in _keys]
_kw1 = dict(_w.field.ac_kwargs())

# ④ 动态侧不读这两个键（源码级反向断言）
_rs = open(_os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), 'src', 'core', 'runspec.py'),
    encoding='utf-8').read()
C.group('B P1-7 AC 精度（窗口/光子块）接线：静态生效 + 动态不生效', [
    ('默认值 = 引擎默认（零侵入，max|ac| %.6g）' % _ac_default,
     _ac_default == _ac_explicit),
    ('改旋钮真的改静态结果（(1,1) → max|ac| %.6g ≠ (4,3) %.6g）'
     % (_ac_loose, _ac_explicit), _ac_loose != _ac_explicit),
    ('ac_kwargs 默认 = (n_photon=4, sos_n_window=3)：%s' % (_kw0,),
     _kw0 == {'n_photon': 4, 'sos_n_window': 3}),
    ('registered 如实反映 registry（标=%s / 实际=%s）'
     % (_registered_now, _registry_says), _registered_now == _registry_says),
    ('人为不存在的键 → 走"未入册"分支（registered=%s）且 tooltip 明说'
     % _unreg_flag,
     _unreg_flag is False and '尚未入册' in _unreg_tip),
    ('真实键（%s）→ 改控件经 choke point 落 session（field.window=%r）'
     % ('/'.join(_keys), _win_key),
     (_win_key == 2) if _registry_says else (_win_key is None)),
    ('总线事件 actor=gui（人机同路 I5，%d 条）' % len(_events),
     bool(_events) and all(e.get('actor') == 'gui' for e in _events)),
    ('ac_kwargs 跟随 registry（%s）' % (_kw1,),
     _kw1['sos_n_window'] == (2 if _registry_says else 3)),
    ('动态侧 runspec.py 不读 n_photon/sos_n_window（"不生效"是结构性事实）',
     'n_photon' not in _rs and 'sos_n_window' not in _rs),
])

# ---------------- C. 两模式收拢：看门人编排 + headless 权威（v1 服务回流防呆） ----------------
# 2026-09-24 两模式收拢：编排归看门人 launcher.py（弹窗选模式、进程交接），
# app_entry 只剩 --gui（纯本地，不起端点）/ --mcp（MCP 服务形态，写权 AI）。
# run_ai/驾驶舱/v1 serve 的 spawn 一旦出现 = v1 回流，本组判红。
_APP_SRC = open(_os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), 'app_entry.py'), encoding='utf-8').read()
_LAUNCH_SRC = open(_os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), 'launcher.py'), encoding='utf-8').read()
C.group('C 两模式收拢：看门人编排 + 权威 headless + GUI 不起端点', [
    ('launcher 按模式 spawn（--gui / --mcp）',
     '--gui' in _LAUNCH_SRC and '--mcp' in _LAUNCH_SRC),
    ('launcher 轮询 .mode-request 做进程交接（切换经看门人）',
     '.mode-request' in _LAUNCH_SRC),
    ('交接先 terminate 再等退出（写租约随进程释放，防 writer_lease_held）',
     '_terminate_group' in _LAUNCH_SRC and '_await_exit' in _LAUNCH_SRC),
    ('app_entry 有 run_mcp（MCP 服务形态，写权自动落 AI）',
     'def run_mcp' in _APP_SRC and "switch_mode('ai'" in _APP_SRC),
    ('GUI 模式装配不起端点（start_services=False，AI 物理看不见）',
     'make_v2_bridge(PROJ_ROOT, start_services=False)' in _APP_SRC),
    ('GUI 模式不再挂 AI 监控侧边栏（monitor_panel 已退役）',
     'monitor_panel' not in _APP_SRC),
    ('无 run_ai / cockpit.server / src.adapter.serve（v1 编排与驾驶舱不回流）',
     'def run_ai' not in _APP_SRC and 'cockpit.server' not in _APP_SRC
     and 'src.adapter.serve' not in _APP_SRC),
    ('PROJ_ROOT 指向仓根（含 src/ 与 save_state/）',
     _os.path.isdir(_os.path.join(app_entry.PROJ_ROOT, 'src'))
     and _os.path.isdir(_os.path.join(app_entry.PROJ_ROOT, 'save_state'))),
])

_ok = C.summary()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0 if _ok else 1)          # 绕过 offscreen teardown 噪声（见模块 docstring）
