# -*- coding: utf-8 -*-
"""ui_dynamic run 流端到端自检（B-1③）：GUI"运行演化"经 ops 唯一分派跑通。

验新链路：面板配置 → `_sync_run_config`（非标准件 write_many 入 session）→ `ops.prepare`
（主线程 build_run，含 V_dd/寿命）→ `RunWorker`（工作线程 ops.execute）→ 结果落 artifact
→ `load_artifact` 取回数组（人机同产物 I5/I3）。这是 GUI 跑物理的核心功能守卫。

用真 MainWindow（offscreen）+ 最小单体 mixed 配置（60S/60P、共振方波、10ns）。
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('EL_NO_AI_SERVER', '1')
os.environ.setdefault('RE0_V2_DATA_ROOT', tempfile.mkdtemp(prefix='guiflow_'))

from PyQt5.QtWidgets import QApplication                  # noqa: E402
from checks.checklib import Checker                        # noqa: E402
from src.core import waveforms as WF                       # noqa: E402
from src.core.atomic_levels import state_key               # noqa: E402
from src.v2_bind import load_artifact                  # noqa: E402
from src.ui.evolve_worker import RunWorker                 # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
C = Checker('gui_run_flow')

import app_entry                                          # noqa: E402

w = app_entry.MainWindow()
d = w.dynamic
ops = w.app_bridge.ops

# ---- 配置动态面板（最小单体 mixed）----
d.main_groups = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
states = d._current_states()                 # 主线程 build_states（碰 cache）
_freq = abs(w.cache.E_mhz(60, 1, 1.5) - w.cache.E_mhz(60, 0, 0.5))
_wave = WF.Wave('opt', freq_mhz=_freq, envelope=WF.SquareEnv(0.0, 1.0, 1.0, 8.0),
                pol='π', intensity_mwcm2=1.0)
d.wave_panel.waves = [_wave]
d.method_cb.setCurrentText('mixed')
d.tmax_sb.setValue(10.0)
d.npts_sb.setValue(11)
d.decay_mode_cb.setCurrentIndex(1)           # 关（最简）
d._psi_sups = {'single': None, 'A': None, 'B': None}
d._states = None                             # 使 _psi_key_for→None，psi 走预置
# 预置 psi_single（身份键）——_sync_run_config 的 psi 为 None 会被过滤，不覆盖它
w.app_bridge.write('dynamic.psi_single', list(state_key(states[0])),
                   reason='判据预置初态', actor='gui')

# ---- A：_sync_run_config 把非标准件写进 session ----
d._sync_run_config()
snap = w.app_bridge.effective()
C.group('A _sync_run_config：GUI 配置 → session（write_many）', [
    ('groups.list 同步（2 组）', len(snap.get('groups.list') or []) == 2),
    ('dynamic.mode=0（单体）', snap.get('dynamic.mode') == 0),
    ('dynamic.method=mixed', snap.get('dynamic.method') == 'mixed'),
    ('dynamic.t_max/npts 同步', snap.get('dynamic.t_max') == 10.0
     and snap.get('dynamic.npts') == 11),
    ('dynamic.waves_single 同步（1 波形）',
     len(snap.get('dynamic.waves_single') or []) == 1),
    ('dynamic.psi_single 保留（预置未被 None 覆盖）',
     snap.get('dynamic.psi_single') is not None),
])

# ---- B：ops.prepare（主线程 build_run）→ RunWorker（工作线程 execute）→ artifact ----
prepared = ops.prepare(actor='gui', reason='判据运行')
_got = []
prog = np.zeros(3)
wk = RunWorker(ops, prepared, prog, actor='gui', reason='判据运行')
wk.done.connect(lambda res, err: _got.append((res, err)))
wk.run()                                     # 同步跑（线程安全已由 ops_threadsplit 守）
res, err = (_got[0] if _got else (None, 'no signal'))
aid = (res or {}).get('artifact_id')
loaded = load_artifact(ops.root, aid) if aid else None
arrays = loaded[1] if loaded else {}
C.group('B prepare(主)→RunWorker execute(工作)→artifact 取数组（人机同产物）', [
    ('prepare 建成 run（含 states）', 'run' in prepared
     and len(prepared['run'].get('states') or []) > 0),
    ('RunWorker 无错误（%s）' % (err or '无'), err is None),
    ('结果 ok', isinstance(res, dict) and res.get('ok') is True),
    ('artifact 可取回 pops 数组', 'pops' in arrays),
])

# ---- C：_on_evolve_done 从 artifact 绘图不炸（含 _run_states 分流）----
d._run_states = states
d.wave_data = WF.waveforms_to_env([_wave], np.linspace(0, 10.0, 11))
_plot_err = None
try:
    d._on_evolve_done(res, None)
except Exception as e:                                    # noqa: BLE001
    _plot_err = '%s: %s' % (type(e).__name__, e)
C.group('C _on_evolve_done：artifact 数组 → 绘图（单体分流 _draw_bot）', [
    ('回调无异常（%s）' % (_plot_err or 'ok'), _plot_err is None),
    ('_last_res 落地（含 pops）', isinstance(getattr(d, '_last_res', None), dict)
     and 'pops' in d._last_res),
])

# ---- D：P0-7 死接线已清（反向断言，防回流）----
# ① `_ai_initiated`：全库从未置位 ⇒ "AI 不弹模态"是**假保护**（恒假分支）。已删；
#    AI 路径不弹模态现在是**结构性**保证（AI 走 MCP run_compute，不进 _run()）。
# ② `_show_progress`/`_dlg`/`_dlg_timer`：进度弹窗 2026-09-14 已废，空方法与
#    "撤下进度框"分支一直留着（恒 None）。已删。
import inspect as _inspect                                    # noqa: E402
_DP = type(d)
_src_cost = _inspect.getsource(_DP._preflight_cost)
_src_run = _inspect.getsource(_DP._run)
_src_fin = _inspect.getsource(_DP._finish_run_ui)
_src_cancel = _inspect.getsource(_DP._cancel_run)
C.group('D P0-7 死接线已清（防回流；死接线会骗人）', [
    ("_ai_initiated 读取代码已删（仅在注释里留说明）",
     "getattr(self, '_ai_initiated'" not in _src_cost),
    ('_show_progress 空方法已删',
     not hasattr(_DP, '_show_progress')),
    ('_dlg / _dlg_timer 已从 _run/_finish_run_ui/_cancel_run 清除',
     not any('_dlg' in s for s in (_src_run, _src_fin, _src_cancel))),
    ('实例上不再有 _dlg 属性（此前恒 None 的死分支）',
     not hasattr(d, '_dlg')),
    ('进度轮询定时器仍在 _run 里接好（常驻底栏进度未被误删）',
     'self._timer' in _src_run and '_poll_progress' in _src_run),
])

_ok = C.summary()
sys.stdout.flush()
os._exit(0 if _ok else 1)          # 绕过 offscreen teardown 噪声
