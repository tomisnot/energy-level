# -*- coding: utf-8 -*-
"""ops prepare/execute 线程拆分自检 —— 守 B-1① 定论。

背景：ARC 的 `sqlite3.connect` 无 `check_same_thread=False` → 连接**线程亲和**；两体 V_dd 大量
调 `cache.dme → getDipoleMatrixElement → 碰 SQLite`。若 build_run 在工作线程跑（而 cache 建于
主线程）→ `ProgrammingError: SQLite objects created in a thread...`。故 `ops.run` 拆成
`prepare`（主线程：gate+snapshot+build_run+check_run）+ `execute`（线程安全：evolve_run 纯数值）。

  A. prepare（主线程 build）+ execute（工作线程 evolve）跑通、结果 ok、落 run 事件。
  B. **不变量**：execute 全程**不碰 cache**（spy 计数 worker 侧 cache 调用 == 0）——这才是它
     能安全进工作线程的根据。**致死性对照**：prepare 侧 cache 调用 ≥1（证明 spy 真在数，
     B 的 ==0 不是 spy 失灵造成的假绿）。
  C. run() = prepare+execute 合成，行为不变（ok + summary + artifact_id）。

真跑最小单体演化（mixed，60S/60P，10ns）——秒级。用临时 root。
"""
import os
import sys
import tempfile
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from arc import Rubidium87                                 # noqa: E402
from checks.checklib import Checker                        # noqa: E402
from src.core.static_atom import ArcCache                  # noqa: E402
from src.core import runspec as RS                         # noqa: E402
from src.core import waveforms as WF                       # noqa: E402
from src.core.atomic_levels import state_key               # noqa: E402
from src.mecha import runtime                              # noqa: E402
from src.adapter import EngineAdapter                      # noqa: E402
from src.mecha.session import Session                      # noqa: E402
from src.mecha.ops import Ops                              # noqa: E402

runtime.set_engine(EngineAdapter())
C = Checker('ops_threadsplit')

_TMP = tempfile.mkdtemp(prefix='opsplit_')
_cache = ArcCache(Rubidium87())
_GROUPS = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
_st = RS.build_states(_cache, _GROUPS, 0.0, 0.0)
_freq = abs(_cache.E_mhz(60, 1, 1.5) - _cache.E_mhz(60, 0, 0.5))
_wave = WF.Wave('opt', freq_mhz=_freq, envelope=WF.SquareEnv(0.0, 1.0, 1.0, 8.0),
                pol='π', intensity_mwcm2=1.0)
_CFG = {'groups.list': _GROUPS, 'field.B_gauss': 0.0, 'field.E_vpcm': 0.0,
        'dynamic.mode': 0, 'dynamic.method': 'mixed', 'dynamic.t_max': 10.0,
        'dynamic.npts': 11, 'dynamic.decay_index': 1,
        'dynamic.psi_single': state_key(_st[0]),
        'dynamic.waves_single': WF.serialize_waves([_wave])}

s = Session(root=_TMP)
for _k, _v in _CFG.items():
    s.set(_k, _v, actor='setup', reason='判据初始化')
ops = Ops(s)

# ---- spy：按线程统计 session.cache() 调用（只本判据，实例级 monkeypatch）----
_main = threading.current_thread()
_calls = {'main': 0, 'worker': 0}
_orig_cache = s.cache


def _spy():
    _calls['main' if threading.current_thread() is _main else 'worker'] += 1
    return _orig_cache()


s.cache = _spy

# ---- A/B：prepare 主线程、execute 工作线程 ----
prepared = ops.prepare(actor='gui', reason='主线程 build')
_main_cache_calls = _calls['main']
_holder = {}


def _work():
    try:
        _holder['res'] = ops.execute(prepared, actor='gui', reason='工作线程 evolve',
                                     progress_arr=np.zeros(3))
    except Exception as e:                                # noqa: BLE001
        _holder['err'] = '%s: %s' % (type(e).__name__, e)


_t = threading.Thread(target=_work)
_t.start()
_t.join()
_res = _holder.get('res') or {}

C.group('A prepare(主线程 build) + execute(工作线程 evolve) 跑通', [
    ('prepare 返回未缓存 handle（含 run）',
     prepared.get('cached') is False and 'run' in prepared),
    ('execute 在工作线程无异常（%s）' % _holder.get('err', '无'), 'err' not in _holder),
    ('结果 ok（真跑了最小单体演化）', _res.get('ok') is True),
    ('落了 run 事件', any(e.get('kind') == 'run' for e in s.activity(0))),
])
C.group('B 不变量：execute 不碰 cache ⇒ 工作线程安全（含致死性对照）', [
    ('**致死性对照**：prepare 在主线程碰了 cache（spy 真在数，≥1）', _main_cache_calls >= 1),
    ('execute 在工作线程 cache 调用 == 0（不碰线程亲和的 SQLite）', _calls['worker'] == 0),
])

# ---- C：run() = prepare+execute 合成，行为不变 ----
_r2 = ops.run(actor='gui', reason='合成路径')
C.group('C run()=prepare+execute 合成，行为不变', [
    ('run 结果 ok', _r2.get('ok') is True),
    ('run 带 summary + artifact_id', bool(_r2.get('summary')) and bool(_r2.get('artifact_id'))),
])

sys.exit(0 if C.summary() else 1)
