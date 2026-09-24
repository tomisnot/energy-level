# -*- coding: utf-8 -*-
"""RunWorker 自检（v2）：QThread 包装 ElOps.execute（主线程 prepare → 工作线程 execute）。

守 I5 人机同路径 + B-1① 线程拆分：GUI 的"跑"经 ops（journal/artifact/可复现），evolve 在
工作线程、build 在主线程（ARC SQLite 线程亲和）。线程安全不变量本身由 ops_threadsplit 守；
这里验 RunWorker 这层 QThread 包装真的把 v2 ElOps.execute 跑通并回传结果。

  A. 主线程 ops.prepare → RunWorker 真起 QThread（start+wait）→ done 收到 (res, None)、res.ok。
  B. 结果落 artifact、可 load_artifact 取回数组（GUI 据此绘图；人机同产物，I3 大载荷不内联）。

offscreen QApplication；最小单体演化（mixed，小网格），秒级。数据根用临时目录。
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from PyQt5.QtWidgets import QApplication                   # noqa: E402
from checks.checklib import Checker                        # noqa: E402
from examples.energy_level.bridge import V2Bridge          # noqa: E402
from src.ui.evolve_worker import RunWorker                 # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)     # offscreen
C = Checker('gui_run_worker')

EL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tmp = tempfile.mkdtemp(prefix='runworker_v2_')
br = V2Bridge.for_gui(el_root=EL_ROOT, data_root=_tmp)     # cache 建在主线程
ops = br.ops

# ---- 最小单体配置（同 ops_threadsplit 范式）----
br.write_many({'groups.list': [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)],
               'field.B_gauss': 0.0, 'field.E_vpcm': 0.0,
               'dynamic.mode': 0, 'dynamic.method': 'mixed',
               'dynamic.t_max': 5.0, 'dynamic.npts': 50, 'dynamic.decay_index': 0},
              reason='判据初始化', actor='gui')
_lv = br._engine.list_levels(br.effective())
br.write('dynamic.psi_single', _lv['states'][0]['key'],
         reason='判据初态', actor='gui')

# ---- A：主线程 prepare → RunWorker 真 QThread execute ----
prepared = ops.prepare(actor='gui', reason='主线程 build')
prog = np.zeros(3)
w = RunWorker(ops, prepared, prog, actor='gui', reason='GUI 运行')
_got = []
w.done.connect(lambda res, err: _got.append((res, err)))
w.start()
w.wait(60000)              # 最多等 60s（最小演化实际秒级）
app.processEvents()        # 投递 worker 线程 queued 过来的 done 信号
res, err = (_got[0] if _got else (None, 'no done signal'))
C.group('A RunWorker(QThread)：主线程 prepare + 工作线程 execute → done(res, None)', [
    ('收到 done 信号（恰好一次）', len(_got) == 1),
    ('无错误（err is None）：%s' % (err or '无'), err is None),
    ('结果 ok（真跑了演化）', isinstance(res, dict) and res.get('ok') is True),
    ('worker 已退出（不悬空）', not w.isRunning()),
])

# ---- B：结果落 artifact、可取回数组（GUI 绘图源）----
aid = res.get('artifact_id') if isinstance(res, dict) else None
loaded = ops.load_artifact(aid) if aid else None
_arrays = loaded[1] if loaded else {}
C.group('B 结果落 artifact、可 load 取回数组（GUI 据此绘图，人机同产物）', [
    ('结果带 artifact_id', bool(aid)),
    ('load_artifact 命中', loaded is not None),
    ('取回演化数组（pops/t 至少其一）',
     ('pops' in _arrays) or ('t' in _arrays)),
])

br.close()
sys.exit(0 if C.summary() else 1)
