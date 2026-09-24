# -*- coding: utf-8 -*-
"""ops prepare/execute 线程拆分判据（v2 ElOps）—— 守 B-1① 定论。

背景：ARC 的 `sqlite3.connect` 无 `check_same_thread=False` → 连接**线程亲和**；
两体 V_dd 大量调 `cache.dme → 碰 SQLite`。若 build_run 在工作线程跑（而 cache 建于
主线程）→ `ProgrammingError: SQLite objects created in a thread...`。故 GUI 运行路径
拆成 `prepare`（主线程：build_run+check_run）+ `execute`（工作线程：evolve_run）。

v2 装配：`V2Bridge.for_gui`（own_thread=False，cache 建在**主线程**）+ `bridge.ops`
（ElOps，与 v1 Ops 同形的 prepare/execute/load_artifact）。

  A. prepare（主线程 build）返回 {run,problems}；execute（工作线程 evolve）无异常、
     结果 ok、artifact 落盘可取回（pops/t）。
  B. **诚实弱化**（对照 v1）：v1 用 spy 断言"execute 全程 cache 调用==0"；v2 的
     ElOps.execute 把 cache 交给 evolve_run 作参数（消耗方是否触碰由 EL 内部决定），
     桥层不再持有可 spy 的 cache 包装 ⇒ 本判据退而测**可证伪的硬后果**：工作线程
     execute 不炸 ProgrammingError 且 ok=True（若 evolve 真在主线程 cache 上跨线程
     动手，SQLite 必抛——这是同一约束的另一面）。
  C. execute 回执顶层带 problems（RunWorker 消费 res['problems'] 发状态栏旁注）。

真跑最小单体演化（mixed，小网格）——秒级。数据根用临时目录（不污染仓内 .mecha-v2）。
"""
import os
import sys
import tempfile
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from mecha_v2.bridge import V2Bridge          # noqa: E402

EL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C = Checker('ops_threadsplit')

_TMP = tempfile.mkdtemp(prefix='opsplit_v2_')
br = V2Bridge.for_gui(el_root=EL_ROOT, data_root=_TMP)   # cache 建在主线程

_GROUPS = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
br.write_many({'groups.list': _GROUPS, 'field.B_gauss': 0.0, 'field.E_vpcm': 0.0,
               'dynamic.mode': 0, 'dynamic.method': 'mixed',
               'dynamic.t_max': 5.0, 'dynamic.npts': 50, 'dynamic.decay_index': 0},
              reason='判据初始化', actor='gui')
_lv = br._engine.list_levels(br.effective())
br.write('dynamic.psi_single', _lv['states'][0]['key'],
         reason='判据初态', actor='gui')

# ---- A/B：prepare 主线程、execute 工作线程 ----
prepared = br.ops.prepare(actor='gui', reason='主线程 build')
_holder = {}


def _work():
    try:
        _holder['res'] = br.ops.execute(prepared, actor='gui',
                                        reason='工作线程 evolve',
                                        progress_arr=np.zeros(3))
    except Exception as e:                                # noqa: BLE001
        _holder['err'] = '%s: %s' % (type(e).__name__, e)


_t = threading.Thread(target=_work)
_t.start()
_t.join()
_res = _holder.get('res') or {}
aid = _res.get('artifact_id')
loaded = br.ops.load_artifact(aid) if aid else None

C.group('A prepare(主线程 build)+execute(工作线程 evolve) 跑通', [
    ('prepare 返回 {run,problems}（v1 handle 形状）',
     isinstance(prepared.get('run'), dict) and 'problems' in prepared),
    ('execute 在工作线程无异常（%s）' % _holder.get('err', '无'), 'err' not in _holder),
    ('结果 ok（真跑了最小单体演化）', _res.get('ok') is True),
    ('artifact 落盘、可取回 pops/t（GUI 绘图消费形状）',
     loaded is not None and 'pops' in loaded[1] and 't' in loaded[1]),
])
C.group('B 不变量（诚实弱化）：工作线程 execute 不炸 SQLite 线程亲和', [
    ('**致死性对照**：prepare 在主线程真碰了 cache（build_run 成功即证）',
     bool(prepared.get('run'))),
    ('execute 无 ProgrammingError（若 evolve 跨线程碰 cache，SQLite 必抛）',
     'err' not in _holder and 'ProgrammingError' not in str(_holder.get('err', ''))),
    ('（v1 的 spy==0 断言退役：v2 ElOps.execute 把 cache 作参数交给 evolve_run，'
     '桥层无可 spy 包装——见 docstring B 组说明）', True),
])
C.group('C execute 回执顶层带 problems（RunWorker 旁注消费形状）', [
    ('顶层 problems 在且与 prepare 一致',
     'problems' in _res and _res['problems'] == prepared.get('problems')),
    ('summary 带 artifact_id（人机同产物，I3）', bool(aid)),
])

br.close()
sys.exit(0 if C.summary() else 1)
