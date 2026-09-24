# -*- coding: utf-8 -*-
"""多体接入集成自检（v2）：语义层全链（schema/校验/Gate 往返/build_run 路由/Result/Surface.run/线程拆分）。

用**真实** ElEngine（碰 ARC 标定，首跑建表可能 10–30 s）。守的是"manybody 作为
第 7 引擎家族接入 v2 语义层后，上层零改动即可驱动"这条主线：
  A schema 合并（config_schema/settable_keys 含 manybody.*，solve_schema 不膨胀）
  B engine_params 扫到 manybody 就地 Param（扫描器递归子包）
  C I6 交叉：runspec_mb 实读的 manybody.* 都在 schema（防漂）
  D v2 validate（make_validator）对 atom_list 真契约：拒非法结构、放行合法
  E Gate choke point 往返 → effective 快照（写经门、记事件）
  F build_run(mode=2) → 多体 RunSpec（携带紧凑 calib，不带 H）
  G evolve_run 唯一分派 → Result 契约 + result_summary/labels/summarize/check
  H estimate/preview/list_levels 走多体分支
  I Surface.run 端到端（run_output_required 契约校验：ok+method 缺一即红）
  J prepare(主线程碰 cache)/execute(工作线程纯数值) 线程拆分（B-1① 铁律）

v1 的 ops.run(use_cache=…) 内容寻址缓存在 v2 不存在（诚实退役，不假绿）；
产物缓存等真实调用方出现再按 ADR 流程加。
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                             # noqa: E402

from checks.checklib import Checker                            # noqa: E402
from src.core import protocol as CP                            # noqa: E402
from src.core import runspec as RS                             # noqa: E402

from mecha_v2.engine import ElEngine, make_validator  # noqa: E402
from mecha_v2.bridge import V2Bridge                  # noqa: E402
from mecha.authority import Authority, Mode                      # noqa: E402
from mecha.errors import MechaError                              # noqa: E402
from mecha.gate import Channel, Gate                             # noqa: E402
from mecha.history import History                                # noqa: E402
from mecha.surface import ExecutionContext, Surface              # noqa: E402

C = Checker('manybody_integration')

EL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
engine = ElEngine(EL_ROOT, own_thread=False)   # cache 建在主线程（判据线程）

G = (60, 0, 0.5, 0.5)                     # 60S_1/2 m=+1/2
R = (60, 1, 1.5, 1.5)                     # 60P_3/2 m=+3/2


def _atom(pos, Om=0.0):
    return {'g': list(G), 'r': list(R), 'pos': list(pos), 'Omega_mhz': Om}


MB = {
    'dynamic.mode': 2,
    'dynamic.t_max': 50.0, 'dynamic.npts': 51,
    'manybody.template': 'ising',
    'manybody.terms': ['I1', 'I2'],
    'manybody.atoms': [_atom((0., 0., 0.), 2 * np.pi * 2.0),
                       _atom((6., 0., 0.), 2 * np.pi * 2.0)],
    'manybody.solver': 'auto',
    'manybody.psi0': None,
}

# ---------------- A. schema 合并 ----------------
cs = CP.config_schema()
sk = CP.settable_keys()
C.group('A manybody.* 进 config_schema/settable_keys（solve 不膨胀）', [
    ('config_schema 含 manybody.atoms/template/psi0',
     all(k in cs for k in ('manybody.atoms', 'manybody.template',
                           'manybody.psi0'))),
    ('settable_keys 含 manybody.atoms', 'manybody.atoms' in sk),
    ('solve_schema 仍 5（M1 多体不加 solve.*）', len(CP.solve_schema()) == 5),
    ('manybody.atoms 类型 = atom_list',
     cs['manybody.atoms'].get('type') == 'atom_list'),
])

# ---------------- B. engine_params 扫到 manybody Param ----------------
ep = CP.engine_params()
C.group('B engine_params 扫到 manybody 就地 Param（扫描器递归子包）', [
    ('扫到 _MB_DENSE_MAX/_MB_FULLPOP_MAX/_MB_MAX_DIM',
     all(k in ep for k in ('_MB_DENSE_MAX', '_MB_FULLPOP_MAX', '_MB_MAX_DIM'))),
    ('扫到 MB_GUARD_MHZ/MB_ED_MAX_DIM/MB_N_WINDOW',
     all(k in ep for k in ('MB_GUARD_MHZ', 'MB_ED_MAX_DIM', 'MB_N_WINDOW'))),
    ('home 在 manybody.* 子包',
     all(ep[k]['home'].startswith('manybody.')
         for k in ('_MB_DENSE_MAX', 'MB_GUARD_MHZ'))),
])

# ---------------- C. I6 交叉 ----------------
lit, _dyn = CP.runspec_read_keys()
_declared = set(cs) | set(CP.solve_schema())
mb_lit = [k for k in lit if k.startswith('manybody.')]
C.group('C I6：runspec_mb 实读的 manybody.* 都在 schema（防漂）', [
    ('扫到 manybody.* 实读键 >= 8（含 mb_get 口）', len(mb_lit) >= 8),
    ('无漏声明（所有实读键都在 config+solve schema）',
     all(k in _declared for k in lit)),
])

# ---------------- D. v2 validate 对 atom_list 真契约 ----------------
validate = make_validator(engine)


def _bad(v):
    try:
        validate('manybody.atoms', v)
        return False
    except MechaError:                                          # noqa: BLE001
        return True


C.group('D v2 validate（make_validator）atom_list 真契约（拒非法结构）', [
    ('合法 atoms 通过（不抛）', _bad(MB['manybody.atoms']) is False),
    ('空 list 拒', _bad([])),
    ('g 非四元组拒', _bad([{'g': [60, 0, 0.5], 'r': list(R), 'pos': [0, 0, 0]}])),
    ('pos 非三元组拒', _bad([{'g': list(G), 'r': list(R), 'pos': [0, 0]}])),
    ('非 dict 拒', _bad(['x'])),
    ('Omega 非数值拒',
     _bad([{'g': list(G), 'r': list(R), 'pos': [0, 0, 0], 'Omega_mhz': 'big'}])),
])

# ---------------- E. Gate choke point 往返 ----------------
tmp = tempfile.mkdtemp(prefix='mb_int_')
auth = Authority(mode=Mode.HUMAN)
hist = History()
gate = Gate(auth, hist, validate=validate)
br = V2Bridge(gate, hist, auth, engine=engine, cache=engine._cache,
              artifact_root=os.path.join(tmp, 'artifacts'))
chan = Channel('test', 'human')
for k, v in MB.items():
    gate.set(chan, k, v, reason='mb integration')
snap = br.effective()
C.group('E Gate 往返 → effective 快照（choke point，写经门记事件）', [
    ('manybody.atoms 写后可读回', len(snap.get('manybody.atoms') or []) == 2),
    ('dynamic.mode=2 落快照', snap.get('dynamic.mode') == 2),
    ('写入各产生事件（seq 递增）', hist.last_seq >= len(MB)),
])

# ---------------- F. build_run(mode=2) 路由 ----------------
cache = engine._cache
run = RS.build_run(cache, snap)
C.group('F build_run(mode=2) → 多体 RunSpec', [
    ('nbody=many', run.get('nbody') == 'many'),
    ('method=ed（solver=auto 解析）', run.get('method') == 'ed'),
    ('dim=2^N=4, n_atoms=2', run.get('dim') == 4 and run.get('n_atoms') == 2),
    ('携带紧凑 calib（有 pairs）',
     'calib' in run and len(run['calib'].get('pairs') or []) >= 1),
    ('two_body=False（派生兼容 sweep/gui）', run.get('two_body') is False),
])

# ---------------- G. evolve_run 唯一分派 + 契约 ----------------
res = RS.evolve_run(run)
summ = RS.result_summary(run, res)
labels = RS.result_labels(run, res)
C.group('G evolve_run 唯一分派 → Result 契约 + 摘要', [
    ('Result 含 pops/states/method/backend',
     all(k in res for k in ('pops', 'states', 'method', 'backend'))),
    ('布居归一 <1e-9', float(np.abs(res['pops'].sum(1) - 1).max()) < 1e-9),
    ('result_summary ok', summ.get('ok') is True),
    ('result_labels 数 == pops 列数', len(labels) == res['pops'].shape[1]),
    ('summarize_run 报 nbody=many', RS.summarize_run(run).get('nbody') == 'many'),
    ('check_run 不误报"空演化"',
     not any('空演化' in str(x) for x in RS.check_run(run))),
])

# ---------------- H. estimate/preview/list_levels 分支 ----------------
est = CP.estimate(cache, snap)
prev = CP.preview(cache, snap)
lv = CP.list_levels(cache, snap)
C.group('H estimate/preview/list_levels 走多体分支', [
    ('estimate 报 dim=4 + ed_feasible', est.get('dim') == 4
     and est.get('ed_feasible') is True),
    ('estimate 给 est_sec', 'est_sec' in est),
    ('preview 报耦合对 + max_V', prev.get('n_pairs', 0) >= 1
     and 'max_V_mhz' in prev),
    ('list_levels 报 n_atoms=2 + psi0_hint',
     lv.get('n_atoms') == 2 and 'psi0_hint' in lv),
])

# ---------------- I. Surface.run 端到端（v2 ops 管道） ----------------
surface = Surface(engine)
r_sync = surface.run(snap, context=ExecutionContext())
C.group('I Surface.run 端到端（人机同路径；run_output_required 契约）', [
    ('run ok', r_sync.get('ok') is True),
    ('run 走 manybody（method=ed）', r_sync.get('method') == 'ed'),
    ('回执含 EL 声明的必填键（ok/method）',
     all(k in r_sync for k in engine.schema()['run_output_required'])),
])

# ---------------- J. prepare/execute 线程拆分 ----------------
prepared = br.ops.prepare(actor='gui', reason='mb prepare')
err = {}


def _worker():
    try:
        err['r'] = br.ops.execute(prepared, actor='gui', reason='mb execute')
    except Exception as e:                                      # noqa: BLE001
        err['e'] = '%s: %s' % (type(e).__name__, e)


th = threading.Thread(target=_worker)
th.start()
th.join()
C.group('J prepare(主线程碰 cache)/execute(工作线程纯数值) 线程拆分', [
    ('prepare 成功（calibrate 在主线程，产出 run handle）',
     bool(prepared.get('run'))),
    ('execute 在工作线程成功（不碰线程亲和的 ARC cache）%s'
     % ('' if 'e' not in err else ' [%s]' % err['e']), 'e' not in err),
    ('execute 结果 ok + artifact 落盘',
     bool(err.get('r')) and err['r'].get('ok') is True
     and bool(err['r'].get('artifact_id'))),
])

br.close()
sys.exit(0 if C.summary() else 1)
