# -*- coding: utf-8 -*-
"""多体接入集成自检：语义层全链（schema/校验/build_run 路由/Result/查询/产物/线程拆分）。

用**真实** EngineAdapter + Session + Ops（碰 ARC 标定，首跑建表可能 10–30 s）。
守的是"manybody 作为第 7 引擎家族接入语义层后，上层零改动即可驱动"这条主线：
  A schema 合并（config_schema/settable_keys 含 manybody.*，solve_schema 不膨胀）
  B engine_params 扫到 manybody 就地 Param（扫描器递归子包）
  C I6 交叉：runspec_mb 实读的 manybody.* 都在 config_schema（防漂）
  D registry.validate atom_list 真契约（拒非法结构）
  E session choke point 往返 → effective 快照
  F build_run(mode=2) → 多体 RunSpec（携带紧凑 calib，不带 H）
  G evolve_run 唯一分派 → Result 契约 + result_summary/labels/summarize/check
  H estimate/preview/list_levels 走多体分支
  I ops.run 端到端（人机同路径 I5）+ 内容寻址产物缓存命中
  J prepare(主线程碰 cache)/execute(工作线程纯数值) 线程拆分（B-1① 铁律）
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                             # noqa: E402

from checks.checklib import Checker                            # noqa: E402
from src.mecha import runtime                                  # noqa: E402
from src.adapter import EngineAdapter                          # noqa: E402

runtime.set_engine(EngineAdapter())       # 注入领域引擎（判据在开头做）

from src.mecha.session import Session                          # noqa: E402
from src.mecha.ops import Ops                                  # noqa: E402
from src.mecha import registry                                 # noqa: E402
from src.core import protocol as CP                            # noqa: E402
from src.core import runspec as RS                             # noqa: E402

C = Checker('manybody_integration')

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
fam = registry.families()
mb_lit = [k for k in lit if k.startswith('manybody.')]
C.group('C I6：runspec_mb 实读的 manybody.* 都在 config_schema（防漂）', [
    ('扫到 manybody.* 实读键 >= 8（含 mb_get 口）', len(mb_lit) >= 8),
    ('无漏声明（所有实读键都在 families）', all(k in fam for k in lit)),
])

# ---------------- D. registry.validate atom_list ----------------
def _bad(v):
    try:
        registry.validate('manybody.atoms', v)
        return False
    except Exception:                                          # noqa: BLE001
        return True


ok_atoms = registry.validate('manybody.atoms', MB['manybody.atoms'])
C.group('D registry.validate atom_list 真契约（拒非法结构）', [
    ('合法 atoms 通过（归一为 list）',
     isinstance(ok_atoms, list) and len(ok_atoms) == 2),
    ('空 list 拒', _bad([])),
    ('g 非四元组拒', _bad([{'g': [60, 0, 0.5], 'r': list(R), 'pos': [0, 0, 0]}])),
    ('pos 非三元组拒', _bad([{'g': list(G), 'r': list(R), 'pos': [0, 0]}])),
    ('非 dict 拒', _bad(['x'])),
    ('Omega 非数值拒',
     _bad([{'g': list(G), 'r': list(R), 'pos': [0, 0, 0], 'Omega_mhz': 'big'}])),
])

# ---------------- E. session choke point 往返 ----------------
tmp = tempfile.mkdtemp(prefix='mb_int_')
sess = Session(root=tmp)
for k, v in MB.items():
    sess.set(k, v, actor='test', reason='mb integration')
snap = sess.effective()
C.group('E session 往返 → effective 快照（choke point）', [
    ('manybody.atoms 写后可读回', len(snap.get('manybody.atoms') or []) == 2),
    ('dynamic.mode=2 落快照', snap.get('dynamic.mode') == 2),
    ('写入各产生事件（seq 递增）', sess.bus.last_seq >= len(MB)),
])

# ---------------- F. build_run(mode=2) 路由 ----------------
ops = Ops(sess)
cache = ops.session.cache()
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

# ---------------- I. ops.run 端到端 + 内容寻址缓存 ----------------
r_sync = ops.run(actor='test', reason='mb run')
r_cache = ops.run(actor='test', reason='mb run again', use_cache=True)
C.group('I ops.run 端到端（人机同路径 I5）+ 内容寻址缓存', [
    ('run ok', r_sync.get('ok') is True),
    ('run 走 manybody（method=ed）', r_sync.get('method') == 'ed'),
    ('产物已落（artifact_id）', bool(r_sync.get('artifact_id'))),
    ('二次 use_cache 零算力命中', r_cache.get('cached') is True),
])

# ---------------- J. prepare/execute 线程拆分 ----------------
prepared = ops.prepare(actor='test', reason='mb prepare')
err = {}


def _worker():
    try:
        ops.execute(prepared, actor='test', reason='mb execute')
    except Exception as e:                                      # noqa: BLE001
        err['e'] = '%s: %s' % (type(e).__name__, e)


th = threading.Thread(target=_worker)
th.start()
th.join()
C.group('J prepare(主线程碰 cache)/execute(工作线程纯数值) 线程拆分', [
    ('prepare 成功（calibrate 在主线程，产出 run handle）',
     bool(prepared.get('run')) or bool(prepared.get('cached'))),
    ('execute 在工作线程成功（不碰线程亲和的 ARC cache）', 'e' not in err),
])

sys.exit(0 if C.summary() else 1)
