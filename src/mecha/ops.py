# -*- coding: utf-8 -*-
"""语义操作（`ops`）：run / estimate / preview / rerun / list_levels / undo。

**通用编排层**：ops 只负责领域无关的事——choke point、authority 门控、事件、可复现产物、
内容寻址缓存；`build_run/evolve_run/estimate/preview/list_levels` 的**物理全部委托注入的
engine**（`runtime.engine()` → adapter → core/protocol）。

**本模块零领域 import**（不 import core/arc/runspec/cost/static_atom/fields）——提炼守卫强制。

纲领落点：
- **I5 人机同一路径**：`run` 经 engine 复用 `runspec` 唯一分派，与"人点运行"完全相同。
- **D7 契约先行**：`estimate`/`preview` 都是纯查询，提交长算前先回答。
- **D8 作业自带可重跑配置**：`run` 落可复现产物（config + RunSpec.pkl + result.npz +
  manifest）到 `.mecha/artifacts/<aid>`；aid = 配置 sha1（内容寻址）⇒ 二次提交零算力命中。
- **I3 大载荷不内联**：`run` 回执只给 `result_summary`（紧凑标量）；大数组落产物 npz。
"""
import json
import os
import time

from . import runtime
from .errors import MechaError
from .autonomy import BACKENDS, route, classify, l3_gate

__all__ = ['Ops', 'BACKENDS', 'artifact_dir', 'save_artifact', 'load_artifact',
           'config_id']


# =====================================================================
# 可复现产物（D8）+ 内容寻址（harvest rid）——领域无关
# =====================================================================
def config_id(config):
    """内容寻址 id：同配置 → 同 id → 二次提交可零算力命中。"""
    import hashlib
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def artifact_dir(root, aid):
    return os.path.join(root, 'artifacts', aid)


def save_artifact(root, aid, config, run, res, summary):
    """落可复现产物：config + RunSpec(pickle) + result(npz) + manifest。"""
    import pickle
    import numpy as np
    d = artifact_dir(root, aid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'runspec.pkl'), 'wb') as f:
        pickle.dump(run, f, protocol=4)
    arrays = {}
    for k in ('pops', 't', 'loss', 'pops_A', 'pops_B', 'loss_A', 'loss_B'):
        v = res.get(k)
        if hasattr(v, 'shape'):
            arrays[k] = np.asarray(v)
    if arrays:
        np.savez_compressed(os.path.join(d, 'result.npz'), **arrays)
    manifest = {'aid': aid, 'config': config, 'method': run.get('method'),
                'two_body': bool(run.get('two_body')), 'summary': summary,
                'created': time.time(), 'result_keys': sorted(arrays)}
    with open(os.path.join(d, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, default=str)
    return {'aid': aid, 'path': os.path.relpath(d, root),
            'files': sorted(os.listdir(d))}


def load_artifact(root, aid):
    """载入产物 → (run, result_arrays, manifest)；缺 → None（不静默造空）。"""
    import pickle
    import numpy as np
    d = artifact_dir(root, aid)
    rp = os.path.join(d, 'runspec.pkl')
    if not os.path.exists(rp):
        return None
    with open(rp, 'rb') as f:
        run = pickle.load(f)
    res = {}
    npz = os.path.join(d, 'result.npz')
    if os.path.exists(npz):
        with np.load(npz) as z:
            res = {k: z[k] for k in z.files}
    man = {}
    mp = os.path.join(d, 'manifest.json')
    if os.path.exists(mp):
        with open(mp, encoding='utf-8') as f:
            man = json.load(f)
    return run, res, man


class Ops:
    """语义操作面。持有 session（配置快照 + 惰性 cache）+ backend；领域计算委托 engine。"""

    def __init__(self, session, backend='sim'):
        route(backend)   # 未实现后端（hardware）→ ReservedBackend（明确报错，不静默当 sim）
        self.session = session
        self.backend = backend

    @property
    def _eng(self):
        return runtime.engine()

    @property
    def root(self):
        return self.session.bus.root

    def autonomy_level(self, op):
        """该操作在当前 backend 下的自治级（L1/L2/L3）。"""
        return classify(op, self.backend)

    # ---------------- D7 纯查询（委托 engine） ----------------
    def estimate(self):
        return self._eng.estimate(self.session.cache(), self.session.effective())

    def preview(self):
        return self._eng.preview(self.session.cache(), self.session.effective())

    def list_levels(self):
        """当前基组的态（label + 身份键 + 能量）——语义寻址，不猜索引（§2.2）。"""
        return self._eng.list_levels(self.session.cache(), self.session.effective())

    # ---------------- L3 数字孪生 dry-run（§6 统一点） ----------------
    def propose(self, *, reason, actor='ai'):
        """AI 提议序列 → 先在 sim 里跑（可逆、零风险）→ 把预测+代价+风险摆给人 →
        **不武装硬件**（requires_confirmation）。放行需 l3_gate 三闸齐过（Phase 5 只留结构）。"""
        if self.session.authority is not None:
            self.session.authority.gate(actor)   # 两模式：propose 也是 AI 侧动作
        est = self.estimate()
        prev = self.preview()
        allow, why = l3_gate('hardware')       # 未武装/未确认/未互锁 → 拒
        proposal = {'proposal': {'backend_target': 'hardware', 'reason': reason,
                                 'actor': actor},
                    'dry_run_sim': {'estimate': est,
                                    'preview_n_states': prev.get('n_states'),
                                    'basis_enough': prev.get('basis_enough'),
                                    'driven': len(prev.get('driven', []))},
                    'requires_confirmation': True, 'armed': False,
                    'l3_gate': {'allow': allow, 'reason': why},
                    'note': '数字孪生预演完成；武装硬件需 人确认 + 硬件互锁（Phase 5 预留）'}
        self.session.bus.append(
            actor=actor, kind='propose', target='hardware', before=None,
            after={'armed': False, 'requires_confirmation': True,
                   'l3_allow': allow}, reason=reason,
            provenance='ops.propose', reversibility='irreversible-if-armed')
        return proposal

    # ---------------- I5 运行 + D8 可复现产物 ----------------
    # run 拆成 prepare（主线程，碰 cache）+ execute（线程安全，纯数值 evolve）：
    # ARC 的 SQLite 连接**线程亲和**（无 check_same_thread=False），两体 build_run 会大量
    # 调 cache.dme → getDipoleMatrixElement → 碰 SQLite。故 build_run 必须在**创建 cache 的
    # 主线程**跑；只有 evolve_run（纯数值、不碰 cache）可进工作线程。GUI RunWorker 与 async
    # job 都用这个拆分（主线程 prepare → 工作线程 execute），run() 则是同线程内两者合成。
    def prepare(self, *, actor, reason, diag=False, use_cache=False,
                land_artifact=True):
        """**主线程阶段**：gate + snapshot + build_run（碰 cache）+ check_run → 不透明 handle。

        handle 交给 `execute`（可在工作线程跑）。内容寻址缓存命中时 handle 标 cached=True，
        execute 直接返回、不 evolve。build_run 在此（主线程）完成 ⇒ 两体 V_dd 的 SQLite
        访问不跨线程（B-1① 定论）。
        """
        eng = self._eng
        if not actor or not reason:
            raise MechaError('run 需要 actor + reason（M1 归因）')
        if self.session.authority is not None:
            self.session.authority.gate(actor)   # 两模式：人工模式下 AI 不能发起运行
        snap = self.session.effective()
        aid = config_id(snap)
        if use_cache and land_artifact:
            cached = load_artifact(self.root, aid)
            if cached is not None and cached[2].get('summary'):
                return {'cached': True, 'aid': aid, 'snap': snap,
                        'summary': cached[2]['summary']}
        cache = self.session.cache()
        run = eng.build_run(cache, snap, diag=diag)          # 唯一分派点（碰 cache→主线程）
        problems = eng.check_run(run)
        return {'cached': False, 'aid': aid, 'snap': snap,
                'run': run, 'problems': problems}

    def execute(self, prepared, *, actor, reason, progress_arr=None,
                land_artifact=True):
        """**线程安全阶段**：evolve_run（纯数值、不碰 cache）+ summary + 产物 + 事件。

        可在工作线程跑（GUI RunWorker / async job）。prepared 来自 `prepare`（主线程）。
        """
        eng = self._eng
        aid = prepared['aid']
        snap = prepared['snap']
        if prepared.get('cached'):
            ev = self.session.bus.append(
                actor=actor, kind='cache_hit', target=aid, before=None,
                after=prepared['summary'], reason=reason,
                provenance={'backend': self.backend, 'config': snap,
                            'artifact': aid}, reversibility='rerun')
            return {'ok': True, 'seq': ev['seq'], 'cached': True,
                    'backend': self.backend, 'artifact_id': aid,
                    'summary': prepared['summary']}
        run = prepared['run']
        res = eng.evolve_run(run, progress_arr=progress_arr)  # 唯一分派点（纯数值）
        summary = eng.result_summary(run, res)
        spec_summ = eng.summarize_run(run)
        art = None
        if land_artifact:
            art = save_artifact(self.root, aid, snap, run, res, summary)
        ann = eng.run_annotations(snap)                       # 领域旁注：n_waves/重叠告警
        ev = self.session.bus.append(
            actor=actor, kind='run',
            target='%s:%s:%s' % (self.backend, run['method'],
                                 'pair' if run['two_body'] else 'single'),
            before=None, after=summary, reason=reason,
            provenance={'backend': self.backend, 'config': snap,
                        'runspec': spec_summ, 'n_waves': ann['n_waves'],
                        'overlap_warning': ann['overlap_warning'],
                        'artifact': (art['path'] if art else None)},
            reversibility='rerun')
        return {'ok': bool(summary.get('ok')), 'seq': ev['seq'], 'cached': False,
                'backend': self.backend, 'method': run['method'],
                'method_in': run.get('method_in'),
                'actual_backend': res.get('backend'),
                'artifact_id': aid, 'artifact': art,
                'problems': prepared['problems'], 'summary': summary,
                'runspec_summary': spec_summ}

    def run(self, *, actor, reason, diag=False, land_artifact=True,
            use_cache=False, progress_arr=None):
        """跑演化：engine.build_run → evolve_run → result_summary（与"人点运行"同一路径 I5）。

        = 同线程内 `prepare` + `execute`。落可复现产物（D8）+ 内容寻址缓存：use_cache=True
        且同配置产物已在 → 零算力命中。需"主线程 build、工作线程 evolve"（GUI/async 两体）
        时分别调 prepare()/execute()。
        """
        prepared = self.prepare(actor=actor, reason=reason, diag=diag,
                                use_cache=use_cache, land_artifact=land_artifact)
        return self.execute(prepared, actor=actor, reason=reason,
                            progress_arr=progress_arr, land_artifact=land_artifact)

    def rerun(self, aid, *, actor, reason):
        """顺可复现产物重跑（D8）：载入 RunSpec → engine.evolve_run（不重新 build）。"""
        eng = self._eng
        loaded = load_artifact(self.root, aid)
        if loaded is None:
            raise MechaError('产物 %r 不存在（无法重跑）' % aid,
                             hint='用 run_compute 先产出，或检查 aid')
        run, _old, man = loaded
        res = eng.evolve_run(run)
        summary = eng.result_summary(run, res)
        ev = self.session.bus.append(
            actor=actor, kind='rerun', target=aid, before=None, after=summary,
            reason=reason, provenance={'backend': self.backend, 'artifact': aid,
                                       'config': man.get('config')},
            reversibility='rerun')
        return {'ok': bool(summary.get('ok')), 'seq': ev['seq'],
                'artifact_id': aid, 'summary': summary}

    # ---------------- 可逆（L2） ----------------
    def undo(self, *, actor, reason='undo'):
        return self.session.undo(actor=actor, reason=reason)

    # ---------------- 只读投影 ----------------
    def state(self):
        """当前配置快照 + 定位包（只读、幂等）。"""
        return {'snapshot': self.session.effective(),
                'orientation': self.session.orientation()}

    def activity(self, since_seq=0):
        return self.session.activity(since_seq)
