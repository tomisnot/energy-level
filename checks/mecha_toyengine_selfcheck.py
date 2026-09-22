# -*- coding: utf-8 -*-
"""Phase 3 P0 判据：mecha 内核在**玩具 EngineProtocol** 下全绿，且**不 import 任何领域**。

提炼守卫的包级版：lib（mecha）只认 `protocol.EngineProtocol`；用一个纯 dict 的 ToyEngine
注入后，registry/session/ops 应能 set/run/undo 全链跑通；且全程 `src.core`/`src.adapter`
**不得**进入 sys.modules（证明 mecha ↛ 领域）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checklib import Checker                              # noqa: E402

C = Checker('mecha_toyengine')

# 在 import mecha 之前记录领域模块是否已被载入（应否）
_DOMAIN_BEFORE = [m for m in ('src.core', 'src.adapter') if m in sys.modules]


class ToyEngine:
    """纯 dict 极简 EngineProtocol（<100 行）：lib 自测基座，零领域。"""

    def engine_params(self):
        return {}

    def _rec(self, name, fam, default, lo, hi):
        return {'name': name, 'type': 'float', 'unit': '', 'lo': lo, 'hi': hi,
                'desc': 'toy', 'default': default, 'family': fam, 'readonly': False}

    def config_schema(self):
        return {'x': self._rec('x', 'config', 0.0, -10.0, 10.0)}

    def solve_schema(self):
        return {'solve.MAX_STEP': self._rec('solve.MAX_STEP', 'solve', 0.1, 0.0, 1.0)}

    def solve_map(self):
        return {'MAX_STEP': 'max_step'}

    def settable_keys(self):
        return ['x', 'solve.MAX_STEP']

    def validate_wave_list(self, v):
        return v

    def make_cache(self):
        return object()

    def build_run(self, cache, snapshot, diag=False):
        # RunSpec 需带 method/two_body（ops.execute 读 run['method']/run['two_body']）
        return {'method': 'toy', 'two_body': False, 'spec': dict(snapshot)}

    def check_run(self, run):
        return []

    def evolve_run(self, run, progress_arr=None):
        return {'t': [0.0, 1.0], 'pops': [1.0, 0.0]}

    def result_summary(self, run, res, **kw):
        # summary 需带 ok（ops.execute 读 summary.get('ok')）
        return {'ok': True, 'n': len(res.get('pops', []) or [])}

    def summarize_run(self, run):
        return 'toy run'

    def estimate(self, cache, snapshot):
        return {'sec': 0.01, 'dim': 1}

    def preview(self, cache, snapshot):
        return {'ok': True, 'n_transitions': 0}

    def list_levels(self, cache, snapshot):
        return [{'label': 'a', 'key': 'a', 'E_mhz': 0.0}]

    def run_annotations(self, snapshot):
        return {'n_waves': 0, 'overlap_warning': None}


def main():
    C.check('import mecha 前领域未载入', not _DOMAIN_BEFORE, str(_DOMAIN_BEFORE))

    from mecha import registry, runtime
    from mecha.ops import Ops
    from mecha.session import Session

    runtime.set_engine(ToyEngine())
    # 数据根**显式**（P0-1：mecha 不再从 __file__ 倒推项目根）——用临时目录，不污染仓库。
    import tempfile as _tf
    _root = _tf.mkdtemp(prefix='toy_engine_')
    sess = Session(root=_root)
    ops = Ops(sess)

    # P0-1 致死性：不给根且无 MECHA_ROOT → 必须**明确报错**，不得静默猜路径
    _saved = os.environ.pop('MECHA_ROOT', None)
    try:
        try:
            Session()
            _no_root_raises = False
        except Exception:                              # noqa: BLE001
            _no_root_raises = True
    finally:
        if _saved is not None:
            os.environ['MECHA_ROOT'] = _saved
    C.check('P0-1 未指定数据根 → 明确报错（不猜路径、不静默）', _no_root_raises)

    rec = sess.set('x', 1.5, actor='test', reason='toy set')
    C.check('set 经 choke point 落事件', rec.get('seq', 0) > 0, str(rec))
    C.check('read 回读一致', sess.get('x') == 1.5, str(sess.get('x')))

    keys = [p['name'] for p in registry.list_params()['params']]
    C.check('registry 扫出 ToyEngine schema', 'x' in keys and 'solve.MAX_STEP' in keys,
            str(keys))

    est = ops.estimate()
    C.check('estimate 走 ToyEngine', est.get('sec') == 0.01, str(est))

    res = ops.run(actor='test', reason='toy run', land_artifact=False)
    C.check('run 全链（build/check/evolve/summary）', res.get('ok') is True, str(res))

    sess.undo(actor='test', reason='toy undo')
    C.check('undo 回滚', sess.get('x') in (None, 0.0), str(sess.get('x')))

    domain_after = [m for m in ('src.core', 'src.adapter') if m in sys.modules]
    C.check('全程未 import 领域（mecha ↛ 领域）', not domain_after, str(domain_after))

    ok = C.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
