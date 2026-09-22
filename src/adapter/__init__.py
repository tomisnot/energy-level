# -*- coding: utf-8 -*-
"""领域适配器（胶水层）——用 `core` 实现 `mecha` 的 `EngineProtocol`。

**分层**：`core`（纯物理 + 调用协议 `core/protocol.py`，不认识 mecha）← 本适配器 →
`mecha`（纯框架，不认识 core）。本层是**唯一**同时 import core 与 mecha 的地方，故
`mecha/`+`cockpit/` 可整包拷走提炼（提炼守卫：mecha 内不得出现 core/arc/adapter 领域名）。

app/判据组装时注入：`mecha.runtime.set_engine(EngineAdapter())`；
GUI 走 `AppBridge(engine=EngineAdapter())`（内部即 set_engine）。
"""
from ..core import protocol as _cp
from mecha.errors import InvalidValue
from mecha.protocol import EngineProtocol

__all__ = ['EngineAdapter']


class EngineAdapter(EngineProtocol):
    """把 `core.protocol` 的领域面适配成 mecha 的 `EngineProtocol`。

    薄委托为主；两处"加工"：① wave_list 校验注入 mecha 的 `InvalidValue`（core 不认识
    mecha 错误类型）；② preview/run_annotations 叠加**建场层的重叠检测**（detect_wave_overlap
    住在 adapter.fields，是领域逻辑，不进 core）。
    """

    def __init__(self, atom='Rubidium87'):
        self.atom = atom

    # ---------- schema ----------
    def engine_params(self):
        return _cp.engine_params()

    def config_schema(self):
        return _cp.config_schema()

    def solve_schema(self):
        return _cp.solve_schema()

    def solve_map(self):
        return _cp.solve_map()

    def settable_keys(self):
        return _cp.settable_keys()

    def validate_wave_list(self, value):
        return _cp.validate_wave_list(value, err=InvalidValue)

    def validate_atom_list(self, value):
        return _cp.validate_atom_list(value, err=InvalidValue)

    # ---------- cache ----------
    def make_cache(self):
        return _cp.make_cache(self.atom)

    # ---------- run ----------
    def build_run(self, cache, snapshot, diag=False):
        return _cp.build_run(cache, snapshot, diag=diag)

    def check_run(self, run):
        return _cp.check_run(run)

    def evolve_run(self, run, progress_arr=None):
        return _cp.evolve_run(run, progress_arr=progress_arr)

    def result_summary(self, run, res, **kw):
        return _cp.result_summary(run, res, **kw)

    def summarize_run(self, run):
        return _cp.summarize_run(run)

    # ---------- query（preview 叠加建场层的重叠检测） ----------
    def estimate(self, cache, snapshot):
        return _cp.estimate(cache, snapshot)

    def preview(self, cache, snapshot):
        p = _cp.preview(cache, snapshot)
        ann = self.run_annotations(snapshot)
        p['overlap_warning'] = ann['overlap_warning']
        if ann['overlap_warning']:
            p.setdefault('notes', []).append(ann['overlap_warning']['warning'])
        return p

    def list_levels(self, cache, snapshot):
        return _cp.list_levels(cache, snapshot)

    def run_annotations(self, snapshot):
        """{n_waves, overlap_warning}——重叠检测复用建场层的 detect_wave_overlap（领域）。"""
        allw = []
        for atom in ('single', 'A', 'B'):
            allw.extend(snapshot.get('dynamic.waves_%s' % atom) or [])
        from .fields import detect_wave_overlap
        return {'n_waves': len(allw),
                'overlap_warning': detect_wave_overlap(allw)}

    # ---------- 建场层（领域可选能力） ----------
    def make_fields(self, session):
        from .fields import Fields
        return Fields(session)
