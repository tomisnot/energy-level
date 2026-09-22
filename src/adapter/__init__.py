# -*- coding: utf-8 -*-
"""领域适配器（胶水层）——用 `core` 实现 `mecha` 的 `EngineProtocol`。

**分层**：`core`（纯物理 + 调用协议 `core/protocol.py`，不认识 mecha）← 本适配器 →
`mecha`（纯框架，不认识 core）。本层是**唯一**同时 import core 与 mecha 的地方，故
`mecha/`+`cockpit/` 可整包拷走提炼（提炼守卫：mecha 内不得出现 core/arc/adapter 领域名）。

app/判据组装时注入：`mecha.runtime.set_engine(EngineAdapter())`；
GUI 走 `AppBridge(engine=EngineAdapter())`（内部即 set_engine）。
"""
from ..core import protocol as _cp
from ..core.manybody import schema_mb as _mb_schema
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
        """core 的引擎就地常量 + 本领域给的**只读原因**（P1-9a）。

        `readonly_reason` 是领域文案：库只负责在报错里拼接，不写死项目名/项目历史。
        core 不认识 mecha 的展示需求，故这一层在此补齐（薄加工，不改 core）。
        """
        recs = _cp.engine_params()
        for rec in recs.values():
            rec.setdefault(
                'readonly_reason',
                '引擎就地常量（import 期定值），本运行期没有覆写通道')
        return recs

    def config_schema(self):
        """core 的配置 schema + 一处**类型改判**（P2-2，跨模块接缝）。

        `manybody.drive_segments` 在 core 里声明为通用 `type='list'`（只判"是不是
        list"），而它的真契约是**逐段结构**。库的 `validate` 只在 `type` 命中
        `engine.custom_types()` 时才委托 `validate_custom` —— 故这里把它改判为
        自定义类型 `'drive_segments'`，使结构校验在 **`session.set` 当场**生效
        （原行为：要到 build 阶段、甚至工作线程才炸，报错不指向病因）。

        只在**适配层**改判，core 的 `CONFIG_SCHEMA_MB` 一字未动（物理侧单一定义点不变）。
        """
        recs = _cp.config_schema()
        _ds = recs.get('manybody.drive_segments')
        if _ds is not None and _ds.get('type') == 'list':
            _ds = dict(_ds)
            _ds['type'] = 'drive_segments'
            recs['manybody.drive_segments'] = _ds
        return recs

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

    # ---------- 领域私有政策（归还项目侧：mecha-sdk 库内零领域信息） ----------
    def custom_types(self):
        """**声明**本引擎接管的三个自定义类型名（P1-9b / P2-2）。

        库只判"type 在不在这个集合里"，类型名与真契约都留在项目侧。
        声明了就**必须**在 `validate_custom` 里有对应分支——否则库会抛
        `ContractError`（"声明接管却不校验"，见 mecha_kernel_selfcheck F2 组）。
        """
        return ('wave_list', 'atom_list', 'drive_segments')

    def validate_custom(self, typ, value):
        """上述自定义类型的真契约（领域知识）在此。"""
        if typ == 'wave_list':
            return _cp.validate_wave_list(value, err=InvalidValue)
        if typ == 'atom_list':
            return _cp.validate_atom_list(value, err=InvalidValue)
        if typ == 'drive_segments':
            # P2-2：把 drive_segments 的结构真契约**前移到 set 当场**（原只在
            # build 阶段拒，元素不是 dict / 缺 t0 / Om 是标量都要到工作线程才炸）。
            # 这里**只做结构校验**（n_atoms=None）——`registry.validate(key,value)`
            # 拿不到快照，故"Om/delta 长度 == 原子数"这条跨键约束仍留在 build 阶段
            # （schema_mb.validate_drive_segments(..., n_atoms=n) 已在那里调）。
            return _mb_schema.validate_drive_segments(value, err=InvalidValue)
        return value

    def codeact_not_exposed(self):
        """CodeAct 额外屏蔽的 EL 私有名字（裸 core 模块/物理库/波形类）。"""
        return ['static_atom', 'sambe', 'evolve_mixed', 'evolve_twofield', 'hard',
                'coupling', 'field_build', 'atomic_levels', 'arc',
                'waveforms', 'Wave', 'ArbitraryWave', 'GaussianEnv', 'SquareEnv',
                'FileEnv', 'serialize_waves', 'restore_waves']

    def codeact_fields_write(self):
        """建场层里算写的动词（注入 actor='codeact' 归因）。"""
        return ('set_drive', 'add_wave', 'edit_wave', 'remove_wave', 'clear_waves',
                'set_sequence', 'set_waves', 'pi_pulse', 'pi2_pulse', 'square_mw',
                'ramsey', 'stirap', 'arp')

    def codeact_fields_read(self):
        """建场层里算读的动词（直通）。"""
        return ('get_waves', 'get_envelope', 'freq_for', 'intensity_for_rabi',
                'rabi_of', 'describe_drive', 'suggest_freq')

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
