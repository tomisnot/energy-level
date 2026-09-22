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

    # P1-9g：`validate_wave_list`/`validate_atom_list` 两个**空壳**已删——它们只是
    # 库 ABC 旧抽象方法的占位（库内零调用）。真契约走下面的 `custom_types()` +
    # `validate_custom()`（`_cp.validate_wave_list`/`validate_atom_list` 在那里被调）。

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

    # ---------- 引擎自报：库只搬运、不解释（P0-5 / P1-9c） ----------
    def run_meta(self, run):
        """`{'method':…, 'body':…}`：把"哪种体系"从库的硬编码判断收回领域侧。

        原库内写的是 `'pair' if run['two_body'] else 'single'`——那是**物理方言**，
        且非物理领域不实现也会拿到 `single`（默认值掩盖未接线）。现在由本领域明说。
        """
        return {'method': run.get('method'),
                'body': 'pair' if run.get('two_body') else 'single'}

    # ---------- 领域自报 MCP 工具（P1-9d：领域工具不再焊进库） ----------
    def mcp_tools(self):
        """→ 本领域的额外 MCP 工具（`set_drive` / `get_waves`），见 `adapter/mcp_tools.py`。

        这两个工具**原本在 mecha 库里**（签名含 `rabi_mhz`/`intensity_mwcm2`/`pol='π'`
        等物理量）⇒ 非物理项目接进来会凭空多出物理工具（控诉书罪一/L1）。现已搬回本层。
        """
        from .mcp_tools import build_tools
        return build_tools()

    # ---------- 领域身份（P1-9e：库不再顶着别人的名字） ----------
    def server_name(self):
        """MCP 服务名（旧版写死在库里 `'re0-mecha'`）。"""
        return 're0-mecha'

    def server_instructions(self):
        """MCP instructions（旧版写死在库里，且正文写"Re0 物理引擎"）。"""
        return ('Re0 物理引擎语义内核的写通道。list_params 看可调项，set_param 逐项写'
                '配置（记进事件总线、可 undo），estimate_cost 看代价，preview_behavior '
                '提交前预检，run_compute 跑演化（超阈值自动转 async）。键名走 registry '
                '契约；打错会给建议。建场用本领域扩展工具 set_drive/get_waves。')

    def result_arrays(self, res):
        """**本领域自报**要落进可复现产物的数组（P1-8 / P1-9c）。

        为什么必须自报而不是让库按 `.shape` 猜：
        - 多体的 **`S`（纠缠熵）是逐时刻算好的数组**、`corr` 可能是 list ⇒
          `hasattr(v,'shape')` **判不出来**，会被静默漏掉；
        - 单/两体与多体的数组集合**不同族** ⇒ 按族自报，`manifest['result_keys']`
          才与实际落盘一致（"声明 ⊇ 实际"的守卫方向不能反）。

        `pops` 的**语义随规模变**（`engine_ed` 的 `_MB_FULLPOP_MAX`）：dim ≤ 65536 时
        是 2^N 联合分布，超过则**退化为逐原子边缘**（N 列）。调用方**从形状看不出**，
        故 `full_pops` 随 `result_meta` 一起给出（见下）。
        """
        r = res or {}
        out = {}
        for k in ('t', 'pops', 'loss', 'pops_A', 'pops_B', 'loss_A', 'loss_B'):
            if r.get(k) is not None:
                out[k] = r[k]
        if r.get('method') == 'manybody' or 'pops_atom' in r:
            # 多体族：pops_atom(逐原子) / S(逐时刻熵) / corr(末态关联) / psi_amp(末态)
            for k in ('pops_atom', 'S', 'corr', 'psi_amp'):
                if r.get(k) is not None:
                    out[k] = r[k]
        return out

    def result_meta(self, res):
        """`{name: meta}`：随数组一起给消费者**必须知道**的语义标记。

        关键是 `full_pops`：它决定 `pops` 是"2^N 联合分布"还是"逐原子边缘"，
        而两者**形状不同却都叫 pops**（同一字段在不同规模下语义不同）。不给这条，
        消费者会把 N 列边缘当联合分布读——与 `--root` 那类"同名不同义"同族。
        """
        er = (res or {}).get('error_report') or {}
        r = res or {}
        return {'full_pops': er.get('full_pops'),
                'dim': er.get('dim'), 'n_atoms': er.get('n_atoms'),
                'entropy_cut': er.get('entropy_cut'),
                'psi_scope': ('final' if r.get('psi_amp') is not None else None)}

    def annotations(self, snapshot):
        """**不透明**旁注：波形条数 + 重叠告警（库不解读，只透传）。"""
        allw = []
        for atom in ('single', 'A', 'B'):
            allw.extend(snapshot.get('dynamic.waves_%s' % atom) or [])
        from .fields import detect_wave_overlap
        return {'n_waves': len(allw),
                'overlap_warning': detect_wave_overlap(allw)}

    def alerts(self, run, summary, annotations):
        """**本领域自报**要落哪些告警（原判断逻辑在库 `ops.py`，现收回领域侧）。

        规则名/级别/文案与迁移前逐字一致 ⇒ `kind='alert'` 事件形状不变，
        `cockpit/backend.py` 透传侧**零改动**。
        """
        out = []
        _ow = (annotations or {}).get('overlap_warning')
        if _ow:
            out.append({'rule': 'overlap', 'level': 'warn',
                        'reason': (_ow.get('warning')
                                   if isinstance(_ow, dict) else str(_ow))})
        if (summary or {}).get('empty') is True:
            out.append({'rule': 'empty', 'level': 'error',
                        'reason': '结果自报空（after.empty=True）——检查驱动/输入配置'})
        return out

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
        """成本预估 + **保证 `est_sec` 存在**（P0-5 的键契约收口）。

        实测：core 的 `estimate` **只在两体/多体**分支给 `est_sec`（单体分支不给）。
        于是 `run_compute(mode='auto')` 的"超阈值转 async"对**单体永远不触发**——
        旧库用 `or 0.0` 把它静默吞掉。现在由本层补齐：单体给 0.0（单体演化秒级，
        确实不该转 async），并注明来源，使该键**在所有分支都有定义**。
        """
        est = _cp.estimate(cache, snapshot)
        if est.get('est_sec') is None:
            est['est_sec'] = 0.0
            est['est_sec_note'] = ('单体演化秒级；core 单体分支不给 est_sec，'
                                   '由适配层补 0.0（两体/多体由 core 给真实值）')
        return est

    def preview(self, cache, snapshot):
        """行为预览 + 叠加建场层的**重叠检测**（领域知识，不进 core）。"""
        p = _cp.preview(cache, snapshot)
        _ow = (self.annotations(snapshot) or {}).get('overlap_warning')
        p['overlap_warning'] = _ow
        if _ow:
            p.setdefault('notes', []).append(_ow['warning'])
        return p

    def list_levels(self, cache, snapshot):
        return _cp.list_levels(cache, snapshot)

    # P1-9g：`run_annotations` 已删——它是旧 ABC 的弃用方法（docstring 写死
    # `{n_waves, overlap_warning}`）。旁注统一走上面的 `annotations()`（不透明 dict），
    # 库不再硬读它的键。

    # ---------- 建场层（领域可选能力） ----------
    def make_fields(self, session):
        from .fields import Fields
        return Fields(session)
