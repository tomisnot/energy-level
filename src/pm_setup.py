# -*- coding: utf-8 -*-
"""pm_setup.py — Param manager 接入点（引导 + 项目 painter + 注册助手）

全项目唯一 import param_manager 的地方：
  · ensure_path()：定位独立框架（PARAM_MANAGER_HOME 或同级目录）；
  · build_registry(win)：建注册表 + 注册 Energy Level 专用 painter；
  · reg_of(win) / register_panel_params(...)：各面板就近登记的助手。

painter 分派（框架 set_painter 机制）——不同组件类型刷灯语法不同：
  table    QTableWidget：不动整表样式，刷表头底色（背景灯）+ 左上角
           角标色块（来源紫）；
  list     QListWidget：刷视口底色 + 来源紫边框；
  ''（默认）  spinbox/combo/lineedit 等：框架默认通道
           （set_highlight 协议 → setStyleSheet 叠加）。
"""
import os
import sys

# ── 引导独立框架包（所有使用方引用同一份 param_manager）──
# 从本文件逐级向上找 'Param manager' 目录（对层级深度不敏感，防误写）。
if 'param_manager' not in sys.modules:
    _home = os.environ.get('PARAM_MANAGER_HOME')
    if not _home:
        _d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            _c = os.path.join(_d, 'Param manager')
            if os.path.isdir(os.path.join(_c, 'param_manager')):
                _home = _c
                break
            _d = os.path.dirname(_d)
    if _home and _home not in sys.path:
        sys.path.insert(0, _home)

from param_manager.core import ParamRegistry, ParamSpec, RegistryStyle  # noqa: E402
from param_manager import widgets as W                                   # noqa: E402

# 标准件工厂（SLM 式原地替换：QDoubleSpinBox()→pm.spinbox(reg,path,...)）
# 创建即登记；非标件（光场表格/WavePanel/psi_combo/衰变通道）用 register。
spinbox = W.spinbox
combobox = W.combobox
lineedit = W.lineedit
checkbox = W.checkbox
__all__ = ['ParamRegistry', 'ParamSpec', 'RegistryStyle', 'spinbox',
           'combobox', 'lineedit', 'checkbox', 'new_registry',
           'build_registry', 'register_panel_params',
           'engine_spec', 'engine_read', 'engine_write',
           'bind_engine_widgets', 'god_key', 'god_read', 'god_write',
           'god_split', 'register_god_group']


def new_registry(win):
    """建空注册表 + 装项目 painter，挂到 win.param_reg。

    面板构造**之前**调用（MainWindow.__init__ 开头）——标准件在创建处
    就地登记（定义即注册），面板构造时即可拿到 reg。

    auto-commit 防抖：EL 参数改了就即时生效（实时可视化语义），人连改
    多个参数 / AI 连发会触发多次提交。注入单发可重启定时器（每次触摸
    restart，最后一次触摸 150ms 后才 commit_pending）——把短窗口内的
    多次触摸合并成一次提交，框架批语义天然把这一批全部成功项一起标绿
    （无需任何逐参数清灯特例）。
    """
    from PyQt5.QtCore import QTimer
    _register_painters()
    reg = ParamRegistry()
    dt = QTimer()
    dt.setSingleShot(True)                    # 每次 start 重启 = 防抖合并
    dt.timeout.connect(reg.commit_pending)
    reg._commit_debounce = dt                 # 持引用防 GC（reg 随 win 存活）
    reg.set_commit_scheduler(lambda _c: dt.start(150))
    win.param_reg = reg
    return reg


# ═══════════════════════════════════════════════════════════════
#  项目 painter
# ═══════════════════════════════════════════════════════════════
_TABLE_KEY = '_pm_tbl_base'


def _paint_table(widget, qss):
    """QTableWidget：表头底色=状态灯；来源紫=表头边框。不动表格主体
    （整表 setStyleSheet 会染全部单元格/网格，观感差）。"""
    import re
    hh = widget.horizontalHeader()
    if not qss:
        hh.setStyleSheet(getattr(widget, _TABLE_KEY, ''))
        return
    if not hasattr(widget, _TABLE_KEY):
        setattr(widget, _TABLE_KEY, hh.styleSheet())
    m = re.search(r'background-color:\s*([^;]+);', qss)
    b = re.search(r'border:\s*([^;]+);', qss)
    css = hh.styleSheet() if False else getattr(widget, _TABLE_KEY)
    parts = [f'QHeaderView::section {{ background-color: {m.group(1)};'
             f' border: {b.group(1) if b else "1px solid #c0c0c0"}; }}']
    setattr(widget, '_pm_tbl_css', parts[0])
    hh.setStyleSheet(parts[0])


def _paint_list(widget, qss):
    """QListWidget：视口底色=状态灯，来源=边框。"""
    import re
    if not qss:
        widget.setStyleSheet('')
        return
    m = re.search(r'background-color:\s*([^;]+);', qss)
    b = re.search(r'border:\s*([^;]+);', qss)
    css = ''
    if m:
        css += f'QListWidget {{ background-color: {m.group(1)};'
        css += f' border: {b.group(1) if b else "1px solid #c0c0c0"}; }}'
    widget.setStyleSheet(css)


def _register_painters():
    ParamRegistry.set_painter('table', _paint_table)
    ParamRegistry.set_painter('list', _paint_list)


# ═══════════════════════════════════════════════════════════════
#  注册表构建 + 面板登记助手
# ═══════════════════════════════════════════════════════════════

def build_registry(win):
    """SimState 兜底入口：注册表已由 new_registry + 面板就地登记建好时
    直接复用；仅在缺失时（异常路径）重建并补登外部参数。"""
    reg = getattr(win, 'param_reg', None)
    if reg is None:
        reg = new_registry(win)
    for panel in ('field', 'group', 'dynamic'):
        register_panel_params(reg, win, panel)
    reg.seed_baseline()
    return reg


def register_panel_params(reg, win, panel_name):
    """面板就近登记入口（MainWindow._build_ui 尾部调用）。

    标准件（spinbox/combo/lineedit）已在各面板创建处用工厂就地登记
    （定义即注册）；这里只登记外部/非标参数（光场表、能级组、波形、
    psi、衰变通道、模式开关）。
    """
    fn = _PANEL_REG.get(panel_name)
    if fn is None:
        raise KeyError(f'未登记的面板: {panel_name}')
    fn(reg, win)


# 索引型组合框登记助手（工厂只支持文本值；decay_index 用索引，保留）
def _combo_index(reg, path, cb, options, desc='', aliases=()):
    # options 传显示名（供 schema 阅读）；int 校验候选用索引集，避免
    # 字符串候选误拒合法索引
    idx_opts = list(range(len(options)))
    reg.register(path, ParamSpec('int', options=idx_opts,
                                 desc=f"{desc}（{'' .join(f'{i}={o} ' for i, o in enumerate(options))}）",
                                 aliases=list(aliases)),
                 read_fn=cb.currentIndex, write_fn=cb.setCurrentIndex,
                 display_fn=cb.setCurrentIndex, widget=cb,
                 touch_sig=cb.currentIndexChanged)


# ── 各面板的外部/非标参数登记（标准件已工厂就地登记）──

def _reg_field_panel(reg, win):
    """fields.FieldPanel 的外部参数（光场表整表）。"""
    f = win.field
    # 光场表（整表 external；painter=table 刷表头不染网格）
    reg.register('field.lights', ParamSpec('list', desc='光场行列表',
                                           aliases=('light', '光')),
                 read_fn=lambda: f.light_cfgs(),
                 write_fn=lambda v: f.set_lights(v),
                 widget=f.table, touch_sig=f.data_changed,
                 external=True, painter='table')


def _reg_group_panel(reg, win):
    """EnergyGroupPanel（ui_group.py）。"""
    gs = win.groups_store
    reg.register('groups.list', ParamSpec('list', desc='能级组列表',
                                          aliases=('level', '能级')),
                 read_fn=lambda: [dict(g) for g in gs.groups],
                 write_fn=lambda v: win.group_panel.set_groups(v),
                 widget=win.group_panel.group_list,
                 touch_sig=gs.changed, external=True, painter='list')


def _reg_dynamic_panel(reg, win):
    """DynamicPanel 的外部/非标参数（标准件已在创建处工厂登记）。"""
    d = win.dynamic
    _combo_index(reg, 'dynamic.decay_index', d.decay_mode_cb,
                 ['自动（ARC 寿命）', '禁用', '自定义'], '自发辐射档位')

    # 单/双原子模式（= 人点单选钮同路；无控件 touch，显隐由 mode 派生）
    def _set_mode(mode):
        d._is_two_body = (mode == 1)
        d._radio_single.setChecked(mode == 0)
        d._radio_pair.setChecked(mode == 1)
        d._on_mode_changed(mode)
    reg.register('dynamic.mode', ParamSpec('int', options=[0, 1],
                                           desc='0 单原子 / 1 双原子'),
                 read_fn=lambda: 1 if d._is_two_body else 0,
                 write_fn=_set_mode)

    # 波形三面板（独立数据；external+各自 changed 信号；painter=list）
    from .core import waveforms as _wf

    def _wread(panel):
        return lambda: _wf.serialize_waves(panel.waves)

    def _wwrite(panel):
        def _f(data):
            panel.set_waves(_wf.restore_waves(data))
        return _f

    for path, panel, sig, desc in (
            ('dynamic.waves_single', d.wave_panel, d.wave_panel.changed,
             '单原子波形序列'),
            ('dynamic.waves_A', d.wave_panel_A, d.wave_panel_A.changed,
             '原子 A 波形序列'),
            ('dynamic.waves_B', d.wave_panel_B, d.wave_panel_B.changed,
             '原子 B 波形序列')):
        reg.register(path, ParamSpec('list', desc=desc),
                     read_fn=_wread(panel), write_fn=_wwrite(panel),
                     widget=panel, touch_sig=sig, external=True,
                     painter='list')

    # 初态 psi 三兄弟（值=身份键；写走 pending 键机制=恢复存档同路）。
    # touch=combo currentIndexChanged（人侧流水进 op）；_on_m_changed 重填
    # 期已 blockSignals，clear/addItem 中间态不发假信号。
    def _psi_read(which):
        combo = {'single': d.psi_combo, 'A': d.psi_combo_A,
                 'B': d.psi_combo_B}[which]
        return lambda: d._psi_key_for(combo)

    def _psi_write(which):
        def _f(key):
            cur = dict(getattr(d, '_pending_psi_keys', None) or {})
            cur[which] = key
            d._pending_psi_keys = cur
            win.groups_changed.emit()
        return _f

    for which in ('single', 'A', 'B'):
        combo = {'single': d.psi_combo, 'A': d.psi_combo_A,
                 'B': d.psi_combo_B}[which]
        reg.register(f'dynamic.psi_{which}',
                     ParamSpec('list', desc=f'初态身份键（{which}）'),
                     read_fn=_psi_read(which), write_fn=_psi_write(which),
                     widget=combo, touch_sig=combo.currentIndexChanged,
                     external=True)

    # 两体**纠缠**初态（AI 优先；GUI 编辑器后议）：值 = ['pair_sup', […]] 或
    # None。逐原子直积写不出"交换后的结果"（|60P_A,60S_B⟩±|60S_A,60P_B⟩）。
    # 无控件 → 无指示灯（同 dynamic.mode）；真值 = 面板属性 _psi_pair_spec，
    # 两体运行时由 ui_dynamic 取出交给引擎（psi0_pair，**取代** psi_A/psi_B）。
    reg.register('dynamic.psi_pair',
                 ParamSpec('list', desc='两体纠缠初态 spec（pair_sup）/ None'),
                 read_fn=lambda: getattr(d, '_psi_pair_spec', None),
                 write_fn=lambda v: setattr(d, '_psi_pair_spec', v))

    # 衰变通道（自定义档）
    reg.register('dynamic.decays', ParamSpec('list', desc='自定义衰变通道'),
                 read_fn=lambda: [dict(c) for c in d.decay_cfgs],
                 write_fn=lambda v: _set_decays(d, v),
                 widget=d.decay_list, painter='list')


def _set_decays(d, cfgs):
    d.decay_cfgs = [dict(c) for c in cfgs]
    d._update_decay_list()


def _fmt_num(v):
    """数值 → 紧凑显示（与 solver_inspector._fmt_num 同规则）。"""
    if v is None:
        return '不限制'          # 成本预算等"空 = 不限"的旋钮
    if isinstance(v, (list, tuple)):
        return '(' + ', '.join(_fmt_num(x) for x in v) + ')'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return '%g' % f


# ═══════════════════════════════════════════════════════════════
#  上帝模式控制量（统一命名空间）
# ═══════════════════════════════════════════════════════════════
# 分组约定（新增控制参数照这个加，不许再开平行通道）：
#   engine.*  数值精度/成本（src/core/params.py 的 SPECS，32 项）
#   solve.*   求解器级 kwarg（不进 params 注册表，直接喂 evolve_* 签名）
#   （规划）acize.*  AC 化逐对手动 keep/AC；inject.* 哈密顿注入项；
#           pipe.*   手动管线分步 —— 都是"真正的控制参数"，按同一通道登记。
_SOLVE_SPECS = (
    # (名字, 框架 type, 说明, 单位, 下游 kwarg)
    ('MAX_STEP', 'float', 'DOP853 最大步长（ns）；0 = 自适应不限',
     'ns', 'max_step'),
    ('PER_FREQ_M', 'list', '逐载频光子块阶数（int 或 [m1, m2…]）',
     '', 'per_freq_M'),
    ('FROZEN_H_NS', 'float', '纯冻结段长（两体 sambe；0=全自适应）',
     'ns', 'frozen_h_ns'),
    # 成本档：**运行级**（原 engine.PHASE_BUDGET 迁来）。默认空 = 不限制 →
    # 不做任何成本剔除（真实耦合一律保留）。成本只该换速度、不该默默改模型：
    # 运行前预检会报成本并让你选（见 ui_dynamic 的运行前对话框）。
    ('COST_BUDGET', 'float', '微波候选窗成本上限（rad）；留空 = 不限制'
     '（保留全部真实耦合）', 'rad', 'phase_budget'),
)
_SOLVE_KWARG = {n: kw for (n, _t, _d, _u, kw) in _SOLVE_SPECS}
_SOLVE_BY_NAME = {n: (t_, d, u) for (n, t_, d, u, _kw) in _SOLVE_SPECS}


def god_key(group, name):
    """会话 store 的键（= 注册表全路径）。"""
    return f'{group}.{name}'


def god_read(win, key, default=None):
    """读会话 store：**全路径优先**，兼容旧的裸参数名键。"""
    store = win.engine_overrides or {}
    if key in store:
        return store[key]
    short = key.split('.', 1)[1]
    if short in store:
        return store[short]
    if default is not None:
        return default
    from .core import params as _P
    return _P.DEFAULTS.get(short)


def god_write(win, key, value, default=None):
    """写会话 store（只存"真正偏离默认"的项；None/等于默认 = 清覆写）。

    ⚠ 框架 `seed_baseline()`/`load()` 会对每个 binding 调一次
    `write_fn(当前值)` 建基准 —— 不判等的话构造期就会把全部参数写进 store。
    """
    short = key.split('.', 1)[1]
    if default is None:
        from .core import params as _P
        default = _P.DEFAULTS.get(short)

    def _same(a, b):
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            try:
                return list(a) == list(b)
            except TypeError:
                return False
        return a == b

    d = dict(win.engine_overrides or {})
    d.pop(short, None)                      # 统一成全路径键
    if value is None or (default is not None and _same(value, default)):
        d.pop(key, None)
    else:
        d[key] = value
    win.set_engine_overrides(d)


def god_split(store):
    """会话 store → (param_overrides, engine_kwargs)。

    单一映射点：新增分组只改这里 + `_SOLVE_SPECS`，不再散落各处。
    """
    from .core import params as _P
    po, ek = {}, {}
    for k, v in (store or {}).items():
        if k.startswith('engine.'):
            po[k.split('.', 1)[1]] = v
        elif k.startswith('solve.'):
            name = k.split('.', 1)[1]
            if name in _SOLVE_KWARG:
                ek[_SOLVE_KWARG[name]] = v
        elif k in _P.SPEC_BY_NAME:           # 兼容：旧的裸参数名键
            po[k] = v
    return po, ek


def register_god_group(reg, win, group, editors=None):
    """登记一组上帝模式控制量（engine.* / solve.* 同一条通道）。

    editors: {全路径或短名: 控件} —— 给了就顺手 `bind_widget` 绑上
    （**必须有控件才有指示灯**：框架 `_paint()` 在 widget 为 None 时直接返回）。
    """
    from .core import params as _P
    got = {}
    # ⚠ 幂等：已登记过的路径**不再 register**——框架的 register 会新建 binding，
    # 把已有的 widget（灯）绑定洗掉。二次调用（补绑求解页旋钮）只做绑定。
    if group == 'engine':
        for sp in _P.specs():
            path = god_key('engine', sp.name)
            if path not in reg._bindings:
                reg.register(path, engine_spec(sp),
                             read_fn=engine_read(win, sp.name),
                             write_fn=engine_write(win, sp.name), painter='')
            got[sp.name] = engine_spec(sp)
    elif group == 'solve':
        for name, (kind, desc, unit) in _SOLVE_BY_NAME.items():
            path = god_key('solve', name)
            spec = ParamSpec(kind, default=None, unit=unit, desc=desc,
                             group='solve', commit='auto')
            if path not in reg._bindings:
                reg.register(path, spec,
                             read_fn=(lambda n=name: god_read(
                                 win, god_key('solve', n))),
                             write_fn=(lambda v, n=name: god_write(
                                 win, god_key('solve', n), v)),
                             painter='')
            got[name] = spec
    if editors:
        for name, w in editors.items():
            path = name if '.' in name else god_key(group, name)
            short = path.split('.', 1)[1]
            if short not in got:
                continue
            reg.bind_widget(path, w, kind='lineedit', spec=got[short],
                            read_fn=(lambda p=path: god_read(win, p)),
                            write_fn=(lambda v, p=path: god_write(win, p, v)),
                            display_fn=_god_disp(w))
    return len(got)


def _god_disp(w):
    def _f(v):
        w.blockSignals(True)
        w.setText('' if v is None else _fmt_num(v))
        w.blockSignals(False)
    return _f


def engine_param_kind(sp):
    """ParamSpec.kind（params 注册表）→ 参数管理框架的 type。"""
    return {'float': 'float', 'int': 'int', 'tuple': 'list'}[sp.kind]


def engine_spec(sp):
    """params 注册表的 ParamSpec → 框架 ParamSpec（desc 带典型范围提示）。

    **不传 range**：框架把 range 当硬校验会拒绝越界值，而 params 里的 lo/hi
    只是 UI 提示（探索性取值不该被拦），故并进 desc。
    """
    rng = ('' if sp.lo is None or sp.hi is None else
           '（典型范围 %g~%g %s）' % (sp.lo, sp.hi, sp.unit))
    return ParamSpec(engine_param_kind(sp), default=None, unit=sp.unit,
                     desc=(sp.desc.splitlines()[0][:150] + rng),
                     group='engine', commit='auto',
                     aliases=list(sp.name.lstrip('_').lower().split('_')))


def engine_read(win, name):
    """`engine.<name>` 的读：会话覆写优先，否则 params 默认值。"""
    return lambda: god_read(win, god_key('engine', name))


def engine_write(win, name):
    """`engine.<name>` 的写：进会话 store（只存偏离默认的项）。"""
    return lambda v: god_write(win, god_key('engine', name), v)


def bind_engine_widgets(reg, win, editors, fmt=None):
    """把引擎参数页的输入框绑到 `engine.*`（灯 / 校验 / 提交都走框架）。

    为什么必须绑：框架 `_paint()` 在 `binding.widget is None` 时直接返回——
    只 register 不绑控件 = 永远没有指示灯（实测踩过）。
    """
    return register_god_group(reg, win, 'engine', editors=editors)


def _reg_engine_params(reg, win):
    """引擎参数（src/core/params.py 的 SPECS）→ `engine.*` 在册面。

    真值 = `win.engine_overrides`（会话覆写）叠加 params 默认值：
      · 人的路径：检视器「引擎参数」页的输入框（`bind_engine_widgets`
        把它们绑到同一批 `engine.*` 上 → 灯/校验/提交走框架）；
      · AI 路径：MCP `set_param('engine.PHASE_TOL', ...)` → 框架 set →
        **同一个 write_fn** → 同一个 store（P2 人机同一状态）；
      · **与模式无关**：极简模式下这些参数照样在册、照样生效
        （议题 0 三轴分离：登记跟真值源走，不跟显隐走）。
    控件绑定发生在检视器页构造时（此时才有控件），见 `bind_engine_widgets`。
    """
    register_god_group(reg, win, 'engine')


_PANEL_REG = {
    'field': _reg_field_panel,
    'group': _reg_group_panel,
    'dynamic': _reg_dynamic_panel,
    'engine': _reg_engine_params,
    'solve': lambda reg, win: register_god_group(reg, win, 'solve'),
}
