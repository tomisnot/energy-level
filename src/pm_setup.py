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
           'build_registry', 'register_panel_params']


def new_registry(win):
    """建空注册表 + 装项目 painter，挂到 win.param_reg。

    面板构造**之前**调用（MainWindow.__init__ 开头）——标准件在创建处
    就地登记（定义即注册），面板构造时即可拿到 reg。
    """
    _register_painters()
    reg = ParamRegistry()
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

    # 衰变通道（自定义档）
    reg.register('dynamic.decays', ParamSpec('list', desc='自定义衰变通道'),
                 read_fn=lambda: [dict(c) for c in d.decay_cfgs],
                 write_fn=lambda v: _set_decays(d, v),
                 widget=d.decay_list, painter='list')


def _set_decays(d, cfgs):
    d.decay_cfgs = [dict(c) for c in cfgs]
    d._update_decay_list()


_PANEL_REG = {
    'field': _reg_field_panel,
    'group': _reg_group_panel,
    'dynamic': _reg_dynamic_panel,
}
