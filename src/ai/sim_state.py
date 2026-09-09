# -*- coding: utf-8 -*-
"""SimState：语义状态的统一门面（人机等效的"唯一真值通道"）。

实现策略（对草案 §2.1 的务实收敛）：不立即把参数从控件搬进独立核，
而是把 MainWindow 现有的三条真值路径收口成一个门面——
  读：group_panel/field/dynamic 的 serialize_state()（= 存档同源）
  写：能级组走 groups_store.add/remove_at（人点"添加"的同一条路）；
      参数走控件 setValue（人敲 spinbox 的同一条路）；
      波形走 wave_panel.set_waves（人恢复存档的同一条路）。
GUI 控件仍是当前阶段的真值源，但**所有消费者（人/存档/AI）都经过
同一组语义入口**，变更自动触发既有信号链刷新界面——P2 等效成立。
后续若需彻底"值入核"（视图与语义分离），替换本文件内部实现即可，
调用方（MCP 工具/journal/undo）不感知。

所有方法必须在 Qt 主线程执行（经 ai.dispatch.run_on_main 投递）。
"""
import json

from ..core import waveforms


def flash_widget(w, ms=1500):
    """P3 协同 UI：AI 改过的控件短暂紫色描边（人自己改的不闪）。

    保存控件原有 level 样式表，singleShot 还原；主线程调用。
    """
    from PyQt5.QtCore import QTimer
    base = w.styleSheet()
    w.setStyleSheet(base + '\nborder: 2px solid #a855f7;')
    QTimer.singleShot(ms, lambda: w.setStyleSheet(base))


def json_safe(obj):
    """numpy 标量/数组 → 原生 JSON 可序列化结构（快照落盘/回传用）。"""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if hasattr(obj, 'item'):            # numpy 标量
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, 'tolist'):          # numpy 数组
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# set_param 白名单：path → (取值, 赋值, 类型)。取值/赋值都经控件，
# 与人的操作逐字节等效（赋值自动触发既有 changed 信号链）。
_PARAM_MAP = {
    'field.B_gauss':      lambda w: (w.field.b_spin.value(), float),
    'field.E_vpcm':       lambda w: (w.field.e_spin.value(), float),
    'field.window':       lambda w: (w.field.window_spin.value(), int),
    'field.n_photon':     lambda w: (w.field.n_photon_spin.value(), int),
    'dynamic.t_max':      lambda w: (w.dynamic.tmax_sb.value(), float),
    'dynamic.npts':       lambda w: (w.dynamic.npts_sb.value(), int),
    'dynamic.method':     lambda w: (w.dynamic.method_cb.currentText(), str),
    'dynamic.frozen_h_ns': lambda w: (w.dynamic.frozen_h_sb.value(), float),
    'dynamic.per_freq_M': lambda w: (w.dynamic.perfreq_m_edit.text(), str),
    'dynamic.pair.R_um':  lambda w: (w.dynamic.vdd_R_sb.value(), float),
    'dynamic.pair.theta_deg': lambda w: (w.dynamic.vdd_theta_sb.value(), float),
    'dynamic.pair.phi_deg':   lambda w: (w.dynamic.vdd_phi_sb.value(), float),
    # 自发辐射档位索引：0 自动 / 1 禁用 / 2 自定义（= 人切下拉框同一路径）
    'dynamic.decay_index': lambda w: (w.dynamic.decay_mode_cb.currentIndex(), int),
    # 单/双原子模式：0 单原子 / 1 双原子（= 人点单选钮同一路径）
    'dynamic.mode': lambda w: (1 if w.dynamic._is_two_body else 0, int),
}
_PARAM_SET = {
    'field.B_gauss':      lambda w, v: w.field.b_spin.setValue(float(v)),
    'field.E_vpcm':       lambda w, v: w.field.e_spin.setValue(float(v)),
    'field.window':       lambda w, v: w.field.window_spin.setValue(int(v)),
    'field.n_photon':     lambda w, v: w.field.n_photon_spin.setValue(int(v)),
    'dynamic.t_max':      lambda w, v: w.dynamic.tmax_sb.setValue(float(v)),
    'dynamic.npts':       lambda w, v: w.dynamic.npts_sb.setValue(int(v)),
    'dynamic.method':     lambda w, v: w.dynamic.method_cb.setCurrentText(str(v)),
    'dynamic.frozen_h_ns': lambda w, v: w.dynamic.frozen_h_sb.setValue(float(v)),
    'dynamic.per_freq_M': lambda w, v: w.dynamic.perfreq_m_edit.setText(str(v)),
    'dynamic.pair.R_um':  lambda w, v: w.dynamic.vdd_R_sb.setValue(float(v)),
    'dynamic.pair.theta_deg': lambda w, v: w.dynamic.vdd_theta_sb.setValue(float(v)),
    'dynamic.pair.phi_deg':   lambda w, v: w.dynamic.vdd_phi_sb.setValue(float(v)),
    'dynamic.decay_index': lambda w, v: w.dynamic.decay_mode_cb.setCurrentIndex(int(v)),
    'dynamic.mode': lambda w, v: _set_mode(w, int(v)),
}


def _set_mode(w, mode):
    """切单/双原子模式（复刻 restore_state 的人用路径：置钮 + on_mode_changed）。"""
    w.dynamic._is_two_body = (mode == 1)
    w.dynamic._radio_single.setChecked(mode == 0)
    w.dynamic._radio_pair.setChecked(mode == 1)
    w.dynamic._on_mode_changed(mode)
# path → 被改控件（P3：AI 改过短暂高亮，人一眼看到"AI 动了这里"）
_PARAM_WIDGET = {
    'field.B_gauss':      lambda w: w.field.b_spin,
    'field.E_vpcm':       lambda w: w.field.e_spin,
    'field.window':       lambda w: w.field.window_spin,
    'field.n_photon':     lambda w: w.field.n_photon_spin,
    'dynamic.t_max':      lambda w: w.dynamic.tmax_sb,
    'dynamic.npts':       lambda w: w.dynamic.npts_sb,
    'dynamic.method':     lambda w: w.dynamic.method_cb,
    'dynamic.frozen_h_ns': lambda w: w.dynamic.frozen_h_sb,
    'dynamic.per_freq_M': lambda w: w.dynamic.perfreq_m_edit,
    'dynamic.pair.R_um':  lambda w: w.dynamic.vdd_R_sb,
    'dynamic.pair.theta_deg': lambda w: w.dynamic.vdd_theta_sb,
    'dynamic.pair.phi_deg':   lambda w: w.dynamic.vdd_phi_sb,
    'dynamic.decay_index': lambda w: w.dynamic.decay_mode_cb,
}


class SimState:
    """门面：持有 MainWindow 引用，不持有任何状态。"""

    def __init__(self, win):
        self.win = win

    # ---------- 读（与"文件→保存状态"同源） ----------
    def to_dict(self):
        w = self.win
        return json_safe({
            'meta': {'version': 2},
            'groups': w.group_panel.serialize_state(),
            'field': w.field.serialize_state(),
            'dynamic': w.dynamic.serialize_state(),
        })

    def get_state(self, scope=None):
        """全量或按模块导出（scope ∈ {None,'groups','field','dynamic'}）。"""
        d = self.to_dict()
        if scope in ('groups', 'field', 'dynamic'):
            return {scope: d[scope]}
        return d

    def summary(self):
        """轻量摘要（灌 context 友好）：组数/场强/波形数/方法。"""
        d = self.to_dict()
        waves = d['dynamic'].get('waves') or []
        return {
            'n_groups': len(d['groups'].get('groups', [])),
            'groups': [f"{g['label']}(n={g['n']},l={g['l']},j={g['j']})"
                       for g in d['groups'].get('groups', [])],
            'B_gauss': d['field'].get('B_gauss'),
            'E_vpcm': d['field'].get('E_vpcm'),
            'n_lights': len(d['field'].get('lights', [])),
            'method': d['dynamic']['dynamic'].get('method'),
            't_max_ns': d['dynamic']['dynamic'].get('t_max'),
            'n_waves': len(waves),
            'two_body': bool(d['dynamic']['dynamic'].get('mode')),
        }

    # ---------- 写（全部走人用的同一条路） ----------
    def set_param(self, path, value):
        """白名单参数写。返回 (ok, msg, {before, after})。"""
        w = self.win
        if path not in _PARAM_MAP:
            return False, f'未知参数 {path!r}；可用: ' + ', '.join(sorted(_PARAM_MAP)), {}
        before, cast = _PARAM_MAP[path](w)
        try:
            value = cast(value)
        except (TypeError, ValueError) as e:
            return False, f'值 {value!r} 类型不合法（需 {cast.__name__}）: {e}', {}
        try:
            _PARAM_SET[path](w, value)
        except Exception as e:        # noqa: BLE001 - 控件范围拒绝等
            return False, f'写入失败: {e}', {}
        after, _ = _PARAM_MAP[path](w)
        if after == value or str(after) == str(value):
            return True, 'ok', {'before': before, 'after': after}
        # spinbox 范围钳制：值被改写 → 如实报告
        return True, f'已写入（控件范围钳制为 {after}）', {'before': before, 'after': after}

    def add_level_group(self, n, l, j):
        """与人点"添加"逐字节等效：设 spin → 走 panel.add_group（含校验）。"""
        w = self.win
        p = w.group_panel
        p.n_spin.setValue(int(n)); p.l_spin.setValue(int(l)); p.j_spin.setValue(float(j))
        before = len(p.owner.groups_store.groups)
        p.add_group()
        after = len(p.owner.groups_store.groups)
        if after == before:
            from ..core import core as _core
            ok, msg = _core.validate_state(int(n), int(l), float(j))
            return False, (msg if not ok else '添加失败（可能重复）'), {}
        return True, 'ok', {'before': before, 'after': after}

    def remove_level_group(self, idx):
        """与人右键删除等效：takeItem + store.remove_at。"""
        w = self.win
        p = w.group_panel
        idx = int(idx)
        if not (0 <= idx < p.group_list.count()):
            return False, f'索引 {idx} 越界（共 {p.group_list.count()} 组）', {}
        label = p.owner.groups_store.groups[idx].get('label', '?')
        p.group_list.takeItem(idx)
        p.owner.groups_store.remove_at(idx)
        return True, f'已删除 {label}', {'before': idx}

    def set_waves(self, waves_data, atom='A'):
        """整表替换波形（= 人恢复存档的同一条路）。waves_data 为
        waveforms.serialize_waves 同构 list。"""
        w = self.win
        panel = {'A': w.dynamic.wave_panel_A, 'B': w.dynamic.wave_panel_B}.get(atom)
        if panel is None:
            return False, f'atom 需为 A/B，收到 {atom!r}', {}
        try:
            waves = waveforms.restore_waves(json_safe(waves_data))
        except Exception as e:        # noqa: BLE001
            return False, f'波形数据不合法: {e}', {}
        before_n = len(panel.waves)
        panel.set_waves(waves)
        return True, 'ok', {'before': before_n, 'after': len(panel.waves)}

    def restore_snapshot(self, snap):
        """undo/存档恢复：走与"文件→恢复状态"相同的三条 restore 路径。"""
        w = self.win
        w.group_panel.restore_state(snap.get('groups', {}))
        w.field.restore_state(snap.get('field', {}))
        w.dynamic.restore_state(snap.get('dynamic', snap))
