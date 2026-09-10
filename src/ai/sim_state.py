# -*- coding: utf-8 -*-
"""SimState：语义状态门面（构建于 param_manager 注册表之上）。

v3 换轨（议题 3）：注册表由 src/pm_setup.build_registry 统一建立
（win.param_reg），本门面只做 AI 工具的语义包装：
  读  to_dict = archive.snapshot(reg)——登记册即存档，无平行清单
  写  set_param = reg.set(source='ai', reason, 变更前快照)——校验/归因/
      灯全在框架；op 由 server 的 journal 桥落账
  恢复 restore_snapshot = archive.restore(reg, snap)（v3 直载；v2/v1 迁移）
结构化操作（能级组增删）走人用路径 + reg.mutate 归因。

所有方法必须在 Qt 主线程执行（经 ai.dispatch.run_on_main 投递）。
"""
import json

from .. import archive
from ..pm_setup import build_registry


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


class SimState:
    """门面：持有 MainWindow + 注册表，不持有状态。"""

    def __init__(self, win):
        self.win = win
        self.reg = getattr(win, 'param_reg', None) or build_registry(win)

    # ---------- 读（登记册即存档） ----------
    def to_dict(self):
        return json_safe(archive.snapshot(self.reg))

    def get_state(self, scope=None):
        """scope ∈ {None,'groups','field','dynamic'}：按前缀裁剪 params。"""
        d = self.to_dict()
        if scope in ('groups', 'field', 'dynamic'):
            pref = scope if scope == 'groups' else scope + '.'
            return {'params': {k: v for k, v in d['params'].items()
                               if k == pref or k.startswith(pref)}}
        return d

    def summary(self):
        p = self.reg.collect()
        groups = p.get('groups.list') or []
        ws = p.get('dynamic.waves_single') or []
        wa = p.get('dynamic.waves_A') or []
        wb = p.get('dynamic.waves_B') or []
        return {
            'n_groups': len(groups),
            'groups': [f"{g['label']}(n={g['n']},l={g['l']},j={g['j']})"
                       for g in groups],
            'B_gauss': p.get('field.B_gauss'),
            'E_vpcm': p.get('field.E_vpcm'),
            'n_lights': len(p.get('field.lights') or []),
            'method': p.get('dynamic.method'),
            't_max_ns': p.get('dynamic.t_max'),
            'n_waves': len(ws) + len(wa) + len(wb),
            'two_body': bool(p.get('dynamic.mode')),
        }

    def schema(self, prefix=''):
        """参数清单（type/unit/desc/range/options/value/writable）→ AI 可读。"""
        return json_safe(self.reg.schema(prefix))

    # ---------- 写（标量：全部经注册表） ----------
    def set_param(self, path, value, reason='', source='ai'):
        """reg.set：校验→写控件（=人敲控件同路）→op(source/reason/快照)。

        返回 {ok, msg, before, after}；op 发射由 reg.on_op 订阅方记账。
        """
        if path not in self.reg._bindings:
            return {'ok': False,
                    'msg': f'未知参数 {path!r}；可用: '
                           + ', '.join(sorted(self.reg.names()))}
        snap = self.to_dict()
        before = self.reg._safe_read(self.reg._bindings[path].read_fn)
        try:
            self.reg.set(path, value, source=source, reason=reason,
                         op_extra={'snapshot': snap})
        except Exception as e:            # noqa: BLE001 - 校验/写入失败
            return {'ok': False, 'msg': str(e)}
        after = self.reg._safe_read(self.reg._bindings[path].read_fn)
        msg = 'ok' if after == value else f'已写入（生效值 {after}）'
        return {'ok': True, 'msg': msg, 'before': before, 'after': after}

    # ---------- 写（结构化/操作：人用路径 + mutate 归因） ----------
    def add_level_group(self, n, l, j, reason='', source='ai'):
        """与人点"添加"逐字节等效：设 spin → panel.add_group（含校验）。

        经 reg.mutate('groups.list')：块内人路生效、块末 diff 归因 op
        （source/reason/变更前快照）→ journal 自动落账 + 列表高亮。
        """
        w = self.win
        p = w.group_panel
        snap = self.to_dict()
        before = len(p.owner.groups_store.groups)
        with self.reg.mutate('groups.list', source=source, reason=reason,
                             op_extra={'snapshot': snap}):
            p.n_spin.setValue(int(n))
            p.l_spin.setValue(int(l))
            p.j_spin.setValue(float(j))
            p.add_group()
        after = len(p.owner.groups_store.groups)
        if after == before:
            from ..core import core as _core
            ok, msg = _core.validate_state(int(n), int(l), float(j))
            return {'ok': False,
                    'msg': msg if not ok else '添加失败（可能重复）'}
        return {'ok': True, 'msg': 'ok', 'before': before, 'after': after}

    def remove_level_group(self, idx, reason='', source='ai'):
        """与人右键删除等效：takeItem + store.remove_at（mutate 归因）。"""
        w = self.win
        p = w.group_panel
        idx = int(idx)
        if not (0 <= idx < p.group_list.count()):
            return {'ok': False,
                    'msg': f'索引 {idx} 越界（共 {p.group_list.count()} 组）'}
        label = p.owner.groups_store.groups[idx].get('label', '?')
        snap = self.to_dict()
        with self.reg.mutate('groups.list', source=source, reason=reason,
                             op_extra={'snapshot': snap}):
            p.group_list.takeItem(idx)
            p.owner.groups_store.remove_at(idx)
        return {'ok': True, 'msg': f'已删除 {label}', 'before': idx}

    def set_waves(self, waves_data, atom='A', reason='', source='ai'):
        """整表替换波形（= 人恢复存档同路；经注册表条目 → op 归因）。

        atom ∈ {single, A, B}（三套数据独立）。
        """
        path = f'dynamic.waves_{atom}'
        if path not in self.reg._bindings:
            return {'ok': False,
                    'msg': f"atom 需为 single/A/B，收到 {atom!r}"}
        before = self.reg.read(path)
        try:
            self.reg.set(path, json_safe(waves_data), source=source,
                         reason=reason,
                         op_extra={'snapshot': self.to_dict()})
        except Exception as e:            # noqa: BLE001
            return {'ok': False, 'msg': f'波形数据不合法: {e}'}
        return {'ok': True, 'msg': 'ok',
                'before': len(before or []), 'after': len(waves_data)}

    # ---------- 快照恢复（undo / 存档载入） ----------
    def restore_snapshot(self, snap):
        """archive.restore：v3 直载 params，v2/v1 自动迁移；
        external 参数走 write_fn 人用路径，标量走 display + 基准回填。"""
        archive.restore(self.reg, snap)
