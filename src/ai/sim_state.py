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



# ---------- 二期 A：语义状态面（P1/P5：AI 只操作语义对象） ----------

def groups_delta(cur, want):
    """纯函数：cur/want = [(n,l,j),…] → (to_add, to_remove)。

    声明式组操作的判定核心（可 headless 自检）。按 (n,l,j) 比较，
    顺序无关、重复无关；返回的 to_add 保持 want 的顺序。
    """
    ck = {(int(n), int(l), round(float(j), 2)) for (n, l, j) in cur}
    wk = [(int(n), int(l), round(float(j), 2)) for (n, l, j) in want]
    wset = set(wk)
    to_add, seen = [], set()
    for k in wk:
        if k not in ck and k not in seen:
            to_add.append(k)
            seen.add(k)
    # 排序保证确定性（集合迭代顺序不定会让 journal 与自检抖动）
    to_remove = sorted(k for k in ck if k not in wset)
    return to_add, to_remove


def _key_builder():
    """找既有的身份键构造器（真值源），找不到则由调用方回退手拼。"""
    import importlib
    for mod, name in (('..core.atomic_levels', 'state_key'),
                      ('..ui.ui_dynamic', 'state_key'),
                      ('..core.project', 'state_key')):
        try:
            f = getattr(importlib.import_module(mod, __package__), name, None)
            if callable(f):
                return f
        except Exception:                          # noqa: BLE001
            pass
    return None


def _manual_key(st):
    return ['mJ', int(st['n']), int(st['l']), float(st['j']),
            float(st['m']), float(st.get('E_mhz', st.get('E', 0.0)))]



# ---------- 二期 B：事前成本（作业模型的一半） ----------

_PAIR_CAL = {'N': 36, 't_ns': 300.0, 'sec': 120.0}   # 2026-09-12 实测标定


def estimate_pair_cost(basis_n, t_max_ns, npts=201):
    """两体运行耗时的**粗估**（秒）——按实测标定外推。

    实测（2026-09-12）：N=36 态 → 1296 两体态、t_max=300 ns 时单次 ≈120 s，
    nfev≈4.3e5、dt≈0.002 ns（被 GHz 载波钉死）。两体 RHS 成本 ∝ dim²，
    dim=N² → 成本 ∝ N⁴；对 t_max 近似线性。**只用于判断"要不要异步"**，
    不是精度承诺。
    """
    N = max(int(basis_n), 1)
    s = _PAIR_CAL['sec'] * (N / _PAIR_CAL['N']) ** 4
    s *= max(float(t_max_ns), 1e-9) / _PAIR_CAL['t_ns']
    s *= max(int(npts), 2) / 201.0
    return float(s)



# ---------- 叠加态初态（AI 优先） ----------

def build_sup_spec(components):
    """[{state:[n,l,j,m], amp:[re,im] | mag/phase_deg}, …] → (spec, info)。

    纯函数（headless 可测）：规格化幅度、算范数与占据谱，不碰窗口/注册表。
    spec = ['sup', [[n,l,j,m,re,im], …]]（写进 dynamic.psi_* 的存储形态）。
    """
    out, pops, norm2 = [], [], 0.0
    for c in (components or []):
        st = c.get('state') if isinstance(c, dict) else c
        if not (isinstance(st, (list, tuple)) and len(st) == 4):
            raise ValueError('每个分量的 state 必须是 [n,l,j,m]：%r' % (st,))
        n, l, j, m = (int(st[0]), int(st[1]), float(st[2]), float(st[3]))
        if isinstance(c, dict) and 'amp' in c:
            amp = complex(c['amp'][0], c['amp'][1])
        elif isinstance(c, dict) and 'mag' in c:
            import math
            ph = math.radians(float(c.get('phase_deg') or 0.0))
            amp = complex(float(c['mag']) * math.cos(ph),
                          float(c['mag']) * math.sin(ph))
        else:
            amp = 1.0 + 0j
        out.append([n, l, j, m, amp.real, amp.imag])
        pops.append([n, l, j, m, abs(amp) ** 2])
        norm2 += abs(amp) ** 2
    if not out:
        raise ValueError('叠加态分量不能为空')
    if norm2 <= 0:
        raise ValueError('叠加态各分量幅度全为零')
    return ['sup', out], {'n_components': len(out),
                          'norm_before': float(norm2 ** 0.5),
                          'populations': [[p[0], p[1], p[2], p[3],
                                           round(p[4] / norm2, 6)]
                                          for p in pops]}


class SimState:
    """门面：持有 MainWindow + 注册表，不持有状态。"""

    def __init__(self, win):
        self.win = win
        self.reg = getattr(win, 'param_reg', None) or build_registry(win)

    # ---------- 二期 A：语义状态面 ----------
    def _states_now(self):
        """基组当前的全部子能级（真值源 = 面板的 _current_states）。"""
        dyn = self.win.dynamic
        try:
            dyn.refresh_groups()
        except Exception:                          # noqa: BLE001 - 无面板时忽略
            pass
        st = getattr(dyn, '_states', None)
        if not st:
            st = dyn._current_states()
            dyn._states = list(st)
        return list(st)

    def _key_of(self, st):
        f = _key_builder()
        return list(f(st)) if f else _manual_key(st)

    def list_levels(self, scope='basis'):
        """基组内子能级清单：index / (n,l,j,m) / **合法身份键**。

        这是 AI 设初态的唯一正确入口（此前 options 为 null，AI 只能猜能量，
        实测猜错会静默落到别的能级）。
        """
        out = []
        for i, st in enumerate(self._states_now()):
            out.append({'index': i, 'n': int(st['n']), 'l': int(st['l']),
                        'j': float(st['j']), 'm': float(st['m']),
                        'key': self._key_of(st),
                        'label': st.get('label') or st.get('name')})
        return {'ok': True, 'n': len(out), 'levels': out}

    def set_initial_state(self, atom, n, l, j, m, reason=''):
        """按**语义身份**设初态；不在基组内就报错（不再静默选错能级）。"""
        want = (int(n), int(l), round(float(j), 2), round(float(m), 2))
        hit = None
        for st in self._states_now():
            if (int(st['n']), int(st['l']), round(float(st['j']), 2),
                    round(float(st['m']), 2)) == want:
                hit = st
                break
        if hit is None:
            return {'ok': False, 'msg': '该能级不在当前基组：n=%d,l=%d,j=%g,m=%g'
                                        '（先 set_groups 或 add_level_group）'
                    % want,
                    'hint': 'list_levels() 可看全部合法项'}
        path = {'single': 'dynamic.psi_single', 'A': 'dynamic.psi_A',
                'B': 'dynamic.psi_B'}.get(atom)
        if path is None:
            return {'ok': False, 'msg': "atom 必须为 single/A/B，收到 %r" % atom}
        key = self._key_of(hit)
        self.set_param(path, key, reason=reason or '设初态 %s' % (want,))
        return {'ok': True, 'atom': atom, 'path': path, 'key': key}

    def set_groups(self, groups, reason=''):
        """**声明式**设定能级组：让基组恰好包含给定 (n,l,j) 集合。

        原子 + 幂等 + 回增量（P1/P5：语义操作，不是按索引点删）。
        本会话两次翻车都源于"按标签前缀判重"（删错组 / 漏加组）。
        """
        cur = [(g['n'], g['l'], g['j'])
               for g in (self.reg.read('groups.list') or [])]
        to_add, to_remove = groups_delta(cur, groups)
        removed, added = [], []
        for (n, l, j) in to_remove:                # 从后往前删，避免索引漂移
            lst = self.reg.read('groups.list') or []
            for k in range(len(lst) - 1, -1, -1):
                g = lst[k]
                if (int(g['n']), int(g['l']), round(float(g['j']), 2)) == \
                        (n, l, round(float(j), 2)):
                    self.remove_level_group(k, reason=reason or 'set_groups')
                    removed.append([n, l, j])
                    break
        for (n, l, j) in to_add:
            r = self.add_level_group(n, l, j, reason=reason or 'set_groups')
            if isinstance(r, dict) and r.get('ok') is False:
                return {'ok': False, 'msg': '加组失败 n=%d,l=%d,j=%g：%s'
                        % (n, l, j, r.get('msg')), 'added': added,
                        'removed': removed}
            added.append([n, l, j])
        kept = [[n, l, j] for (n, l, j) in cur
                if (n, l, round(float(j), 2)) not in
                {(a, b, round(float(c), 2)) for (a, b, c) in to_remove}]
        # A8：组变了，初态可能失效——显式报告，不许静默
        warn = []
        for atom, path in (('single', 'dynamic.psi_single'),
                           ('A', 'dynamic.psi_A'), ('B', 'dynamic.psi_B')):
            try:
                v = self.reg.read(path)
            except Exception:                      # noqa: BLE001
                continue
            if not (isinstance(v, (list, tuple)) and len(v) >= 5):
                continue
            key = (int(v[1]), int(v[2]), round(float(v[3]), 2),
                   round(float(v[4]), 2))
            ok = any((int(s['n']), int(s['l']), round(float(s['j']), 2),
                      round(float(s['m']), 2)) == key
                     for s in self._states_now())
            if not ok:
                warn.append('%s 的初态 %s 已不在新基组内' % (atom, list(v)[:5]))
        return {'ok': True, 'added': added, 'removed': removed, 'kept': kept,
                'psi_warnings': warn}

    def set_superposition(self, atom, components, reason=''):
        """设**叠加态初态**（AI 优先；GUI 后议）。

        components: [{'state': [n,l,j,m], 'amp': [re,im]}
                     | {'state': [n,l,j,m], 'mag': 0.7, 'phase_deg': 45}, …]
        · 每个分量必须在**当前基组**内，否则报错（不许静默丢项）
        · 自动归一（返回归一前范数）；存储形态 ['sup', …] 写进 dynamic.psi_*
        """
        path = {'single': 'dynamic.psi_single', 'A': 'dynamic.psi_A',
                'B': 'dynamic.psi_B'}.get(atom)
        if path is None:
            return {'ok': False, 'msg': "atom 必须为 single/A/B，收到 %r" % atom}
        try:
            spec, info = build_sup_spec(components)
        except ValueError as e:
            return {'ok': False, 'msg': str(e)}
        levels = {(x['n'], x['l'], round(x['j'], 2), round(x['m'], 2))
                  for x in (self.list_levels().get('levels') or [])}
        missing = []
        for (n, l, j, m, _re, _im) in spec[1]:
            if (int(n), int(l), round(float(j), 2), round(float(m), 2)) \
                    not in levels:
                missing.append([n, l, j, m])
        if missing:
            return {'ok': False, 'missing': missing, 'n_levels': len(levels),
                    'msg': '这些分量不在当前基组：%s（先 set_groups；'
                           'list_levels() 看合法项）' % (missing,)}
        self.set_param(path, spec, reason=reason or '设叠加态初态')
        return {'ok': True, 'atom': atom, 'path': path,
                'spec': spec, 'info': info}

    def estimate_cost(self):
        """事前成本：要不要走异步、要不要开短窗口，先看这个数。

        单体：走引擎 dry_run 的真值报告（`_cost_report`：保留对数/最远失谐/
        预计步数）。两体：按实测标定的 N⁴·t 粗估（`estimate_pair_cost`）。
        """
        p = self.reg.collect()
        mode = int(p.get('dynamic.mode') or 0)
        t_max = float(p.get('dynamic.t_max') or 0.0)
        npts = int(p.get('dynamic.npts') or 0)
        try:
            levels = self._states_now()
        except Exception:                          # noqa: BLE001
            levels = []
        N = len(levels)
        out = {'ok': True, 'mode': mode, 'basis_n': N, 't_max_ns': t_max,
               'npts': npts, 'method': p.get('dynamic.method'),
               'n_waves': sum(len(p.get('dynamic.waves_' + a) or [])
                              for a in ('single', 'A', 'B'))}
        if mode == 1:
            sec = estimate_pair_cost(N, t_max, npts)
            out.update({'dim': N * N if N else None,
                        'est_seconds': round(sec, 1),
                        'est_minutes': round(sec / 60.0, 1),
                        'advice': ('建议 run_python_async' if sec > 120
                                   else '可同步等待'),
                        'note': '两体：按实测标定 N⁴·t 粗估（36 态/300 ns≈120 s）'})
        else:
            out.update({'dim': N,
                        'note': '单体的引擎真值报告：见 evolve_mixed(dry_run=True) '
                                '的 cost（配对/最远失谐/预计步数）'})
        return out

    def get_config(self):
        """规范化运行配置（**唯一真值源**快照；P5）。

        与 summary()/get_state() 的关系：后两者是"渲染视图"，本方法是
        AI/脚本据以自检配置的规范形态。
        """
        p = self.reg.collect()
        cfg = {'groups': [[int(g['n']), int(g['l']), float(g['j'])]
                          for g in (p.get('groups.list') or [])]}
        for k in ('mode', 'method', 't_max', 'npts'):
            cfg[k] = p.get('dynamic.' + k)
        for k in ('R_um', 'theta_deg', 'phi_deg'):
            cfg[k] = p.get('dynamic.pair.' + k)
        cfg['B_gauss'] = p.get('field.B_gauss')
        cfg['E_vpcm'] = p.get('field.E_vpcm')
        cfg['psi'] = {a: p.get('dynamic.psi_' + a)
                      for a in ('single', 'A', 'B')}
        cfg['n_waves'] = {a: len(p.get('dynamic.waves_' + a) or [])
                          for a in ('single', 'A', 'B')}
        cfg['lights'] = len(p.get('field.lights') or [])
        return {'ok': True, 'config': json_safe(cfg)}

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
