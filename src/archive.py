# -*- coding: utf-8 -*-
"""archive.py — 状态存档 v3：登记册即存档（议题 3 落地）。

设计（用户定案）：
  · 存档 = {'meta': {'version': 3}, 'params': reg.collect()}——
    参数在不在档里，由"它在不在册上"唯一决定；各组件 serialize_state/
    restore_state 平行清单全部删除。
  · 恢复 = reg.load(params, apply_external=True)（external 参数走 write_fn
    人用路径真正应用；标量走 display 通道 + 基准回填）。
  · 显示类字段（display_mode/readout/readout_weights/focus/fields.psi/
    topn）不入册即不入档（议题 2：与科学计算无关）。
  · 旧档兼容：v2（嵌套 groups/field/dynamic）与 v1（顶层平铺）经
    migrate() 转 v3 params；未识别键不丢（返回清单提示）。
"""
import json

VERSION = 3


def snapshot(reg):
    """当前状态 → v3 存档 dict。"""
    return {'meta': {'version': VERSION}, 'params': reg.collect()}


def restore(reg, state):
    """存档 dict → 应用。返回未识别键列表（不丢）。

    state: v3（含 params）或 v2/v1（经 migrate 转换）。
    """
    params, unknown = _as_params(state)
    reg.load(params, apply_external=True, tolerant=True)
    return unknown


def load_file(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
#  v2/v1 → v3 params 迁移
# ═══════════════════════════════════════════════════════════════

# v2 标量字段 → v3 全名（conv 可选：值转换，如旧方法名归一）
def _method_v2(v):
    """旧档方法名归一（t71 语义）：exact→sambe / ip4→hard / auto→mixed。"""
    m = {'exact': 'sambe', 'ip4': 'hard', 'auto': 'mixed'}.get(v, v)
    return m if m in ('sambe', 'hard', 'mixed') else 'mixed'


_SCALAR_MAP = [
    (('field', 'B_gauss'), 'field.B_gauss', None),
    (('field', 'E_vpcm'), 'field.E_vpcm', None),
    (('field', 'window'), 'field.window', None),
    (('field', 'n_photon'), 'field.n_photon', None),
    (('dynamic', 'dynamic', 't_max'), 'dynamic.t_max', None),
    (('dynamic', 'dynamic', 'npts'), 'dynamic.npts', None),
    (('dynamic', 'dynamic', 'method'), 'dynamic.method', _method_v2),
    (('dynamic', 'dynamic', 'frozen_h_ns'), 'dynamic.frozen_h_ns', None),
    (('dynamic', 'dynamic', 'per_freq_M'), 'dynamic.per_freq_M', None),
    (('dynamic', 'dynamic', 'decay_mode'), 'dynamic.decay_index', None),
    (('dynamic', 'dynamic', 'mode'), 'dynamic.mode', None),
    (('dynamic', 'pair', 'R_um'), 'dynamic.pair.R_um', None),
    (('dynamic', 'pair', 'theta_deg'), 'dynamic.pair.theta_deg', None),
    (('dynamic', 'pair', 'phi_deg'), 'dynamic.pair.phi_deg', None),
    (('dynamic', 'waves'), 'dynamic.waves_single', None),
    (('dynamic', 'waves_A'), 'dynamic.waves_A', None),
    (('dynamic', 'waves_B'), 'dynamic.waves_B', None),
    (('dynamic', 'decays'), 'dynamic.decays', None),
    (('dynamic', 'dynamic', 'psi_key'), 'dynamic.psi_single', None),
    (('dynamic', 'dynamic', 'psi_key_A'), 'dynamic.psi_A', None),
    (('dynamic', 'dynamic', 'psi_key_B'), 'dynamic.psi_B', None),
]

# v2 显示类字段（有意不迁移，静默丢弃）
_DROPPED = {'display_mode', 'readout', 'readout_weights', 'psi',
            'psi_A', 'psi_B', 'focus_index'}


def _dig(state, keys):
    cur = state
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    return cur, True


def _lights_v2_to_v3(rows):
    """v2 lights 行（frequency_ghz）→ light_cfgs 形制（frequency_hz）。"""
    out = []
    for r in (rows or []):
        cfg = {'frequency_hz': float(r.get('frequency_ghz', 0.0)) * 1e9,
               'intensity': float(r.get('intensity', 0.0)),
               'polarization': r.get('polarization', 'π'),
               'scope': r.get('scope', 'both'), 'enabled': True}
        if r.get('elliptical'):
            cfg['elliptical'] = dict(r['elliptical'])
        out.append(cfg)
    return out


def migrate(state):
    """v2/v1 存档 → v3 params（尽力而为，未识别键返回）。"""
    params, unknown = {}, []
    if not isinstance(state, dict):
        return {}, ['(非 dict 存档)']
    ver = (state.get('meta') or {}).get('version', 0)
    if ver >= VERSION and 'params' in state:
        return dict(state['params']), []

    # v1 顶层平铺（groups 裸列表、dynamic/waves/decays 在顶层）→ v2 形状
    src = state
    if not ver:
        groups = state.get('groups')
        src = {'groups': {'groups': groups} if isinstance(groups, list)
               else state.get('groups', {}),
               'field': {k: state[k] for k in ('B_gauss', 'E_vpcm')
                         if k in state},
               'dynamic': {'dynamic': dict(state.get('dynamic') or {}),
                           'waves': state.get('waves'),
                           'decays': state.get('decays')}}

    # 能级组
    g, ok = _dig(src, ('groups', 'groups'))
    if ok:
        params['groups.list'] = g
    # 光场表
    l, ok = _dig(src, ('field', 'lights'))
    if ok:
        params['field.lights'] = _lights_v2_to_v3(l)
    # 标量映射（conv 应用）
    for keys, name, conv in _SCALAR_MAP:
        v, ok = _dig(src, keys)
        if ok:
            params[name] = conv(v) if conv else v
    # 旧档无 mode 字段（v1 / mode 引入前的 v2）→ 默认单原子
    # （与旧 restore_state 的 dyn.get('mode', 0) 同语义）
    params.setdefault('dynamic.mode', 0)
    # v2 双原子档：A 数据在 'waves'（共享去重），waves_A 空 → 回填 A
    if params.get('dynamic.mode') == 1 and not params.get('dynamic.waves_A') \
            and params.get('dynamic.waves_single'):
        params['dynamic.waves_A'] = params['dynamic.waves_single']

    # 未识别键（排除已知容器与显示类）
    for top in state:
        if top not in ('meta', 'groups', 'field', 'dynamic') \
                and top not in _DROPPED \
                and top not in ('B_gauss', 'E_vpcm', 'waves', 'decays'):
            unknown.append(top)
    dyn = (src.get('dynamic') or {}).get('dynamic')
    if isinstance(dyn, dict):
        known = {kk[-1] for kk, _, _ in _SCALAR_MAP}
        for k in dyn:
            if k not in _DROPPED and k not in known:
                unknown.append(f'dynamic.{k}')
    return params, unknown


def _as_params(state):
    if isinstance(state, dict) and 'params' in state:
        return dict(state['params']), []
    return migrate(state)
