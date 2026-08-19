# -*- coding: utf-8 -*-
"""存档序列化（纯数据，无 UI 依赖）——**仅保留旧档兼容助手**。

序列化/反序列化已按模块自治：每个页面组件（ui_group / fields /
ui_dynamic）各自实现 serialize_state / restore_state，主程序只负责
装配调用（见 能级.py _save_state / _load_state）。

本模块的函数是旧集中式序列化遗留：
  - restore_wave_data：读取 v2 之前存档里的 wave_data（波形预览曲线，
    派生显示数据，v2 起不再保存）——供外部工具/脚本转换旧档用；
  - serialize_dynamic：旧接口，已无调用方（保留以防外部脚本引用）。
"""
from ..core import waveforms


def serialize_dynamic(groups, B_gauss, dyn_params, waves, decays, wave_data):
    """动态页全部参数与数据 → dict（含共用能级组、场 B、波形、波形数据）。"""
    return {
        'groups': [dict(g) for g in groups],
        'B_gauss': B_gauss,
        'dynamic': dict(dyn_params),
        'waves': waveforms.serialize_waves(waves),
        'decays': [dict(c) for c in decays],
        'wave_data': [
            {'name': nm,
             't': list(th),
             'env': list(env),
             'freq': float(f),
             'intensity': float(I)}
            for nm, th, env, f, I in wave_data],
    }


def restore_wave_data(wd):
    """存档 wave_data → [(name, t, env, freq[, intensity]), ...]。

    t/env 返回 numpy 数组（供 _draw_top 直接使用）。
    """
    import numpy as np
    out = []
    for d in (wd or []):
        row = (d['name'], np.array(d['t']),
               np.array(d.get('env', d.get('E', 0.0))),
               float(d.get('freq', 0.0)))
        if 'intensity' in d:
            row = row + (float(d['intensity']),)
        out.append(row)
    return out
