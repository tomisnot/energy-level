# -*- coding: utf-8 -*-
"""存档序列化（纯数据，无 UI 依赖）。

serialize 从"页面状态快照"（dict）生成 JSON 可序列化 dict；
restore 把 dict 写回"页面状态快照"。UI 控件读写由调用方（DynamicPanel）
负责，本模块只做数据组装/解析——解耦存档格式与 Qt 控件。
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
