# -*- coding: utf-8 -*-
"""生成 STIRAP / 拉曼 的 GUI 存档 JSON（可直接「文件 → 恢复状态」载入）。

纯 Python 实现（无 Qt/离屏依赖，规避环境 Qt 偶发崩溃）：
  - waves 用 waveforms.serialize_waves 的 schema（纯波形列表，无通道无 TTL）
  - wave_data 用 waveforms 包络直接算
  - 验证 = 波形 → real_field(near_cutoff) → evolve（含 auto 衰减）→ 保真度

运行：D:\\python310\\python.exe -u scripts\\make_stir_rap_states.py
产出：save_state/STIRAP.json（B=300G，近共振截断 50 MHz）
      save_state/拉曼.json（B=300G，近共振截断 350 MHz）
"""
import os
import sys
import json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from arc import Rubidium87

from src import core, evolve, waveforms

cache = core.ArcCache(Rubidium87())
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = ['#e74c3c', '#3498db', '#2ecc71']


def gui_states(groups, B):
    specs = []
    for g in groups:
        n, l, j = g['n'], g['l'], g['j']
        hf = core.get_hyperfine_levels(cache, n, l, j, B)
        if hf is None:
            for m in np.arange(-j, j + 1e-9, 1.0):
                specs.append(dict(n=n, l=l, j=j, kind='mJ', m=float(m),
                                  E_mhz=core._sublevel_energy(cache, n, l, j, m, B)))
        else:
            for F, mF, e_hf, _w, amps in hf:
                specs.append(dict(n=n, l=l, j=j, kind='F', F=F, mF=mF,
                                  E_hf=float(e_hf), E_mhz=cache.E_mhz(n, l, j),
                                  amps=amps))
    return evolve.make_levels(cache, specs)


def env_to_callable(env):
    if env[0] == 'gauss':
        return waveforms.GaussianEnv(env[1], env[2], env[3])
    return waveforms.SquareEnv(env[1], env[2], env[3], env[4])


def waves_dict(pulses):
    """pulses: [(freq_mhz, intensity, pol, env_tuple)] → waveforms 序列化 schema。"""
    out = []
    for k, (freq, I, pol, env) in enumerate(pulses):
        out.append({'name': f'光{k + 1}', 'freq_mhz': freq, 'intensity': I,
                    'pol': pol, 'offset_ns': 0.0, 'env': env, 'drive': None})
    return out


def make_json(name, groups, B, pulses, psi, t_max, npts, near_cutoff,
              decay_mode=0):
    t = np.linspace(0, t_max, npts)
    wave = []
    for k, (freq, I, pol, env) in enumerate(pulses):
        f = env_to_callable(env)
        wave.append({'name': f'光{k + 1}', 't': t.tolist(),
                     'env': np.asarray(f(t)).tolist(), 'freq': float(freq),
                     'intensity': float(I)})
    state = {
        'groups': [dict(g) for g in groups],
        'B_gauss': B,
        'dynamic': {'psi': psi, 't_max': t_max, 'npts': npts,
                    'method': 'auto', 'backend': 'auto', 'topn': 8,
                    'decay_mode': decay_mode, 'near_cutoff': near_cutoff},
        'waves': waves_dict(pulses),
        'decays': [],
        'wave_data': wave,
    }
    path = os.path.join(HERE, 'save_state', name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    return path, state


def verify(state, target_fid, note, use_max=False):
    """解析 JSON 的波形 → real_field(near_cutoff) → evolve（auto 衰减）。"""
    st = gui_states(state['groups'], state['B_gauss'])
    i0 = state['dynamic']['psi']
    t = np.linspace(0, state['dynamic']['t_max'], state['dynamic']['npts'])
    cut = state['dynamic'].get('near_cutoff', 5000.0)
    waves = waveforms.restore_waves(state.get('waves', []))
    fields = waveforms.build_fields(waves, st, cache, near_cutoff_mhz=cut)
    r = evolve.evolve(st, fields, i0, t, method='auto',
                      decays=evolve.auto_decays(cache, st))
    p = r['pops'][:, -1]
    target = float(p.max()) if use_max else float(p[-1])
    ok = target > target_fid
    print(f'  验证: 初态={st[i0]["label"]}  目标(60S){"max" if use_max else "末"}'
          f'={target:.4f} (要求>{target_fid})  {"✅" if ok else "❌"}  [{note}]',
          flush=True)
    return ok


def main():
    groups = [{'n': 5, 'l': 0, 'j': 0.5, 'label': '5S_{1/2}', 'color': COLORS[0]},
              {'n': 6, 'l': 1, 'j': 1.5, 'label': '6P_{3/2}', 'color': COLORS[1]},
              {'n': 60, 'l': 0, 'j': 0.5, 'label': '60S_{1/2}', 'color': COLORS[2]}]

    # ---------------- STIRAP：5S F2m2 →σ+ 6P3/2 F3m3 →σ- 60S mJ+1/2 ----------------
    # B=300G 拉开 Zeeman（F3m3 与流形其余态 ≥168MHz）；Δ=10/cut=50 → 只相干目标对
    B1, DELTA, OM, SIG, SEP = 300.0, 10.0, 50.0, 400.0, 800.0
    t_p, t_s = 2.5 * SIG + SEP, 2.5 * SIG
    TMAX = 2.5 * SIG + SEP + 4 * SIG
    st = gui_states(groups, B1)
    i5 = next(i for i, s in enumerate(st) if s.get('F') == 2 and s.get('mF') == 2)
    i6 = next(i for i, s in enumerate(st) if s.get('F') == 3 and s.get('mF') == 3)
    i60 = len(st) - 1
    d_p = core.dipole_amplitude(cache, evolve._sublevel_row(st[i5]),
                                evolve._sublevel_row(st[i6]), 1)
    nu_p = st[i6]['E_mhz'] - st[i5]['E_mhz']
    I_p = core.intensity_mwcm2_from_rabi(abs(d_p), OM)
    d_s = core.dipole_amplitude(cache, evolve._sublevel_row(st[i6]),
                                evolve._sublevel_row(st[i60]), -1)
    nu_s = st[i60]['E_mhz'] - st[i6]['E_mhz']
    I_s = core.intensity_mwcm2_from_rabi(abs(d_s), OM)
    pulses = [
        (nu_p + DELTA, I_p, 'σ+', ('gauss', t_p, SIG, 1.0)),
        (nu_s - DELTA, I_s, 'σ-', ('gauss', t_s, SIG, 1.0)),
    ]
    p1, s1 = make_json('STIRAP.json', groups, B1, pulses, i5, TMAX,
                       int(TMAX / 4.0) + 1, 50.0)
    ok1 = verify(s1, 0.8, 'STIRAP B=300G 截断50MHz 加自动衰减')

    # ---------------- 拉曼：5S F2m2 ↔ 60S（CW 双光，Δ=300，cut=350） ----------------
    # 流形态在目标失谐之上（Δ+168·k）→ cut 取 Δ 与 Δ+168 之间只留目标对；
    # CW 拉曼受流形差频光移限制在 ~0.4（真实物理，见 stir_rap_selfcheck R3）
    B2, D2, CUT2 = 300.0, 300.0, 350.0
    st2 = gui_states(groups, B2)
    i5b = next(i for i, s in enumerate(st2) if s.get('F') == 2 and s.get('mF') == 2)
    i6b = next(i for i, s in enumerate(st2) if s.get('F') == 3 and s.get('mF') == 3)
    d_p2 = core.dipole_amplitude(cache, evolve._sublevel_row(st2[i5b]),
                                 evolve._sublevel_row(st2[i6b]), 1)
    nu_p2 = st2[i6b]['E_mhz'] - st2[i5b]['E_mhz']
    I_p2 = core.intensity_mwcm2_from_rabi(abs(d_p2), 20.0)
    d_s2 = core.dipole_amplitude(cache, evolve._sublevel_row(st2[i6b]),
                                 evolve._sublevel_row(st2[-1]), -1)
    nu_s2 = st2[-1]['E_mhz'] - st2[i6b]['E_mhz']
    I_s2 = core.intensity_mwcm2_from_rabi(abs(d_s2), 20.0)
    cw = ('square', 0.0, 1.0, 1.0, 3000.0)
    pulses2 = [(nu_p2 + D2, I_p2, 'σ+', cw), (nu_s2 - D2, I_s2, 'σ-', cw)]
    p2, s2 = make_json('拉曼.json', groups, B2, pulses2, i5b, 3000.0, 601,
                       CUT2)
    ok2 = verify(s2, 0.3, 'Λ 拉曼 B=300G 截断350MHz 加自动衰减（光移限制~0.4）',
                 use_max=True)

    print('\n生成完成:', flush=True)
    print('  ', p1, flush=True)
    print('  ', p2, flush=True)
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == '__main__':
    main()
