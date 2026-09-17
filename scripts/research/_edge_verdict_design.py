# -*- coding: utf-8 -*-
"""判决实验的点级规格：生成 10 条轨迹，写档，提交并行扫描。

⚠ 坑（本次踩到）：`submit_sweep(points=[...])` 里**不能放 `tag` 等元数据键**——
`src/sweep/spec.py:apply_delta` 把 delta 的每个键都当**参数路径**校验，
多余键直接 KeyError（"轴键 'tag' 不在登记册快照里"）。
⇒ 因此本脚本**先写档**（tag ↔ 折线 ↔ 点数索引一一对应），
  扫描提交时只放登记册路径；分析时读同一份档按索引对齐。

设计（全部回到同一末态 I_res，故 A0 的差异只可能来自"沿"）：
  A 族 对称三角 excursion：I_res → depth → I_res（穿越两次）
  B 族 单阶跃平台：跳到 depth 停 dwell，再跳到 I_res（穿越一次，E1 式）
  C 族 前人 F 族定义的单向下降（交叉校验用）
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

IRES, I3 = 0.42, 2.0e8
Q_FS = 295724187.476
TMAX = 2000.0
OUT = os.path.join(ROOT, 'scripts', 'tmp', '_edge_profiles.json')


def fs_wave(breaks, name='频移光'):
    t = [b[0] for b in breaks]
    v = [b[1] for b in breaks]
    return {'name': name, 'freq_mhz': Q_FS, 'intensity': I3, 'pol': 'π',
            'offset_ns': 0.0,
            'env': ['file', t, [float(np.sqrt(max(x, 0.0))) for x in v]]}


def tri(depth, tau, npts):
    """I_res → depth → I_res 对称三角（末态 = I_res）"""
    br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0)]      # 上升沿：先建立平台
    # 注意：为了让"末态同 = I_res"，excursion 之后停回 0.42
    br = [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (254.999, 1.0),
          (255.0, IRES)]
    for k in range(1, npts + 1):
        br.append((255.0 + tau * k / npts,
                   IRES + (depth - IRES) * k / npts))
    for k in range(1, npts + 1):
        br.append((255.0 + tau + tau * k / npts,
                   depth - (depth - IRES) * k / npts))
    br.append((TMAX, IRES))
    return br


def hold(depth, dwell):
    return [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (254.999, 1.0),
            (255.0, depth), (255.0 + dwell, depth),
            (255.0 + dwell + 0.001, IRES), (TMAX, IRES)]


def down(tau):
    return [(0.0, 0.0), (119.999, 0.0), (120.0, 1.0), (255.0, 1.0),
            (255.0 + tau, IRES), (TMAX, IRES)]


def profiles():
    """→ [(tag, breaks, meta)]，顺序 = 扫描点顺序"""
    out = []
    out.append(('A/tri50_d0.71', tri(0.71, 50.0, 50),
                dict(family='A', depth=0.71, tau=50.0, rate=2 * (0.71 - IRES) / 50)))
    out.append(('A/tri50_d1.00', tri(1.00, 50.0, 50),
                dict(family='A', depth=1.00, tau=50.0, rate=2 * (1.00 - IRES) / 50)))
    out.append(('A/tri25_d1.00', tri(1.00, 25.0, 25),
                dict(family='A', depth=1.00, tau=25.0, rate=2 * (1.00 - IRES) / 25)))
    out.append(('A/tri100_d1.00', tri(1.00, 100.0, 100),
                dict(family='A', depth=1.00, tau=100.0, rate=2 * (1.00 - IRES) / 100)))
    for dw in (10.0, 25.0, 50.0):
        out.append(('B/hold%g@0.71' % dw, hold(0.71, dw),
                    dict(family='B', depth=0.71, tau=dw, rate=0.0, dwell=dw)))
    out.append(('C/step', down(0.0), dict(family='C', depth=1.0, tau=0.0)))
    out.append(('C/down50', down(50.0), dict(family='C', depth=1.0, tau=50.0)))
    return out


def main():
    from src.core.waveforms import FileEnv, GaussianEnv, Wave
    prof = profiles()

    # 微波两支 π（与存档一致）
    mw1 = Wave('p70', 17236.743, GaussianEnv(70.0, 20.0, 1.0), pol='σ-',
               intensity_mwcm2=5.5e-5)
    mw2 = Wave('p200', 17236.743, GaussianEnv(200.0, 20.0, 1.0), pol='σ-',
               intensity_mwcm2=5.5e-5)
    from src.core.waveforms import serialize_waves
    mwser = serialize_waves([mw1, mw2])

    pts, tags = [], []
    for tag, br, meta in prof:
        w = fs_wave(br)
        pts.append({'dynamic.t_max': TMAX, 'dynamic.npts': 1001,
                    'dynamic.pair.R_um': 14.1, 'dynamic.pair.theta_deg': 31.1,
                    'dynamic.method': 'mixed',
                    'dynamic.waves_A': mwser + [w]})
        tags.append(tag)

    with io.open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'tags': tags,
                   'profiles': [{'tag': tg, 'breaks': b, 'meta': m}
                                for tg, b, m in prof]},
                  f, ensure_ascii=False, indent=1)
    print('  写档 → %s（%d 条轨迹）' % (OUT, len(prof)))
    for tag, br, meta in prof:
        print('    %-16s %s' % (tag, meta))

    # 交付给 MCP 提交用（JSON 一行，可直接喂给 submit_sweep）
    with io.open(OUT.replace('.json', '_points.json'), 'w',
                 encoding='utf-8') as f:
        json.dump(pts, f, ensure_ascii=False)
    print('\n  扫描点 → %s' % OUT.replace('.json', '_points.json'))
    print('  由 MCP 侧提交（本脚本**不**自己起 runner：ARC sqlite 并发会互踩，'
          '研究文档 §5 教训 1）')

    if '--submit' in sys.argv:
        r = submit_sweep(points=pts, workers=7, record='final', top_n=6,
                         states=['(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)',
                                 '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'],
                         tag='edge-phase-verdict',
                         reason='判决：下降沿是相位还是幅度；穿越速率是否主控')
        print('\n  提交 →', r.get('sweep_id'), '| ok', r.get('ok'))


if __name__ == '__main__':
    main()
