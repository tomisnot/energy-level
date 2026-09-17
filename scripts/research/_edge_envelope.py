# -*- coding: utf-8 -*-
"""模型无关判据：用解析包络（Hilbert）问"交换能力本身有没有被削掉"。

单条阻尼余弦拟合的残差已达振幅的 ~1%（多通道拍），不足以支撑强结论。
改用**与模型无关**的做法：

    布居 a(t) = C(t) + A(t)·cos(θ(t))  ⇒ 解析信号 |a + i·H(a)| = 上包络 U(t)

U(t) 是"交换能力"的**轨迹级**量，不依赖相位落在哪里。若下降沿真的削掉交换，
U(t) 应当随 τ_f 系统下降；若只是相位/时间重排，U(t) 应当**跨沿宽重合**。

同时给出三个判据：
  J1  各条 U(t) 在公共时间段的**最大差**，与"观测峰值差"对比
  J2  a(t)+b(t) 的上包络是否也是沿宽不变量（守恒量）
  J3  在 t=2000 ns 时刻的瞬时值（终点布居）随沿宽的变化
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = os.path.join(ROOT, 'scripts', 'tmp')

TAGS = ['F/tf=0', 'F/tf=10', 'F/tf=20', 'F/tf=35', 'F/tf=50', 'F/tf=70',
        'F/tf=100', 'E1/hold0@0.6', 'E1/hold25@0.6', 'E1/hold50@0.6',
        'E1/hold100@0.6', 'E3/stair50']


def hilbert_env(t, y, t_lo=340.0):
    """稳态段的上包络（去趋势 + Hilbert 模）。返回 (t, env)。"""
    try:
        from scipy.signal import hilbert
    except ImportError:
        return None, None
    m = t >= t_lo
    tt, yy = t[m], y[m]
    # 去缓慢趋势（多通道拍会使基线漂移）：与稳态均值比较即可
    z = hilbert(yy - yy.mean())
    return tt, np.abs(z)


def main():
    data = {}
    for tag in TAGS:
        p = os.path.join(TMP, '_fall_mech_%s.npz' % tag.replace('/', '_'))
        if not os.path.exists(p):
            continue
        z = np.load(p)
        data[tag] = dict(t=np.asarray(z['t'], float),
                         a=np.asarray(z['aft'], float),
                         b=np.asarray(z['bef'], float),
                         tot=np.asarray(z['aft'], float) + np.asarray(z['bef'], float))

    # 公共时间栅格
    tlo = max(340.0, max(d['t'][0] for d in data.values()))
    thi = min(d['t'][-1] for d in data.values())
    grid = None
    for d in data.values():
        m = (d['t'] >= tlo) & (d['t'] <= thi)
        g = d['t'][m]
        grid = g if grid is None or g.size < grid.size else grid
    tgrid = next(d['t'][(d['t'] >= tlo) & (d['t'] <= thi)] for d in data.values())

    print('  公共稳态段：%.0f – %.0f ns（%d 点）' % (tlo, thi, tgrid.size))

    # ---- J1：a(t) 的上包络 ----
    print('\n  === J1 「换后态」上包络 U_a(t)（Hilbert，模型无关）===')
    envs = {}
    for tag, d in data.items():
        tt, U = hilbert_env(d['t'], d['a'], tlo)
        if U is None:
            print('  ⚠ scipy 不可用，跳过 Hilbert')
            return
        v = np.interp(tgrid, tt, U)
        envs[tag] = v
        print('  %-16s U_a: max %.4f  mean %.4f  min %.4f'
              % (tag, v.max(), v.mean(), v.min()))

    M = np.vstack([envs[t] for t in TAGS if t in envs])
    spread = M.max(axis=0) - M.min(axis=0)
    print('  ---------------------------------------------')
    print('  跨 12 条曲线的 U_a 最大差 = %.4f（均值 %.4f）' % (spread.max(), M.mean()))
    pk_spread = max(d['a'].max() for d in data.values()) - \
        min(d['a'].max() for d in data.values())
    print('  而"观测峰值"的跨曲线差      = %.4f' % pk_spread)
    print('  ⇒ 若 U 差 ≪ 峰值差 ⇒ 峰值差主要是**相位采样**，不是能力被削')

    # ---- J2：a+b 的上包络（守恒量？） ----
    print('\n  === J2 「两通道和」上包络 U_tot(t) ===')
    tots = {}
    for tag, d in data.items():
        tt, U = hilbert_env(d['t'], d['tot'], tlo)
        v = np.interp(tgrid, tt, U)
        tots[tag] = v
        print('  %-16s U_tot: max %.4f  mean %.4f' % (tag, v.max(), v.mean()))
    Mt = np.vstack([tots[t] for t in TAGS if t in tots])
    print('  跨曲线 U_tot 最大差 = %.4f' % (Mt.max(axis=0) - Mt.min(axis=0)).max())

    # ---- J3：终点布居 ----
    print('\n  === J3 终点(t=2000 ns)布居 ===')
    print('  %-16s %9s %9s %9s' % ('tag', 'a(换后)', 'b(换前)', 'a+b'))
    for tag in TAGS:
        if tag not in data:
            continue
        d = data[tag]
        print('  %-16s %9.4f %9.4f %9.4f'
              % (tag, d['a'][-1], d['b'][-1], d['a'][-1] + d['b'][-1]))

    out = os.path.join(TMP, '_edge_envelope.json')
    json.dump({'t': tgrid.tolist(),
               'U_a': {k: v.tolist() for k, v in envs.items()},
               'U_tot': {k: v.tolist() for k, v in tots.items()}},
              io.open(out, 'w', encoding='utf-8'))
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()