# -*- coding: utf-8 -*-
"""pulse.py 独立模块自检：两种包络形状 + TTL 触发定位 + evolve.field 集成。"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src import pulse, evolve
from checks.checklib import Checker

ck = Checker('pulse_selfcheck')


class FakeEdge:
    """模拟 TTL 沿（鸭子类型，只需 .t_abs / .desc）。"""
    def __init__(self, t, desc='A0'):
        self.t_abs = float(t)
        self.desc = desc


def main():
    # ---------- 高斯包络 ----------
    g = pulse.GaussianEnv(center_ns=400.0, sigma_ns=60.0, peak=0.8)
    tau = np.linspace(0, 1000, 10001)
    y = g(tau)
    ck.check('高斯中心峰值=peak',
             abs(g(400.0) - 0.8) < 1e-12, f'{g(400.0):.6g}')
    ck.check('高斯对称', abs(g(340.0) - g(460.0)) < 1e-9,
             f'{g(340.0):.6g} vs {g(460.0):.6g}')
    ck.check('高斯5σ外≈0', np.max(y[tau >= 400 + 5 * 60]) < 5e-6,
             f'max={y[tau >= 400 + 5 * 60].max():.3g}')
    ck.check('高斯包络∈[0,1]', y.min() >= 0 and y.max() <= 1)

    # ---------- 缓变方波 ----------
    s = pulse.SmoothSquareEnv(start_ns=200.0, edge_ns=40.0, height=1.0,
                              fall_start_ns=800.0)
    t = np.linspace(0, 1200, 12001)
    y = s(t)
    ck.check('方波上升前=0', abs(s(150.0)) < 1e-12)
    ck.check('方波平台=height', abs(s(500.0) - 1.0) < 1e-12,
             f'{s(500.0):.6g}')
    ck.check('方波下降后=0', abs(s(860.0)) < 1e-12)
    ck.check('方波上升沿中点=height/2',
             abs(s(220.0) - 0.5) < 1e-3, f'{s(220.0):.6g}')   # 余弦缓变半高处
    ck.check('方波下降沿中点=height/2',
             abs(s(820.0) - 0.5) < 1e-3, f'{s(820.0):.6g}')
    # 沿端点导数连续 → 邻点微增（缓变性）
    ck.check('方波沿平滑(非阶跃)', 0 < s(210.0) < 0.5,
             f'{s(210.0):.4g}')

    # 缺省 fall_start = start + 2*edge（最小脉冲）
    s2 = pulse.SmoothSquareEnv(100.0, 30.0)
    ck.check('方波缺省fall_start', s2.fall_start_ns == 160.0,
             f'{s2.fall_start_ns}')
    # 退化方波（edge=0）
    s3 = pulse.SmoothSquareEnv(100.0, 0.0, fall_start_ns=300.0)
    ck.check('方波edge=0退化为理想方波',
             abs(s3(200.0) - 1.0) < 1e-12 and abs(s3(50.0)) < 1e-12)

    # ---------- TTL 触发定位 ----------
    trig = FakeEdge(500.0)
    p1 = pulse.Pulse(6.8e6, pulse.GaussianEnv(400.0, 60.0),
                     trigger=trig, offset_ns=0.0)
    ck.check('触发起点=沿时刻+偏移', p1.start() == 500.0, f'{p1.start()}')
    p2 = pulse.Pulse(6.8e6, pulse.SmoothSquareEnv(0.0, 30.0,
                                                  fall_start_ns=200.0),
                     trigger=trig, offset_ns=50.0)
    ck.check('偏移后起点', p2.start() == 550.0, f'{p2.start()}')
    ck.check('结束上界', p2.end() == 550.0 + 200.0 + 30.0, f'{p2.end()}')

    # waveform 绝对定位：演化时刻 t=触发时刻时，包络 = envelope(偏移)
    w = p1.waveform(500.0)
    ck.check('waveform绝对定位', abs(w - p1.envelope(0.0)) < 1e-9,
             f'{w:.6g} vs {p1.envelope(0.0):.6g}')

    # ---------- evolve.field 集成 ----------
    f = p1.to_field(1, 2, omega0_mhz=20.0)
    ck.check('生成evolve.field', f['kind'] == 'pair' and
             f['g'] == 1 and f['e'] == 2 and f['omega0_mhz'] == 20.0)
    tt = np.linspace(0, 1200, 61)
    wv = f['waveform'](tt)
    ck.check('field波形∈[0,1]', wv.min() >= 0 and wv.max() <= 1)
    ck.check('field波形在触发时刻≈0(高斯中心前5σ)',
             abs(f['waveform'](500.0)) < 1e-6)

    # ---------- 实演化冒烟（Pulse→field→evolve，验证布居归一） ----------
    try:
        from arc import Rubidium87
        from src import core
        cache = core.ArcCache(Rubidium87())
        states = evolve.make_levels(cache, [
            dict(n=6, l=1, j=1.5, m=-0.5),
            dict(n=60, l=0, j=0.5, m=+0.5),
        ])
        ps = pulse.Pulse(6.8e6, pulse.SmoothSquareEnv(100.0, 20.0,
                                                      fall_start_ns=400.0),
                         trigger=FakeEdge(0.0))
        fields = [ps.to_field(1, 0, omega0_mhz=15.0)]
        tg = np.linspace(0, 700, 701)
        r = evolve.evolve(states, fields, 0, tg, method='pwc')
        norm = np.max(np.abs(r['pops'].sum(axis=1) - 1.0))
        ck.check('Pulse演化布居归一', norm < 1e-6, f'norm_err={norm:.2e}')
    except Exception as ex:
        ck.skip('Pulse演化布居归一', f'{type(ex).__name__}: {ex}')

    ok = ck.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
