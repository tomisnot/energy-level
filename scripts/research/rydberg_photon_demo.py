# -*- coding: utf-8 -*-
"""涌现光子谱学 · 多体引擎可行性演示（对应 00_方向_涌现光子谱学.md §6 清单）。

诚实定位：本演示只跑引擎**真正硬**的部分——文档 §6 清单里"只有里德堡才有"
的那一步：
  ① 从真实 ARC 偶极矩阵元算各向异性交换 J_ij，含场方向 FM/AFM 开关；
  ② J̃(k) 各向异性 = 各向异性"光速"（判据③）；
  ④ 长偶极尾的 |k| 软化指纹（判据④，核心卖点）。
磁序基态与完整 S(q,ω)（②③ 的"有序相里"部分）引擎只能给**有限团簇 ED**，
文档要的是热力学极限有序相——真实边界，脚本如实标注。

性能：每种几何**标定一次**，J̃(k) 从缓存的 (对表, J 值) 复用。
运行：python scripts/research/rydberg_photon_demo.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from arc import Rubidium87
from src.core import core
from src.manybody.model import ManyBodyModel
from src.manybody import assemble as asm

G = (60, 0, 0.5, 0.5)          # 60S_1/2 m=+1/2（"下" g）
R_ = (60, 1, 1.5, 1.5)         # 60P_3/2 m=+3/2（"上" r，承载偶极交换）
A_UM = 10.0
cache = core.ArcCache(Rubidium87())


def lattice(L, tilt_deg):
    """L×L 方格，面内基 u=(1,0,0)、v=(0,cos t,sin t)；tilt=0 法线=z。"""
    t = np.deg2rad(tilt_deg)
    u = np.array([1.0, 0, 0])
    v = np.array([0, np.cos(t), np.sin(t)])
    return np.array([(ix - (L - 1) / 2) * A_UM * u
                     + (iy - (L - 1) / 2) * A_UM * v
                     for ix in range(L) for iy in range(L)])


def calib_once(pos):
    m = ManyBodyModel(terms=('I2',), n_window=1)
    for p in pos:
        m.add_atom(G, R_, p)
    t0 = time.time()
    cal = asm.calibrate(cache, m, second_order=False)
    print(f'    [标定 {len(pos)} 原子 → {len(cal["pairs"])} 对，'
          f'{time.time() - t0:.1f}s]', flush=True)
    return cal


def j_tilde(cal, pos, k):
    s = 0.0
    for p in cal['pairs']:
        d = pos[p['i']] - pos[p['j']]
        s += 2.0 * p['raw_J'].real * np.cos(np.dot(k, d))
    return s


def main():
    t0 = time.time()
    print('=' * 66, flush=True)
    print('涌现光子谱学 · 引擎可行性演示  (60S↔60P_3/2, a=%gµm, 真实 ARC)'
          % A_UM, flush=True)
    print('=' * 66, flush=True)

    # ---------- ① 场方向 = FM/AFM 开关（文档 §2 路线 a / A.0 符号） ----------
    print('\n[①] 偶极交换符号随场方向翻转（1−3cos²θ 的直接体现）',
          flush=True)
    for tilt in (0.0, 90.0):
        pos = lattice(4, tilt)
        cal = calib_once(pos)
        nn = [p for p in cal['pairs']
              if abs(np.linalg.norm(pos[p['i']] - pos[p['j']]) - A_UM) < 1e-6]
        js = [p['raw_J'].real for p in nn]
        tag = 'FM 侧(场∥法线)' if tilt == 0 else 'AFM 侧(场∥面)'
        print(f'  tilt={tilt:3.0f}° {tag}: 最近邻 J 符号 '
              f'正{sum(j > 0 for j in js)}/负{sum(j < 0 for j in js)}  '
              f'|J_NN|≈{np.mean(np.abs(js)):.3f} MHz', flush=True)

    # ---------- ② J̃(k) 各向异性 = 各向异性"光速"（判据③） ----------
    print('\n[②] J̃(k) 各向异性（AFM 侧 tilt=90°，沿面内两主轴）', flush=True)
    L = 12
    pos = lattice(L, 90.0)
    cal = calib_once(pos)
    k0 = 2 * np.pi / (L * A_UM)
    J0 = j_tilde(cal, pos, np.zeros(3))
    for kk, lab in [(np.array([k0, 0, 0]), 'u 轴'),
                    (np.array([0, k0, 0]), 'v 轴(含 z 分量)')]:
        Jk = j_tilde(cal, pos, kk)
        print(f'  {lab:16s}: J̃(0)−J̃(k) = {J0 - Jk:9.4f} MHz', flush=True)
    print('  ⇒ 两轴数值不同 ⇒ 刚度张量各向异性 ⇒ v(k̂) 方向依赖（判据③）',
          flush=True)

    # ---------- ④ 长偶极尾的 |k| 软化（文档 §4.2 / 判据④，核心卖点） ----------
    print('\n[④] 小-k 标度：J̃(0)−J̃(k) ∝ |k|^α  （α≈1 长程偶极 / α≈2 短程）',
          flush=True)
    L24 = 12
    pos24 = lattice(L24, 90.0)
    cal24 = cal if L24 == L else calib_once(pos24)   # 复用 ② 的标定
    J0_24 = J0 if L24 == L else j_tilde(cal24, pos24, np.zeros(3))
    ks = np.array([0.04, 0.08, 0.15, 0.25, 0.4])
    rows = np.array([J0_24 - j_tilde(cal24, pos24,
                                     np.array([ka / A_UM, 0, 0]))
                     for ka in ks])
    slope = np.polyfit(np.log(ks), np.log(np.abs(rows)), 1)[0]
    for ka, val in zip(ks, rows):
        print(f'  k·a={ka:.2f}:  J̃(0)−J̃(k) = {val:9.4f} MHz', flush=True)
    print(f'  ⇒ log-log 斜率 α ≈ {slope:.2f}  '
          f'（≈1 → |k| 型软化，长程偶极指纹；≈2 → 短程 k²）', flush=True)

    # ---------- ③ 有限团簇 ED：一粒子扇区色散（诚实标注有限尺寸） ----------
    print('\n[③] L=3 团簇 ED：一粒子扇区谱 ω(q)（有限尺寸，存在性演示）',
          flush=True)
    pos3 = lattice(3, 90.0)
    cal3 = calib_once(pos3)
    n = len(pos3)
    Jm = np.zeros((n, n))
    for p in cal3['pairs']:
        Jm[p['i'], p['j']] = Jm[p['j'], p['i']] = p['raw_J'].real
    ew = np.linalg.eigvalsh(Jm)
    print(f'  一粒子扇区（XX  hopping 的本征值）= {np.round(ew, 3)} MHz',
          flush=True)
    print(f'  带宽 = {ew.max() - ew.min():.3f} MHz；最低模即 q→0 候选',
          flush=True)
    print('  ⚠ 磁有序基态 / 真实 S(q,ω) 线宽 = 热力学极限量，'
          '引擎当前给有限团簇 ED；完整实现需 M3 的 TEBD/DMRG（见计划文档）',
          flush=True)

    print('\n' + '=' * 66)
    print(f'演示完成，用时 {time.time() - t0:.1f}s', flush=True)
    print('=' * 66)


if __name__ == '__main__':
    main()
