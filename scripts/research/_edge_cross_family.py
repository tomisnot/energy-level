# -*- coding: utf-8 -*-
"""跨族一致性检验：a/s 与幅度损失，是否只是净相位 Δφ 的函数？

两族（前人 F 族单向 / 本轮 A 族对称往返 / B 族停留）并排比较。

关键检验：把"损失"对**同一个相位尺度** Δφ 作图。
  · 若三族落在同一曲线 ⇒ 相位是唯一控制量
  · 若不同族各成一条曲线 ⇒ 除相位外还有"轨迹结构"的独立贡献

前人 F 族的 Δφ 由其锁相相位给出（_edge_lockin.json 的 phi0，
以 F/tf=0 为参照）；本轮各族的 Δφ 由 _edge_loss_vs_phase.json 给出。
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

# 统一相位基准：以各自族的"理想阶跃"为 0
LEG = os.path.join(TMP, '_edge_lockin.json')       # 前人 12 条
NEW = os.path.join(TMP, '_edge_loss_vs_phase.json')  # 本轮 9 条
LEAK = os.path.join(TMP, '_edge_leak_test.json')


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def main():
    # ---- 前人 F 族 ----
    L = json.load(io.open(LEG, encoding='utf-8'))['rows']
    refF = next(r for r in L if r['tag'] == 'F/tf=0')
    F = []
    for r in L:
        if not r['tag'].startswith('F/'):
            continue
        F.append(dict(tag=r['tag'], fam='前人 F 族（单向）',
                      dphi=wrap(r['phi0'] - refF['phi0']),
                      loss=(refF['A0'] - r['A0']) / refF['A0']))
    # ---- 本轮三族 ----
    N = json.load(io.open(NEW, encoding='utf-8'))['rows']
    leak = {r['tag']: r for r in
            json.load(io.open(LEAK, encoding='utf-8'))['rows']}
    N.sort(key=lambda r: abs(r['dphi']))
    refN = N[0]
    famname = {'A': '本轮 A 族（对称往返）', 'B': '本轮 B 族（停留式）',
               'C': '本轮 C 族（单向）'}
    NN = [dict(tag=r['tag'], fam=famname[r['tag'][0]], dphi=r['dphi'] - refN['dphi'],
               loss=r['loss_rel'] - refN['loss_rel'],
               frac=(leak[r['tag']]['pk'] /
                     (leak[r['tag']]['pk'] + leak[r['tag']]['bef_at_pk'])))
          for r in N]

    print('  === 前人 F 族（单向下降，τ_f 0→100 ns）===')
    print('  %-10s %10s %10s' % ('tag', 'Δφ(rad)', '相对损失'))
    for r in sorted(F, key=lambda x: abs(x['dphi'])):
        print('  %-10s %10.3f %+9.1f%%' % (r['tag'], r['dphi'], 100 * r['loss']))

    print('\n  === 本轮三族 ===')
    print('  %-20s %10s %10s %10s' % ('tag', 'Δφ(rad)', '相对损失', 'a/s'))
    for r in sorted(NN, key=lambda x: abs(x['dphi'])):
        print('  %-20s %10.3f %+9.1f%% %10.4f'
              % (r['tag'], r['dphi'], 100 * r['loss'], r['frac']))

    print('\n  === 跨族检验：同为 |Δφ| ≈ 0.6–0.9 rad 的点，损失是否相同？ ===')
    band = [r for r in F if 0.5 < abs(r['dphi']) < 0.9] + \
           [r for r in NN if 0.5 < abs(r['dphi']) < 0.9]
    for r in sorted(band, key=lambda x: abs(x['dphi'])):
        print('     %-14s |Δφ|=%.2f  损失 %+6.1f%%'
              % (r['tag'], abs(r['dphi']), 100 * r['loss']))

    # ---- 统一拟合 vs 分族拟合 ----
    allr = F + NN
    x = np.abs(np.array([r['dphi'] for r in allr]))
    y = np.array([r['loss'] for r in allr])
    k = np.polyfit(x, y, 1)
    rr = y - np.polyval(k, x)
    print('\n  === 若强行用单一 |Δφ| 线性律描述全部点 ===')
    print('     loss = %+.3f·|Δφ| %+.3f   rms = %.4f（数据 std %.4f）'
          % (k[0], k[1], np.sqrt((rr ** 2).mean()), y.std()))
    print('     R² = %.3f  ⇒ %s'
          % (1 - (rr ** 2).mean() / y.var(),
             '相位能解释大部分方差' if (1 - (rr ** 2).mean() / y.var()) > 0.6
             else '**相位只解释一部分**，另有独立变量（轨迹结构）'))

    # ---- 分族斜率 ----
    print('\n  === 分族斜率（同一相位尺度上的"效率"）===')
    for fam in sorted(set(r['fam'] for r in allr)):
        g = [r for r in allr if r['fam'] == fam]
        if len(g) < 3:
            continue
        xx = np.abs(np.array([r['dphi'] for r in g]))
        yy = np.array([r['loss'] for r in g])
        kk = np.polyfit(xx, yy, 1)
        print('     %-22s 斜率 %+.3f /rad（每 rad 相位造成 %.0f%% 损失）'
              % (fam, kk[0], 100 * abs(kk[0])))

    out = os.path.join(TMP, '_edge_cross_family.json')
    json.dump({'legacy_F': F, 'new': NN}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()