# -*- coding: utf-8 -*-
"""最终定量：交换幅度损失 vs 沿造成的相位积累（4 窗口平均，抗窗口噪声）。

动机：判决实验显示幅度损失既不随"停留时间/光强积分"（S 相同差 7.7 倍），
也不随"穿越速率"单调。剩下的候选控制量是**沿造成的净相位**
Δφ = φ0(轨迹) − φ0(理想阶跃)。本脚本测 相对幅度损失 与 Δφ 的关系，
并同时检验 P2（时长无关）的否证。

输出一个可检验的经验定律：loss_rel = f(Δφ)。
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIN = os.path.join(ROOT, 'scripts', 'tmp', '_edge_final.json')
WIN = os.path.join(ROOT, 'scripts', 'tmp', '_edge_final_windows.json')
WIN4 = [(340, 2000), (340, 1200), (600, 1800), (1000, 2000)]


def main():
    fin = {r['tag']: r for r in json.load(io.open(FIN, encoding='utf-8'))['rows']}
    win = json.load(io.open(WIN, encoding='utf-8'))
    tab = win['A0']
    order = [w for w in WIN4]
    # JSON 把 tuple 存成 list ⇒ 索引时统一成 tuple
    wl = [tuple(x) for x in win['windows']]
    idx = [wl.index(w) for w in order]

    rows = []
    for tag, v in tab.items():
        a4 = np.array([v[j] for j in idx], float)
        rows.append(dict(tag=tag, A0=float(a4.mean()), A0_spread=float(a4.max() - a4.min()),
                         phi0=fin[tag]['phi0'], f0=fin[tag]['f0']))
    ref = next(r for r in rows if r['tag'] == 'C/step')
    print('  参照 C/step: A0=%.4f  φ0=%.3f  f0=%.4f MHz'
          % (ref['A0'], ref['phi0'], ref['f0'] * 1000))
    for r in rows:
        # 相位差取最近邻展开（频率相同 ⇒ 相位差无歧义到 2π 倍数）
        d = r['phi0'] - ref['phi0']
        d = (d + np.pi) % (2 * np.pi) - np.pi
        r['dphi'] = float(d)
        r['loss_rel'] = float((ref['A0'] - r['A0']) / ref['A0'])   # 正 = 损失

    print('\n  %-18s %8s %8s %9s %9s %8s' %
          ('tag', 'A0(4窗均)', '窗内极差', 'Δφ(rad)', '|Δφ|', '相对损失'))
    print('  ' + '-' * 72)
    for r in sorted(rows, key=lambda x: abs(x['dphi'])):
        print('  %-18s %8.4f %8.4f %9.3f %9.3f %+7.1f%%'
              % (r['tag'], r['A0'], r['A0_spread'], r['dphi'], abs(r['dphi']),
                 r['loss_rel'] * 100))

    d = np.array([abs(r['dphi']) for r in rows])
    L = np.array([r['loss_rel'] for r in rows])
    print('\n  === 判据：损失是 |Δφ| 的函数吗 ===')
    print('     corr(|Δφ|, 相对损失) = %+.3f' % np.corrcoef(d, L)[0, 1])
    k1 = np.polyfit(d, L, 1)
    r1 = L - np.polyval(k1, d)
    k2 = np.polyfit(d ** 2, L, 1)
    r2 = L - np.polyval(k2, d ** 2)
    print('     线性  loss = %+.4f·|Δφ| %+.3f   rms %.4f' % (k1[0], k1[1], np.sqrt((r1**2).mean())))
    print('     平方  loss = %+.4f·|Δφ|² %+.3f  rms %.4f' % (k2[0], k2[1], np.sqrt((r2**2).mean())))
    # 纯相位 vs 真损失：若损失只是 |Δφ| 的确定函数，则给定 Δφ 可预测 A0
    pred = ref['A0'] * (1 - np.polyval(k1, d))
    print('     由线性律回代的 A0 残差 rms = %.4f'
          % np.sqrt(np.mean((pred - np.array([r['A0'] for r in rows])) ** 2)))

    print('\n  === P2 否证（同一族内，时长是否无关）===')
    for fam, tags in (('A 族(对称往返)', ['A/tri25_d1.00', 'A/tri50_d1.00', 'A/tri100_d1.00']),
                      ('B 族(停留式)', ['B/hold10@0.71', 'B/hold25@0.71', 'B/hold50@0.71']),
                      ('C 族(单向)', ['C/down25', 'C/down50'])):
        g = [r for r in rows if r['tag'] in tags]
        if len(g) > 1:
            a = np.array([r['A0'] for r in g])
            print('     %-14s A0 = %s  极差 %.4f（相对 %.1f%%）⇒ %s'
                  % (fam, ' '.join('%.4f' % x for x in a), a.max() - a.min(),
                     100 * (a.max() - a.min()) / a.mean(),
                     '与时长无关' if (a.max() - a.min()) < 0.01 else '**与时长强相关**'))

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_loss_vs_phase.json')
    json.dump({'rows': rows,
               'fit_linear': {'k': float(k1[0]), 'b': float(k1[1]),
                              'rms': float(np.sqrt((r1 ** 2).mean()))},
               'corr': float(np.corrcoef(d, L)[0, 1])},
              io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()