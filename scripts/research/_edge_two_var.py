# -*- coding: utf-8 -*-
"""判别性检验：幅度损失由"相位积累"还是"单向性（穿越方向）"决定？

两个候选控制量（都对 9 个点可算，且**互不共线**）：
  X1 = |Δφ|            沿造成的净相位（相对理想阶跃）
  X2 = 单向净扫过的强度行程 = I₃ −（末态强度） 的方向性积分
       = ∫ |dI/dt|·sign(单向) dt  ⇒ 对称往返 = 0，单向下降 = 幅度差

为什么必须做双变量：单变量相关会互相污染。
  · |Δφ| 与损失 corr=+0.75（但不解释 B/hold50：Δφ 大而损失 0）
  · 若拟合后 X2 的系数显著、X1 不显著 ⇒ 机制是"单向穿越"
  · 反之 ⇒ 机制是相位积累
  · 两者都显著 ⇒ 两个机制并存（需物理模型）
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research'))
from _edge_profiles import profiles, IRES          # noqa: E402

LP = os.path.join(ROOT, 'scripts', 'tmp', '_edge_loss_vs_phase.json')


def oneway_stroke(kind, tag):
    """单向净扫过的强度行程：对称往返 = 0；单向下降 = I_平台 − I_末态。"""
    for tag2, br, n in profiles(kind):
        if tag2 != tag:
            continue
        t = np.array([b[0] for b in br], float)
        v = np.array([b[1] for b in br], float)
        # 只算 255 ns 之后的 excursion 段
        m = t >= 255.0
        tt, vv = t[m], v[m]
        if tt.size < 2:
            return 0.0
        dv = np.diff(vv)
        up = float(np.clip(dv, 0, None).sum())
        dn = float(np.clip(-dv, 0, None).sum())
        return abs(up - dn)          # 单向部分；往返 ⇒ 0
    return np.nan


def main():
    rows = json.load(io.open(LP, encoding='utf-8'))['rows']
    # J（净失谐积分）不在本文件里，从 _edge_final.json 取
    fin = {r['tag']: r for r in json.load(
        io.open(os.path.join(ROOT, 'scripts', 'tmp', '_edge_final.json'),
                encoding='utf-8'))['rows']}
    for r in rows:
        r['J'] = fin.get(r['tag'], {}).get('J', float('nan'))
    kind_of = {r['tag']: ('A' if r['tag'].startswith('A/')
                          else 'B' if r['tag'].startswith('B/') else 'C')
               for r in rows}
    for r in rows:
        r['X2'] = oneway_stroke(kind_of[r['tag']], r['tag'])

    print('  %-18s %8s %10s %10s %10s' %
          ('tag', '损失%', '|Δφ|', '单向行程', '超期积分J'))
    print('  ' + '-' * 62)
    for r in sorted(rows, key=lambda x: x['loss_rel']):
        print('  %-18s %+7.1f%% %10.3f %10.4f %10.1f'
              % (r['tag'], r['loss_rel'] * 100, abs(r['dphi']), r['X2'], r['J']))

    X1 = np.abs(np.array([r['dphi'] for r in rows]))
    X2 = np.array([r['X2'] for r in rows])
    y = np.array([r['loss_rel'] for r in rows])
    print('\n  === 单变量 ===')
    for nm, X in (('|Δφ|', X1), ('单向行程', X2)):
        c = np.corrcoef(X, y)[0, 1]
        k = np.polyfit(X, y, 1)
        rr = y - np.polyval(k, X)
        print('     %-8s corr=%+.3f  rms=%.4f' % (nm, c, np.sqrt((rr ** 2).mean())))
    print('     两者共线? corr(|Δφ|, 单向行程) = %+.3f' % np.corrcoef(X1, X2)[0, 1])

    print('\n  === 双变量最小二乘 y = a·|Δφ| + b·单向行程 + c ===')
    G = np.column_stack([X1, X2, np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(G, y, rcond=None)
    pred = G @ coef
    rms = float(np.sqrt(np.mean((pred - y) ** 2)))
    # 标准化系数（便于比较量纲不同的两个量）
    sd = [X1.std(), X2.std()]
    print('     a(|Δφ|)   = %+.4f   (标准化 %+.4f)' % (coef[0], coef[0] * sd[0] / y.std()))
    print('     b(单向)   = %+.4f   (标准化 %+.4f)' % (coef[1], coef[1] * sd[1] / y.std()))
    print('     c         = %+.4f' % coef[2])
    print('     拟合 rms  = %.4f  （数据 rms = %.4f）'
          % (rms, float(np.sqrt(np.mean((y - y.mean()) ** 2)))))
    print('     ⇒ %s' % ('两个机制都显著' if (coef[0] * sd[0] > 0.3 * y.std()
                                          and coef[1] * sd[1] > 0.3 * y.std())
                        else '其中一个主导'))

    # 逐点残差（看是否有离群）
    print('\n  %-18s %10s %10s %10s' % ('tag', '实测损失', '预测', '残差'))
    for r, p in zip(rows, pred):
        print('  %-18s %+9.1f%% %+9.1f%% %+9.1f%%'
              % (r['tag'], r['loss_rel'] * 100, p * 100, (p - r['loss_rel']) * 100))

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_two_var.json')
    json.dump({'coef': [float(x) for x in coef], 'rms': rms,
               'X1': X1.tolist(), 'X2': X2.tolist(), 'loss': y.tolist(),
               'tags': [r['tag'] for r in rows]},
              io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()