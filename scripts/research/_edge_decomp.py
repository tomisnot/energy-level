# -*- coding: utf-8 -*-
"""机制分解（最终）：交换振幅的损失中，多少来自"两通道分配"，多少来自"总量"。

恒等式：a = s · (a/s)
    Δa = a − a_ref = s·Δ(a/s) + (a/s)_ref·Δs + Δs·Δ(a/s)
         \___________/   \______________/   \_________/
           分配项            总量项            二阶小量

对 20 个实验点逐点计算两项的贡献比例 ⇒ 给出"机制由什么主导"的定量答案。
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
D = os.path.join(TMP, '_edge_all.json')


def main():
    T = json.load(io.open(D, encoding='utf-8'))['rows']
    ref = T['C/step']
    s_ref, f_ref = ref['s'], ref['as_frac']
    print('  参照 C/step: a=%.4f  s=%.4f  a/s=%.4f' % (ref['pk'], s_ref, f_ref))
    print()
    print('  %-20s %8s %8s %9s %10s %10s %8s'
          % ('tag', 'a', 's', 'a/s', '分配项', '总量项', '分配占比'))
    print('  ' + '-' * 78)

    rows = []
    for k, r in T.items():
        if k == 'C/step':
            continue
        A = s_ref * (r['as_frac'] - f_ref)          # 分配项
        B = f_ref * (r['s'] - s_ref)                # 总量项
        C2 = (r['s'] - s_ref) * (r['as_frac'] - f_ref)
        tot = A + B + C2                            # = r.a − ref.a
        share = abs(A) / (abs(A) + abs(B) + abs(C2)) if (abs(A) + abs(B) + abs(C2)) else np.nan
        # 相位差折叠到 (−π, π]
        dphi = (r['phi0'] - ref['phi0'] + np.pi) % (2 * np.pi) - np.pi
        rows.append(dict(tag=k, a=r['pk'], s=r['s'], frac=r['as_frac'],
                         A=A, B=B, C2=C2, tot=tot, share=share,
                         loss=r['pk'] - ref['pk'], phi=float(dphi)))
    rows.sort(key=lambda x: x['a'])
    for r in rows:
        print('  %-20s %8.4f %8.4f %9.4f %+10.4f %+10.4f %7.1f%%'
              % (r['tag'], r['a'], r['s'], r['frac'], r['A'], r['B'],
                 100 * r['share']))

    A = np.array([r['A'] for r in rows])
    B = np.array([r['B'] for r in rows])
    tot = np.array([r['tot'] for r in rows])
    loss = np.array([r['loss'] for r in rows])
    print('\n  === 汇总 ===')
    print('     |分配项| 均值 %.4f   |总量项| 均值 %.4f'
          % (np.abs(A).mean(), np.abs(B).mean()))
    print('     分配项占总变动 %.1f%%  /  总量项占 %.1f%%'
          % (100 * np.abs(A).sum() / (np.abs(A).sum() + np.abs(B).sum()),
             100 * np.abs(B).sum() / (np.abs(A).sum() + np.abs(B).sum())))
    print('     恒等式校验：max|Δa − (A+B+C2)| = %.2e'
          % np.max(np.abs(loss - tot)))
    print('     corr(分配项, Δa) = %+.3f ；corr(总量项, Δa) = %+.3f'
          % (np.corrcoef(A, loss)[0, 1], np.corrcoef(B, loss)[0, 1]))
    print('\n     ⇒ %s' % ('**分配主导**：交换能力的损失基本由两通道分配决定'
                          if np.abs(A).sum() > 3 * np.abs(B).sum() else
                          '总量也贡献显著'))

    # 分配项与相位的散点（供绘图）
    print('\n  === a/s 与 Δφ（有符号，按族）===')
    for fam in ('F', 'C', 'A', 'B', 'S', 'T'):
        g = [r for r in rows if r['tag'].startswith(fam + '/')]
        if not g:
            continue
        x = np.array([r['phi'] for r in g])
        y = np.array([r['frac'] for r in g])
        o = np.argsort(x)
        print('  %s: ' % fam, ' '.join('(%.2f,%.3f)' % (x[i], y[i]) for i in o))

    out = os.path.join(TMP, '_edge_decomp.json')
    json.dump({'rows': rows, 'ref': {'s': s_ref, 'frac': f_ref, 'a': ref['pk']}},
              io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()