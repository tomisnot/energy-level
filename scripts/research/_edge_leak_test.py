# -*- coding: utf-8 -*-
"""关键物理判据：幅度损失是"布居泄漏出交换子空间"还是"子空间内相干重分布"？

区分方式（不依赖模型）：
  在**峰值时刻**取两个量
      a = P(换后态)          目标通道
      b = P(换前态)          源通道
      s = a + b              交换子空间（两通道）内总布居
  若损失是**泄漏**：s 应随沿变宽而下降（布居跑去别的态）
  若损失是**相干重分布**：s 应基本不变，只有 a/b 的分配与相位变

同时给出：
  · 峰值时刻的"交换占比" a/s
  · 与理想阶跃相比的 Δs（泄漏量）与 Δa（表观幅度损失）
  · 峰值时刻的旁观者布居（6P×P 族等）是否反向变化
另附：终态(t=2000)的 s 与 (a+b)，作为时间无关的对照。
"""
import glob
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research'))
from _edge_profiles import profiles, one          # noqa: E402

K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
ORDER = ['C/step', 'B/hold10@0.71', 'B/hold25@0.71', 'B/hold50@0.71',
         'C/down25', 'C/down50', 'A/tri25_d1.00', 'A/tri50_d1.00',
         'A/tri100_d1.00']


def locate():
    """指纹 → (npz 路径)。"""
    ref = {}
    for kind in ('A', 'B', 'C'):
        for tag, br, n in profiles(kind):
            env = one(tag, br, n)['dynamic.waves_A'][2]['env']
            ref[tag] = np.array(env[1], float)
    loc = {}
    for d in sorted(glob.glob(os.path.join(ROOT, 'runtime', 'sweeps', 'sw*')),
                    key=os.path.getmtime):
        sp = os.path.join(d, 'spec.json')
        if not os.path.exists(sp):
            continue
        spec = json.load(io.open(sp, encoding='utf-8'))
        for i, p in enumerate(spec.get('points') or []):
            wa = p.get('dynamic.waves_A')
            if not wa or len(wa) != 3 or wa[2].get('env', [None])[0] != 'file':
                continue
            tt = np.array(wa[2]['env'][1], float)
            for tag, rt in ref.items():
                if rt.size == tt.size and np.allclose(rt, tt, atol=1e-6):
                    f = os.path.join(d, 'pt_%04d.npz' % i)
                    if os.path.exists(f) and (tag not in loc or
                                              os.path.getmtime(f) > loc[tag][1]):
                        loc[tag] = (f, os.path.getmtime(f))
    return loc


def main():
    loc = locate()
    rows = []
    for tag in ORDER:
        if tag not in loc:
            continue
        z = np.load(loc[tag][0], allow_pickle=False)
        labels = [str(x) for x in z['labels']]
        if K_AFT not in labels or K_BEF not in labels:
            continue
        t = np.asarray(z['series_t'], float)
        P = np.asarray(z['series_pop'], float)
        ia, ib = labels.index(K_AFT), labels.index(K_BEF)
        a, b = P[:, ia], P[:, ib]
        k = int(np.argmax(a))
        # 峰值时刻的交换子空间（两通道）布居
        s = float(a[k] + b[k])
        # 峰值时刻的"其余"布居（含 6P×P 族与其它 60P/60S 组合）
        rest = float(1.0 - P[k].sum() + P[k, ia] + P[k, ib])
        rest_full = float(1.0 - P[k].sum())
        rows.append(dict(tag=tag, pk=float(a[k]), t_pk=float(t[k]),
                         bef_at_pk=float(b[k]), s_at_pk=s,
                         frac_at_pk=float(a[k] / s) if s else np.nan,
                         loss_at_pk=rest_full, end_s=float(a[-1] + b[-1]),
                         end_a=float(a[-1]), end_b=float(b[-1]),
                         mean_s=float(np.mean(a + b))))

    ref = next(r for r in rows if r['tag'] == 'C/step')
    print('  参照 C/step: 峰值 %.4f @ %.0f ns  s(峰值时)=%.4f  峰值时交换占比 %.3f'
          % (ref['pk'], ref['t_pk'], ref['s_at_pk'], ref['frac_at_pk']))
    print()
    print('  %-18s %7s %7s %8s %8s %9s %9s %9s' %
          ('tag', '峰值a', '峰值b', 's=a+b', 'a/s', 'Δa', 'Δs', '堆外布居'))
    print('  ' + '-' * 84)
    for r in rows:
        print('  %-18s %7.4f %7.4f %8.4f %8.3f %+9.4f %+9.4f %9.4f'
              % (r['tag'], r['pk'], r['bef_at_pk'], r['s_at_pk'], r['frac_at_pk'],
                 r['pk'] - ref['pk'], r['s_at_pk'] - ref['s_at_pk'], r['loss_at_pk']))

    # ---- 判据 ----
    print('\n  === 判据：Δs（泄漏）与 Δa（表观损失）的量级对比 ===')
    da = np.array([abs(r['pk'] - ref['pk']) for r in rows])
    ds = np.array([abs(r['s_at_pk'] - ref['s_at_pk']) for r in rows])
    print('     |Δa| 最大 %.4f（相对 %.1f%%）' % (da.max(), 100 * da.max() / ref['pk']))
    print('     |Δs| 最大 %.4f（相对 %.1f%%）' % (ds.max(), 100 * ds.max() / ref['s_at_pk']))
    print('     ⇒ %s' % ('s 几乎不变 ⇒ 损失是**子空间内相干重分布**，不是泄漏'
                        if ds.max() < 0.4 * da.max() else
                        's 也明显变化 ⇒ 有相当部分是真泄漏'))
    print('\n     s(峰值时) 跨 9 条：%.4f – %.4f（极差 %.4f，相对 %.1f%%）'
          % (min(r['s_at_pk'] for r in rows), max(r['s_at_pk'] for r in rows),
             max(r['s_at_pk'] for r in rows) - min(r['s_at_pk'] for r in rows),
             100 * (max(r['s_at_pk'] for r in rows) - min(r['s_at_pk'] for r in rows))
             / ref['s_at_pk']))
    print('     a/s(峰值时) 跨 9 条：%.3f – %.3f'
          % (min(r['frac_at_pk'] for r in rows), max(r['frac_at_pk'] for r in rows)))

    # ---- 全时均值（时间无关的对照）----
    print('\n  === 对照：全程平均的两通道布居和 ===')
    for r in sorted(rows, key=lambda x: x['mean_s']):
        print('     %-18s 平均 a+b = %.4f' % (r['tag'], r['mean_s']))
    ms = np.array([r['mean_s'] for r in rows])
    print('     极差 %.4f（相对 %.1f%%）' % (ms.max() - ms.min(), 100 * (ms.max() - ms.min()) / ms.mean()))

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_leak_test.json')
    json.dump({'rows': rows, 'ref': ref['tag']},
              io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()