# -*- coding: utf-8 -*-
"""用**终态布居**做判决（首批扫描 record='final'，无时间序列）。

物理：若沿只改相位，则末态 = C + A·cos(2πf·τ_traj + φ0)，其中
  τ_traj = 平台持续时长（从 255 ns 到 2000 ns 末尾，固定）
  ⇒ 末态随"平台段积分"变化。两条独立检验：

  E1  **平台段积分不变量**：末态依赖轨迹的方式若只是通过
      S = ∫₂₅₅²⁰⁰⁰ (I/I3) dt（平台段光强时间积分），则所有点应落在
      "末态 vs S" 的同一条曲线上 —— 这是比前人"超额积分"更干净的判据
      （超额积分只看 (I−I_res)⁺，丢了低于 I_res 的部分）。
  E2  **总布居守恒**：若损失是真布居泄漏，(a+b) 会随沿宽下降；
      若只是相位，(a+b) 应当不变。

数据：runtime/sweeps/sw20260916_094856_703/pt_*.npz（labels/final_pop）
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
SID = 'sw20260916_094856_703'
SDIR = os.path.join(ROOT, 'runtime', 'sweeps', SID)
PF = os.path.join(ROOT, 'scripts', 'tmp', '_edge_profiles.json')
K_AFT = '(60S_{1/2} m_J=-0.5) x (60P_{3/2} m_J=+1.5)'
K_BEF = '(60P_{3/2} m_J=-0.5) x (60S_{1/2} m_J=+0.5)'
IRES = 0.42
T0 = 255.0


def load_profile(tag):
    """从本地档取折线（tag↔轨迹的唯一真值源）。"""
    pf = json.load(io.open(PF, encoding='utf-8'))
    for p in pf['profiles']:
        if p['tag'] == tag:
            return p['breaks']
    return None


def integrals(br, t_end=2000.0):
    """平台段 [T0, t_end] 的光强积分 S=∫(I/I3)dt、净失谐积分 J=∫(I/I3−res)dt。"""
    t = np.asarray([b[0] for b in br], float)
    v = np.asarray([b[1] for b in br], float)
    m = t >= T0
    tt = np.clip(t[m], T0, t_end)
    vv = v[m]
    if tt.size < 2:
        return 0.0, 0.0
    S = float(np.trapezoid(vv, tt))
    J = float(np.trapezoid(vv - IRES, tt))
    return S, J


def main():
    pf = json.load(io.open(PF, encoding='utf-8'))
    tags = pf['tags']
    rows = []
    for f in sorted(glob.glob(os.path.join(SDIR, 'pt_*.npz'))):
        k = int(os.path.basename(f)[3:7])
        z = np.load(f, allow_pickle=False)
        labels = [str(x) for x in z['labels']]
        pop = np.asarray(z['final_pop'], float)
        if K_AFT not in labels:
            continue
        a = float(pop[labels.index(K_AFT)])
        b = float(pop[labels.index(K_BEF)])
        tag = tags[k] if k < len(tags) else 'pt%d' % k
        br = load_profile(tag)
        S, J = integrals(br) if br else (np.nan, np.nan)
        rows.append(dict(k=k, tag=tag, a=a, b=b, tot=a + b, S=S, J=J))

    print('  %-18s %8s %8s %8s %9s %9s' %
          ('tag', 'a(换后)', 'b(换前)', 'a+b', 'S(平台段)', 'J(净失谐)'))
    print('  ' + '-' * 74)
    for r in sorted(rows, key=lambda x: x['k']):
        print('  %-18s %8.4f %8.4f %8.4f %9.1f %9.1f'
              % (r['tag'], r['a'], r['b'], r['tot'], r['S'], r['J']))

    # ---- E2：总布居守恒 ----
    print('\n  === E2 两通道总布居 a+b（真泄漏会让它下降）===')
    famA = [r for r in rows if r['tag'].startswith('C/')]
    famF = [r for r in rows if r['tag'].startswith('B/')]
    for nm, fam in (('C 族(单向下降)', famA), ('B 族(停留式)', famF)):
        if fam:
            tots = [r['tot'] for r in fam]
            print('    %-14s a+b = %s   极差 %.4f'
                  % (nm, ' '.join('%.4f' % x for x in tots),
                     max(tots) - min(tots)))

    # ---- E1：末态 vs S 曲线 ----
    print('\n  === E1 末态 a 是否只是平台段积分 S 的函数 ===')
    print('      （若所有点落在同一曲线 ⇒ 沿的作用=改变积分；否则相位另有贡献）')
    ok = [r for r in rows if r['S'] is not None and not np.isnan(r['S'])]
    ok.sort(key=lambda r: r['S'])
    print('  %-18s %9s %8s' % ('tag', 'S', 'a'))
    for r in ok:
        print('  %-18s %9.1f %8.4f' % (r['tag'], r['S'], r['a']))

    # 在 S 相近的点对之间比较 a，看是否一致
    print('\n  === 关键配对（S 相近但沿宽不同）===')
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            ds = abs(ok[i]['S'] - ok[j]['S'])
            if ds < 1.0:
                print('    %-16s (S=%.1f, a=%.4f)  vs  %-16s (S=%.1f, a=%.4f)'
                      '  Δa=%+.4f'
                      % (ok[i]['tag'], ok[i]['S'], ok[i]['a'],
                         ok[j]['tag'], ok[j]['S'], ok[j]['a'],
                         ok[i]['a'] - ok[j]['a']))

    out = os.path.join(ROOT, 'scripts', 'tmp', '_edge_endpoint_%s.json' % SID)
    json.dump({'rows': rows}, io.open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n  数据 → %s' % out)


if __name__ == '__main__':
    main()