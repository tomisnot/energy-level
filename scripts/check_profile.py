# -*- coding: utf-8 -*-
"""判据耗时剖析（一次性诊断工具，不进判据集）。

为什么需要：全量基线一次 ~600s，而开发迭代需要的是"改哪跑哪"。本脚本逐个跑
每个判据并记录**墙钟耗时 + rc + 子条件数**，产出一张耗时表，供设计"快档/慢档"
与"按改动自动选判据"用。

用法：
    & D:\python310\python.exe scripts\check_profile.py            # 全部
    & D:\python310\python.exe scripts\check_profile.py --timeout 300
输出：runtime/check_profile.csv（+ 控制台按耗时降序的表）
"""
import argparse
import csv
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKS = os.path.join(ROOT, 'checks')
SUPPORT = {'checklib.py', 'pair_evolve.py'}
OUT = os.path.join(ROOT, 'runtime', 'check_profile.csv')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--limit', type=int, default=0, help='只跑最慢的 N 个（0=全部）')
    ap.add_argument('--names', default='', help='逗号分隔的脚本名子串过滤')
    a = ap.parse_args()

    scripts = [f for f in sorted(os.listdir(CHECKS))
               if f.endswith('.py') and f not in SUPPORT]
    if a.names:
        want = [w.strip() for w in a.names.split(',') if w.strip()]
        scripts = [s for s in scripts if any(w in s for w in want)]

    env = dict(os.environ)
    env.setdefault('PYTHONPATH', '')
    env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    env.setdefault('EL_NO_AI_SERVER', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')

    print('剖析 %d 个判据（单个超时 %.0fs）…\n' % (len(scripts), a.timeout))
    rows = []
    t_all = time.perf_counter()
    for s in scripts:
        t0 = time.perf_counter()
        rc, note = 0, ''
        try:
            p = subprocess.run([sys.executable, '-u', os.path.join(CHECKS, s)],
                               cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=a.timeout)
            rc = p.returncode
            out = (p.stdout or '') + (p.stderr or '')
            import re
            m = re.search(r'PASS=(\d+)\s+FAIL=(\d+)\s+SKIP=(\d+)', out)
            note = ('P%d/F%d/S%d' % (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    if m else '?')
        except subprocess.TimeoutExpired:
            rc, note = -9, 'TIMEOUT>%.0fs' % a.timeout
        sec = time.perf_counter() - t0
        rows.append({'script': s, 'sec': round(sec, 1), 'rc': rc, 'counts': note})
        print('  %7.1fs  rc=%-3d %-12s %s' % (sec, rc, note, s), flush=True)

    total = time.perf_counter() - t_all
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['script', 'sec', 'rc', 'counts'])
        w.writeheader()
        w.writerows(rows)

    srow = sorted(rows, key=lambda r: -r['sec'])
    acc, top = 0.0, []
    for r in srow:
        acc += r['sec']
        if acc <= total * 0.8:
            top.append(r['script'])
    print('\n合计 %.0fs；占 80%% 时长的判据（%d 个）：%s'
          % (total, len(top), ', '.join(top)))
    print('CSV → %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
