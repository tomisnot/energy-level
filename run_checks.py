# -*- coding: utf-8 -*-
"""Re0 重构 · 物理基线跑：把判据跑一遍并汇总。

用法：
    python run_checks.py                # 快档（默认，跳过 18 分钟的第一性原理基准）
    python run_checks.py --slow         # 含慢基准（mw_bench_firstprinciples，约 18 分钟）
    python run_checks.py --only sambe
    python run_checks.py --list
    python run_checks.py --dir old/checks

判据一律以**独立进程**跑（与原来 selftest_all 同一纪律：脚本间无共享状态）。
退出码 = FAIL 的脚本数（0 = 全绿）。

判据目录：默认依次找 `checks/`、`old/checks/`（沙盒被重排过，两处都认，
同一批判据的两份拷贝 → 只跑第一处）。
"""
import argparse
import os
import re
import subprocess

# 判据一律 in-process GUI（不连活 Hub）：仓库 cwd 可能有活 Hub 写的 .mcp-port，
# 若 GUI 判据连了活 Hub 会断言 in-process cache 非 None 失败。
os.environ.setdefault('RE0_GUI_INPROC', '1')
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.abspath(__file__))

# 支撑模块（不是判据，不单独跑）
SUPPORT = {'checklib.py', 'pair_evolve.py'}
# 慢基准：默认跳过（约 18 分钟），--slow 才跑
SLOW = {'mw_bench_firstprinciples.py'}
# **接线判据**：曾经验旧架构 GUI 登记册（pm_setup）的接线、在纯引擎沙盒里跑不了而默认跳过。
# Re0 回归主仓库后 GUI 已迁入（solve_max_step_hard 的 D 组改用 runspec.SOLVE_KWARG/registry），
# 这些判据现在**属于主干、默认跑**——WIRING 清空（--wiring 开关保留但已无对象）。
WIRING = {}


CANDIDATE_DIRS = ['checks', 'old/checks']


def find_dir(explicit=None):
    """取**第一个**命中的判据目录（不合并）。

    ⚠ 别改成"全部候选目录都跑"：本仓 `checks/` 与 `old/checks/` 是**同一批
    判据的两份拷贝**（沙盒被重排过），同时跑会各跑一遍、且 `old/` 那份的
    `sys.path` 前缀不同 → 一堆假 FAIL（实测 29 个跑出 14 个 FAIL）。
    """
    cands = [explicit] if explicit else list(CANDIDATE_DIRS)
    for c in cands:
        p = os.path.join(ROOT, c.replace('/', os.sep))
        if os.path.isdir(p) and any(f.endswith('.py') for f in os.listdir(p)):
            return p
    return None

RE_SUM = re.compile(r'\[(\w+)\]\s+PASS=(\d+)\s+FAIL=(\d+)\s+SKIP=(\d+)'
                    r'(?:\s+SUB=(\d+))?')
RE_NN = re.compile(r'(\d+)/(\d+)\s+PASS')
RE_OK = re.compile(r'\[PASS\]|\[ ok \]|^\s*OK\b', re.M)
RE_BAD = re.compile(r'\[FAIL\]|FAIL=[1-9]', re.M)


def counts(text):
    m = RE_SUM.search(text)
    if m:
        return int(m.group(2)), int(m.group(3)), int(m.group(4))
    nn = RE_NN.findall(text)
    if nn:
        p, t = max(((int(a), int(b)) for a, b in nn), key=lambda x: x[1])
        return p, max(0, t - p), 0
    return len(RE_OK.findall(text)), len(RE_BAD.findall(text)), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--slow', action='store_true',
                    help='含慢基准（mw_bench_firstprinciples，约 18 分钟）')
    ap.add_argument('--wiring', action='store_true',
                    help='含接线判据（需旧 GUI 登记册，沙盒里跑不了）')
    ap.add_argument('--dir', default=None, help='判据目录（默认自动找 checks/ 或 old/checks/）')
    a = ap.parse_args()

    CHECKS = find_dir(a.dir)
    if CHECKS is None:
        print('找不到判据目录：试过 %s。用 --dir 指定。'
              % '、'.join(CANDIDATE_DIRS))
        return 2

    scripts = [f for f in sorted(os.listdir(CHECKS))
               if f.endswith('.py') and f not in SUPPORT]
    if not a.wiring:
        scripts = [s for s in scripts if s not in WIRING]
    if not a.slow:
        scripts = [s for s in scripts if s not in SLOW]
    if a.only:
        scripts = [s for s in scripts if a.only in s]
    if a.list:
        for s in scripts:
            print(' ', s)
        return 0

    env = dict(os.environ)
    env.setdefault('PYTHONPATH', '')
    env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    env.setdefault('EL_NO_AI_SERVER', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')

    print('判据目录：%s' % CHECKS)
    print('跑 %d 个判据%s\n' % (len(scripts), '' if a.slow else '（已跳过慢基准）'))
    if not a.wiring:
        for s, why in sorted(WIRING.items()):
            print('  [跳过·接线] %-38s %s' % (s, why))
        print()
    t_all = time.perf_counter()
    rows, fail = [], 0
    for s in scripts:
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, '-u', os.path.join(CHECKS, s)],
                           cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=1800)
        sec = time.perf_counter() - t0
        out = (p.stdout or '') + (p.stderr or '')
        ok, bad, skip = counts(out)
        bad = max(bad, 0 if p.returncode == 0 else -p.returncode and 0) or bad
        if p.returncode != 0 and bad == 0:
            bad = 1
        fail += 1 if bad else 0
        rows.append((s, p.returncode, ok, bad, skip, sec))
        print('  %-40s rc=%-3d PASS=%-4d FAIL=%-3d SKIP=%-3d %5.1fs'
              % (s, p.returncode, ok, bad, skip, sec))
        if bad:
            tail = [l for l in out.splitlines() if 'FAIL' in l][:4]
            for l in tail:
                print('        %s' % l.strip()[:110])

    print('\n%s' % ('-' * 72))
    print('##RE0_SUMMARY scripts=%d fail=%d seconds=%.0f'
          % (len(rows), fail, time.perf_counter() - t_all))
    print('%s' % ('全部通过 ✅' if fail == 0 else '有 %d 个未过 ❌' % fail))
    return fail


if __name__ == '__main__':
    sys.exit(main())
