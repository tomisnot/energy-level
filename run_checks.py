# -*- coding: utf-8 -*-
"""按改动范围运行物理与软件回归。默认 core，--all 运行常规全量。
--fast 选择较宽的快速场景；--slow 额外包含第一性原理慢基准。
--only NAME 不限制档位；--list 查看范围。文档维护、迁移词汇不属于门禁。
"""
import argparse
import os
import re
import subprocess
import sys
import time

# 判据一律 in-process（v2：GUI 判据自建 V2Bridge，不连活实例）。
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.abspath(__file__))

# 支撑模块（不是判据，不单独跑）
SUPPORT = {'checklib.py', 'pair_evolve.py'}
RELEASE = {'personal_paths_selfcheck.py'}
CORE = {
    'checks_sentinel.py', 'site_declaration_selfcheck.py',
    'field_loss_selfcheck.py', 'human_control_selfcheck.py', 'authority_selfcheck.py',
    'engine_contract_selfcheck.py', 'runspec_contract_selfcheck.py',
    'field_loss_engine_selfcheck.py', 'ops_threadsplit_selfcheck.py',
    'solve_max_step_hard_selfcheck.py', 'mw_pair_keep_selfcheck.py',
    'gui_bind_selfcheck.py', 'manybody_m1_ed_selfcheck.py',
    'ai_h_snapshot_guard_selfcheck.py', 'equibench_engine_ed_selfcheck.py',
    'equibench_gs_solvers_selfcheck.py', 'equibench_analytic_thermo_selfcheck.py',
    'equibench_heisenberg2d_selfcheck.py',
}
# 慢基准：默认跳过（约 18 分钟），--slow 才跑
SLOW = {'mw_bench_firstprinciples.py'}

# 较宽的快速场景；没有耗时登记准入要求。
FAST = CORE | {
    'drive_wave_selfcheck.py',
    'equibench_dmrg_general_selfcheck.py',
    'equibench_gpu_batched_selfcheck.py',
    'equibench_router_selfcheck.py',
    'gui_app_entry_selfcheck.py',
    'gui_inspector_selfcheck.py',
    'gui_run_flow_selfcheck.py',
    'gui_run_worker_selfcheck.py',
    'hard_dedup_selfcheck.py',
    'hard_vdd_pack_selfcheck.py',
    'hub_mcp_e2e_selfcheck.py',
    'manybody_integration_selfcheck.py',
    'mcp_defaults_selfcheck.py',
    'mcp_entry_smoke_selfcheck.py',
    'mixed_arb_mw_selfcheck.py',
    'narrow_pulse_env_selfcheck.py',
    'pair_evolve_selfcheck.py',
    'seed_rejection_visible_selfcheck.py',
    'sweep_gui_selfcheck.py',
}


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
    ap.add_argument('--all', action='store_true', help='常规全量，不含慢基准与发布扫描')
    ap.add_argument('--release', action='store_true', help='发布前个人路径扫描')
    ap.add_argument('--core', action='store_true',
                    help='核心契约与小尺寸物理结果（默认）')
    ap.add_argument('--fast', action='store_true',
                    help='较宽的快速场景，按受影响范围选择')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--slow', action='store_true',
                    help='含慢基准（mw_bench_firstprinciples，约 18 分钟）')
    ap.add_argument('--timeout', type=float, default=300.0,
                    help='单个判据墙钟上限（秒，默认 300）；超时记 FAIL，不挂整条基线')
    ap.add_argument('--dir', default=None, help='判据目录（默认自动找 checks/ 或 old/checks/）')
    a = ap.parse_args()

    CHECKS = find_dir(a.dir)
    if CHECKS is None:
        print('找不到判据目录：试过 %s。用 --dir 指定。'
              % '、'.join(CANDIDATE_DIRS))
        return 2

    scripts = [f for f in sorted(os.listdir(CHECKS))
               if f.endswith('.py') and f not in SUPPORT]
    if a.timeout <= 0:
        ap.error("--timeout 必须大于 0")
    if not a.slow and not a.only:
        scripts = [s for s in scripts if s not in SLOW]
    if not a.release and not a.only:
        scripts = [s for s in scripts if s not in RELEASE]
    if a.release:
        scripts = [s for s in scripts if s in RELEASE]
    elif a.core or not (a.all or a.fast or a.slow or a.only):
        a.core = True
        scripts = [s for s in scripts if s in CORE]
    elif a.fast:
        scripts = [s for s in scripts if s in FAST]
    if a.only:
        scripts = [s for s in scripts if a.only in s]
    if not scripts:
        ap.error("没有匹配的验证项")
    if a.list:
        for s in scripts:
            print(' ', s)
        return 0

    env = dict(os.environ)
    env.setdefault('PYTHONPATH', '')
    env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    env.setdefault('EL_NO_AI_SERVER', '1')
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # v2 判据的 GUI 装配（MainWindow→make_v2_bridge）写临时数据根，
    # 不污染仓内权威 journal（v1 时代 RE0_GUI_INPROC+临时 root 的同款纪律）。
    if not env.get('RE0_V2_DATA_ROOT'):
        import tempfile
        env['RE0_V2_DATA_ROOT'] = tempfile.mkdtemp(prefix='re0v2-checks-')

    print('判据目录：%s' % CHECKS)
    _tier = '（内循环档 --core）' if a.core else ('（提交前档 --fast）' if a.fast else '')
    print('跑 %d 个判据%s%s\n' % (len(scripts), _tier,
                                  '' if a.slow else '（已跳过慢基准）'))
    t_all = time.perf_counter()
    rows, fail = [], 0
    for s in scripts:
        t0 = time.perf_counter()
        timed_out = False
        try:
            p = subprocess.run([sys.executable, '-u', os.path.join(CHECKS, s)],
                               cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=a.timeout)
            rc, out = p.returncode, (p.stdout or '') + (p.stderr or '')
        except subprocess.TimeoutExpired as e:
            # 单判据挂死不该拖垮整条基线（实测踩过：一个判据卡到 843s）。
            rc = -9
            timed_out = True
            out = ((e.stdout or b'').decode('utf-8', 'replace') if isinstance(e.stdout, bytes)
                   else (e.stdout or '')) or ''
        sec = time.perf_counter() - t0
        ok, bad, skip = counts(out)
        if timed_out:
            bad = max(bad, 1)
        elif rc != 0 and bad == 0:
            bad = 1
        fail += 1 if bad else 0
        rows.append((s, rc, ok, bad, skip, sec))
        print('  %-40s rc=%-3d PASS=%-4d FAIL=%-3d SKIP=%-3d %5.1fs%s'
              % (s, rc, ok, bad, skip, sec,
                 '  ⏱ TIMEOUT>%.0fs' % a.timeout if timed_out else ''))
        if bad:
            print(out.rstrip(), flush=True)
            if timed_out:
                print('        该判据超过 %.0fs 未结束，已按 FAIL 记账并继续（'
                      '排查：单独跑它，或用 --only 缩小范围）' % a.timeout)

    print('\n%s' % ('-' * 72))
    print('##RE0_SUMMARY scripts=%d fail=%d seconds=%.0f'
          % (len(rows), fail, time.perf_counter() - t_all))
    print('%s' % ('全部通过 ✅' if fail == 0 else '有 %d 个未过 ❌' % fail))
    return fail


if __name__ == '__main__':
    sys.exit(main())
