# -*- coding: utf-8 -*-
"""Re0 重构 · 物理基线跑：把判据跑一遍并汇总。

用法：
    python run_checks.py                # 全档（约 590s；跳过 18 分钟的第一性原理基准）
    python run_checks.py --core         # **内循环档（约 20s）**：只跑 3 秒以内的结构/契约/守卫判据
    python run_checks.py --fast         # 提交前档（约 150s）：跑 15 秒以内的判据
    python run_checks.py --slow         # 含慢基准（mw_bench_firstprinciples，约 18 分钟）
    python run_checks.py --only sambe
    python run_checks.py --list
    python run_checks.py --list --core  # 看内循环档包含哪些
    python run_checks.py --dir old/checks

**为什么要分档**：全档实测 554s 计算时间，但分布极不均——前 5 个判据占一半
（90/80/72/45/36s），后 36 个加起来只有 72s。开发内循环需要的是"改一行马上知道有没有
闯祸"，不是每次等十分钟；全档留给"提交前/收工前"。
两档按「判据名 + 单判据实测耗时」自动选出（耗时数据见 `scripts/check_profile.py`
→ `runtime/check_profile.csv`），**不需要人工维护名单**：新写的秒级判据自动进档。

判据一律以**独立进程**跑（与原来 selftest_all 同一纪律：脚本间无共享状态）。
`--timeout`（默认 300s）给**每个**判据封顶——防止一个卡死的判据把整条基线无限期挂住
（实测踩过：一个判据卡到 843s，让整次基线失去意义）。
退出码 = FAIL 的脚本数（0 = 全绿）。

判据目录：默认依次找 `checks/`、`old/checks/`（沙盒被重排过，两处都认，
同一批判据的两份拷贝 → 只跑第一处）。

⚠ 2026-09-24 代替 v1：一批测 v1 库内核（Session/Bus/Ops/registry/cockpit/
Fields/interlock/gc）的判据已退役，处置清单见 mecha 仓 ADR
`docs/v2/notes/implemented/2026-09-24-代替v1-判据迁移与落地.md`；
v2 库本身的守卫在 mecha 仓 `checks/run_all.py`，不跨仓重复。
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
# 慢基准：默认跳过（约 18 分钟），--slow 才跑
SLOW = {'mw_bench_firstprinciples.py'}

# =====================================================================
# 分档（--core / --fast）：给开发内循环用的秒级判据集
#
# **唯一的判据是"实测耗时"**，登记在下面的 FAST_KNOWN_SEC 里（来源
# `scripts/check_profile.py` → `runtime/check_profile.csv`）。
#
# ⚠ 这里原先还有一道 `FAST_KINDS`（"名字命中 `_selfcheck`/`guard`/`contract`…"）——
# 实测发现它是**假门槛**：48 个判据**全部**命中（都叫 `*_selfcheck`），真正在起作用的
# 只有耗时阈值。留着它只会让人以为"按名字筛过了"。已删除，让机制说真话。
#
# 想让新判据进档：跑一次 `scripts/check_profile.py` 量出它的耗时，填进 FAST_KNOWN_SEC。
# **未登记的判据不进档**（保守：宁可漏跑，不可拖慢内循环）——代价是它只会在全档里跑，
# 所以新判据一写完就该来登记。
# =====================================================================
# 两档（按单判据实测耗时切）：内循环 / 提交前。
TIER_CORE_SEC = 3.0       # --core：秒级，改一行就跑
TIER_FAST_SEC = 15.0      # --fast：提交前
# 单判据实测耗时（秒；2026-09-22 由 scripts/check_profile.py 测得）。
FAST_KNOWN_SEC = {
    'docs_consistency_selfcheck.py': 0.4,
    'checks_sentinel.py': 0.5,
    'cleanliness_guard_selfcheck.py': 1.6,
    'field_loss_selfcheck.py': 1.7,
    'human_control_selfcheck.py': 1.7,
    'authority_selfcheck.py': 2.6,
    'engine_contract_selfcheck.py': 2.9,
    'runspec_contract_selfcheck.py': 2.4,
    'field_loss_engine_selfcheck.py': 2.7,
    'ops_threadsplit_selfcheck.py': 2.6,
    'solve_max_step_hard_selfcheck.py': 1.1,
    'mw_pair_keep_selfcheck.py': 2.1,
    'gui_bind_selfcheck.py': 1.9,
    'manybody_m1_ed_selfcheck.py': 1.1,
    'sweep_gui_selfcheck.py': 6.4,
    'gui_run_worker_selfcheck.py': 6.6,
    'mixed_arb_mw_selfcheck.py': 6.9,
    'hard_vdd_pack_selfcheck.py': 4.7,
    'ai_h_snapshot_guard_selfcheck.py': 2.3,
    'gui_app_entry_selfcheck.py': 8.3,
    'gui_run_flow_selfcheck.py': 10.0,
    'drive_wave_selfcheck.py': 10.4,
    'launcher_selfcheck.py': 1.5,
    'capability_map_selfcheck.py': 1.6,
    'hard_dedup_selfcheck.py': 11.0,
    'gui_inspector_selfcheck.py': 11.2,
    'manybody_integration_selfcheck.py': 11.3,
    'pair_evolve_selfcheck.py': 11.8,
    # 新增判据请在此登记**实测**耗时（跑 scripts/check_profile.py 现测），否则它不会进
    # --fast/--core 档——内循环就看不到它的红绿，只能等全档（数分钟）才发现。
    'narrow_pulse_env_selfcheck.py': 5.1,
    # 2026-09-25 现测 0.5-0.6s（取大）。监控端点契约门（框架化迁移的前置夹具）——
    # 必须进 --core/--fast：迁移时要"改一行马上知道面板契约破没破"。
    'cockpit_contract_selfcheck.py': 0.6,
}


def fast_lane(scripts, max_sec):
    """→ 已登记且实测耗时 < max_sec 的判据。

    判据**只有一个**：FAST_KNOWN_SEC 里的实测耗时。未登记的不进档（保守）。
    想调整档位粒度：改 TIER_CORE_SEC / TIER_FAST_SEC，或补登新判据的实测耗时。
    """
    return [s for s in scripts
            if FAST_KNOWN_SEC.get(s) is not None and FAST_KNOWN_SEC[s] < max_sec]

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
    ap.add_argument('--core', action='store_true',
                    help='内循环档：只跑 3 秒以内的结构/契约/守卫判据（约 20s）')
    ap.add_argument('--fast', action='store_true',
                    help='提交前档：跑 15 秒以内的判据（约 150s）')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--slow', action='store_true',
                    help='含慢基准（mw_bench_firstprinciples，约 18 分钟）')
    ap.add_argument('--timeout', type=float, default=300.0,
                    help='单个判据墙钟上限（秒，默认 300）；超时记 FAIL，不挂整条基线')
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
    if a.core:
        scripts = fast_lane(scripts, TIER_CORE_SEC)
    elif a.fast:
        scripts = fast_lane(scripts, TIER_FAST_SEC)
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
    # v2 判据的 GUI 装配（MainWindow→make_v2_bridge）写临时数据根，
    # 不污染仓内权威 journal（v1 时代 RE0_GUI_INPROC+临时 root 的同款纪律）。
    if not env.get('RE0_V2_DATA_ROOT'):
        import tempfile
        env['RE0_V2_DATA_ROOT'] = tempfile.mkdtemp(prefix='re0v2-checks-')

    print('判据目录：%s' % CHECKS)
    _tier = '（内循环档 --core）' if a.core else ('（提交前档 --fast）' if a.fast else '')
    print('跑 %d 个判据%s%s\n' % (len(scripts), _tier,
                                  '' if a.slow else '（已跳过慢基准）'))
    if not a.wiring:
        for s, why in sorted(WIRING.items()):
            print('  [跳过·接线] %-38s %s' % (s, why))
        print()
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
            tail = [l for l in out.splitlines() if 'FAIL' in l][:4]
            for l in tail:
                print('        %s' % l.strip()[:110])
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
