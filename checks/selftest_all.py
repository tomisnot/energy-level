# -*- coding: utf-8 -*-
"""精简回归入口：**全量（含 deep）≤3 分钟**（不变式包计数，子条件可展开）。

    python checks\\selftest_all.py --jobs 4         # 默认档 + 并行 4 ⇒ 实测 128-135 s
    python checks\\selftest_all.py                 # 默认档，串行（实测 358 s）
    python checks\\selftest_all.py --tier fast      # 提交前快筛（更少的脚本）
    python checks\\selftest_all.py --tier deep      # 只跑 deep 档（重对拍/长演化）
    python checks\\selftest_all.py --tier all --jobs 4   # 全量含 deep ⇒ 实测 165 s
    python checks\\selftest_all.py --verbose        # 展开每个不变式包的子条件
    python checks\\selftest_all.py --only ai_       # 只跑名字含该子串的

## 实测基准（本机，2026-09-14；`##SELFTEST_SUMMARY` 原样抄录）

| 档 | jobs | 脚本 | 项数(包) | 子条件 | 耗时 | 结果 |
|---|---|---|---|---|---|---|
| full（默认） | 4 | 31 | 183 | 661 | **128 / 135 / 134 s**（连跑 3 次） | 0 FAIL |
| full（默认） | 1 | 31 | 183 | 661 | **358 s** | 0 FAIL |
| all（含 deep） | 4 | 36 | 238 | 716 | **165 s** | 0 FAIL（1 SKIP） |

## 三个档（2026-09-14 用户裁决："动动脑子，哪些不用测…给我把全量测试时间压到 3 分钟"）

- **full（默认）**：日常必须绿的一切——哨兵/纯函数/单体静态/轻 GUI + 两个大守卫
  （`ui_selftest` 用户裁决项、`sublevel_trim` 裁剪不动物理）。目标 ≤180 s（`--jobs 4`）。
- **fast**：更小的子集（提交前快筛）。
- **deep**：**按需**——五个"引擎算得对"的**权威对拍/长演化**体检
  （`evolve_exact` / `mixed_nolight` / `two_field_pair` / `evolve_exact_pair` /
  `pair_evolve`，共约 260 s）。它们与 `engine_e2e_selfcheck.py`（默认档，~20 s）
  覆盖**同一批物理不变式**，但保留了各自引擎的**解析解/外部权威包**交叉验证——
  所以**不删，降为按需**：动过引擎、或发版前跑一次 `--tier all`。

## 计数模型：不变式包（"一个测出好几项"）

每个脚本打印 `[name] PASS=<项数> FAIL=… SKIP=… SUB=<子条件数>`：
**项数 = 不变式包数**（`Checker.group` 把同一不变式的十几条碎片合成 1 项），
`SUB` 是包内子条件总数（**覆盖面指标**——项数少了但 SUB 不能少）。
`--verbose` 或 `SELFTEST_VERBOSE=1` 展开子条件；包失败时**自动展开**。

## 并行与串行（`--jobs N`）

- 脚本间无共享状态，可并行；但重脚本峰值内存可达 1 GB ⇒ **确认空闲内存 ≥ 2GB×N 再用**。
- `_SERIAL_ONLY`（Qt/QThread 时序类）**先串行跑完**再并行其余：2026-09-14 使用者
  独立验收时抓到 `gui_worker_lifetime_test.py` 在 `--jobs 2` 下偶发假红
  （隔离复跑全绿）——它是 0xc0000409 事故的唯一守卫，**必须确定性**。
- 子进程一律带 `EL_NO_AI_SERVER=1` + `QT_QPA_PLATFORM=offscreen`：
  **绝不抢活实例的 8765 端口 / 覆盖 `.mcp-port`**（2026-09-12 事故，见
  `docs/AI协同-分工与请示.md` §六 #6）。需要真 MCP 端点的 6 个 `ai_*` 自检因此
  **不在聚合里**，要跑请在**没有活实例**时手动单跑（脚本自己会 SKIP）。

## 不在聚合里、按需单跑

    hard_vdd_pack_selfcheck.py     hard 的 V_dd 打包 + Θ 判据语义
    route_auto_selfcheck.py        route_auto 判据
    basis_convergence_selfcheck.py T2(E1) 验收工具（慢 + 修好前预期 FAIL）
    e1_pair_forensics.py           E1 取证/回归锚（7 项）
    gui_arb_mixed_wiring_test.py   GUI 接线（任意波形 × 三引擎，~140s）
    ai_selftest / ai_compute_test / ai_codeact_test / ai_chain_selftest /
    ai_params_slim_test / ai_runstore_mcp_test（MCP 端点契约；无活实例时单跑）
    checks/_archive/*              已封存，不进日常回归
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))

# (脚本, 档) —— 顺序 = 时间序（快→慢）
_PLAN = [
    ('checks_sentinel.py', 'fast'),
    ('journal_size_cap_selfcheck.py', 'fast'),
    ('engine_e2e_selfcheck.py', 'fast'),
    ('core_selfcheck.py', 'fast'),
    ('params_registry_selfcheck.py', 'fast'),
    ('pair_mem_estimate_selfcheck.py', 'fast'),
    ('ai_code_freshness_selfcheck.py', 'fast'),
    ('solve_max_step_hard_selfcheck.py', 'fast'),
    ('port_file_ownership_selfcheck.py', 'fast'),
    ('ai_h_snapshot_guard_selfcheck.py', 'fast'),
    ('ai_job_artifacts_selfcheck.py', 'fast'),
    ('run_store_selfcheck.py', 'fast'),
    ('ai_result_card_selfcheck.py', 'fast'),
    ('mw_pair_keep_selfcheck.py', 'fast'),
    ('ai_codeact_tools_selfcheck.py', 'fast'),
    ('ai_semantic_state_selfcheck.py', 'fast'),
    ('ai_superposition_selfcheck.py', 'fast'),
    ('ai_pair_superposition_selfcheck.py', 'fast'),
    ('gui_diag_summary_test.py', 'fast'),
    ('gui_ai_job_indicator_test.py', 'fast'),
    ('gui_worker_lifetime_test.py', 'fast'),
    ('ai_diag_test.py', 'fast'),
    ('evolve_dissipative_selfcheck.py', 'fast'),
    ('mixed_arb_mw_selfcheck.py', 'fast'),
    ('gui_params_page_test.py', 'slow'),
    ('gui_mode_switch_test.py', 'slow'),
    ('gui_engine_lamps_test.py', 'slow'),
    ('gui_ellip_row_test.py', 'slow'),
    ('gui_cost_dialog_test.py', 'slow'),
    ('sublevel_trim_selfcheck.py', 'slow'),
    ('ui_selftest.py', 'slow'),
]

# **按需深挖档**（不进默认 full）：五个"引擎算得对"的权威对拍/长演化体检。
# 与 engine_e2e_selfcheck.py 覆盖同一批物理不变式，但那边的"解析解/外部权威包"
# 交叉验证不可替代 ⇒ 不删，降为按需（动过引擎或发版前跑 --tier all）。
_DEEP = {
    'evolve_exact_selfcheck.py',
    'mixed_nolight_selfcheck.py',
    'two_field_pair_selfcheck.py',
    'evolve_exact_pair_selfcheck.py',
    'pair_evolve_selfcheck.py',
}

# 只能串行的脚本（Qt/QThread 时序敏感）：jobs>1 时先串行跑完它们。
_SERIAL_ONLY = {'gui_worker_lifetime_test.py'}

_RE_SUM = re.compile(r'\[(\w+)\]\s+PASS=(\d+)\s+FAIL=(\d+)\s+SKIP=(\d+)'
                     r'(?:\s+SUB=(\d+))?')
_RE_NN = re.compile(r'(\d+)/(\d+)\s+PASS')
_RE_OK = re.compile(r'\[PASS\]|\[ ok \]|^\s*OK\b|结果：全部通过', re.M)
_RE_BAD = re.compile(r'\[FAIL\]|^\s*FAIL\b|结果：\d+ 项失败', re.M)
_LOCK = threading.Lock()


def _counts(text):
    """脚本输出 → (项数, FAIL, SKIP, 子条件数)；认不出来给 (-1,-1,-1,-1)。

    三级认法：① 脚本自报 `[name] PASS=x FAIL=y SKIP=z [SUB=s]`；② `N/M PASS`；
    ③ 逐条标记（`[PASS]`/`[ ok ]`/`OK`）——第三条让"只打 OK 不报总数"的老脚本
    也进汇总，不然会**系统性少算**。
    """
    m = _RE_SUM.search(text)
    if m:
        sub = int(m.group(5)) if m.group(5) else int(m.group(2))
        return int(m.group(2)), int(m.group(3)), int(m.group(4)), sub
    nn = _RE_NN.findall(text)
    if nn:
        p, t = max(((int(a), int(b)) for a, b in nn), key=lambda x: x[1])
        return p, max(0, t - p), 0, t
    n_ok = len(_RE_OK.findall(text))
    n_bad = len(_RE_BAD.findall(text))
    if n_ok or n_bad:
        return n_ok, n_bad, 0, n_ok
    return -1, -1, -1, -1


def _run_one(script, env, quiet, timeout_s=600.0):
    """跑一个脚本：实时回显（串行）或整段回显（并行）；→ (rc, sec, counts, lines).

    `timeout_s` 是**看门狗**：脚本挂死时杀掉它并记 FAIL_T（否则并行档会被一个
    挂死的脚本永久拖住——"≤3 分钟"的承诺也就没了）。
    """
    path = os.path.join(_HERE, script)
    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, '-u', path], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding='utf-8', errors='replace',
                         bufsize=1)
    killed = {'v': False}

    def _kill():
        killed['v'] = True
        try:
            p.kill()
        except Exception:                                      # noqa: BLE001
            pass

    wd = threading.Timer(float(timeout_s), _kill)
    wd.daemon = True
    wd.start()
    lines = []
    try:
        for ln in p.stdout:
            lines.append(ln)
            if not quiet:
                with _LOCK:
                    sys.stdout.write(ln)
                    sys.stdout.flush()
        rc = p.wait()
    finally:
        wd.cancel()
    if killed['v']:
        lines.append('\n[看门狗] 超过 %.0f s 未结束 → 已杀（记为 FAIL）\n'
                     % float(timeout_s))
        rc = -9
    return rc, time.perf_counter() - t0, _counts(''.join(lines)), lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['fast', 'full', 'deep', 'all'],
                    default='full')
    ap.add_argument('--jobs', type=int, default=1,
                    help='并行档数（默认 1=串行；见 docstring 的内存警告）')
    ap.add_argument('--only', default='', help='只跑名字含该子串的脚本')
    ap.add_argument('--verbose', action='store_true',
                    help='展开不变式包的子条件（等价 SELFTEST_VERBOSE=1）')
    ap.add_argument('--timeout', type=float, default=600.0,
                    help='单脚本看门狗（秒，默认 600）：超时杀进程并记 FAIL')
    a = ap.parse_args()

    def _in_tier(s, t):
        if a.tier == 'all':
            return True
        if a.tier == 'deep':
            return s in _DEEP
        if s in _DEEP:
            return False                      # deep 脚本只在 all/deep 档跑
        if a.tier == 'fast':
            return t == 'fast'
        return True                           # full = 除 deep 外全部

    plan = [(s, t) for s, t in _PLAN + [(s, 'deep') for s in sorted(_DEEP)]
            if _in_tier(s, t) and (not a.only or a.only in s)]
    missing = [s for s, _t in plan if not os.path.isfile(os.path.join(_HERE, s))]
    if missing:
        print('!! 清单里的脚本不存在（清单该更新了）：%s' % missing)
        return 2
    if not plan:
        print('没有匹配的脚本')
        return 2
    jobs = max(1, min(a.jobs, len(plan)))
    env = dict(os.environ)
    env.setdefault('EL_NO_AI_SERVER', '1')
    env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    if a.verbose:
        env['SELFTEST_VERBOSE'] = '1'
    print('跑 %d 个脚本（tier=%s, jobs=%d, verbose=%s）'
          % (len(plan), a.tier, jobs, a.verbose))

    results = [None] * len(plan)
    t_all = time.perf_counter()

    def _work(i, force_serial=False):
        script = plan[i][0]
        single = (jobs == 1 or force_serial)
        if single:
            print(f'\n===== {script} =====')
        t0 = time.perf_counter()
        rc, sec, cnt, lines = _run_one(script, env, quiet=not single,
                                       timeout_s=a.timeout)
        if not single:
            with _LOCK:
                print(f'\n===== {script} =====（{sec:.0f}s, rc={rc}）')
                sys.stdout.write(''.join(lines))
                sys.stdout.flush()
        p_, f_, s_, sub = cnt
        print(f'—— {script} 完成，{sec:.0f}s'
              + (f'（项={p_} 子条件={sub} FAIL={f_} SKIP={s_}）' if p_ >= 0
                 else '（无计数行）'))
        results[i] = (script, plan[i][1], rc, sec, cnt)

    serial_only = [i for i, (s, _t) in enumerate(plan) if s in _SERIAL_ONLY]
    if jobs == 1:
        for i in range(len(plan)):
            _work(i)
    elif serial_only:
        print('串行优先（不并行）：%s' % ', '.join(plan[i][0] for i in serial_only))
        for i in serial_only:
            _work(i, force_serial=True)
        rest = [i for i in range(len(plan)) if i not in serial_only]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(lambda i: _work(i, force_serial=True), rest))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(_work, range(len(plan))))

    tot = time.perf_counter() - t_all
    print('\n==== 汇总（tier=%s, jobs=%d）====' % (a.tier, jobs))
    print('%-38s %-5s %6s %5s %6s %5s %5s'
          % ('script', 'tier', 'sec', '项', '子条件', 'FAIL', 'SKIP'))
    tp = tf = ts = tsub = 0
    rc_all = 0
    for script, tier, rc, sec, (p_, f_, s_, sub) in results:
        rc_all |= rc
        tp += max(0, p_)
        tf += max(0, f_)
        ts += max(0, s_)
        tsub += max(0, sub)
        print('%-38s %-5s %6.0f %5s %6s %5s %5s'
              % (script, tier, sec,
                 p_ if p_ >= 0 else '-', sub if sub >= 0 else '-',
                 f_ if f_ >= 0 else '-', s_ if s_ >= 0 else '-'))
    print('##SELFTEST_SUMMARY scripts=%d items=%d subs=%d fail=%d skip=%d '
          'seconds=%.0f tier=%s jobs=%d rc=%d'
          % (len(results), tp, tsub, tf, ts, tot, a.tier, jobs, rc_all))
    print('==== 回归完成，%.0fs，%s ===='
          % (tot, '全部通过 ✅' if rc_all == 0 else '有失败 ❌'))
    return rc_all


if __name__ == '__main__':
    sys.exit(main())
