# -*- coding: utf-8 -*-
"""精简回归入口（分档 + 可选并行）：日常自检。

    python checks\\selftest_all.py                # 全量（tier=full，默认串行，35 脚本 ~10.5 min）
    python checks\\selftest_all.py --tier fast    # 快筛档（28 脚本 ~4.5 min；--jobs 2 实测 140 s）
    python checks\\selftest_all.py --jobs 2       # 并行 2 档（见下方内存警告）

## 分档（2026-09-14 瘦身时按**实测耗时**分）

**fast 档（28 个脚本，约 4.5 分钟 / 532 项）**——哨兵/纯函数/单原子静态/轻 GUI：
    checks_sentinel.py            存活哨兵（语法 + src.* 导入 + 兼容导出 + BOM 类文本卫生）
    journal_size_cap_selfcheck.py journal 的 detail 体积闸（巨型 detail → 摘要，21 项）
    core_selfcheck.py             单原子静态（能级/偶极/AC 位移，22 项）
    params_registry_selfcheck.py  引擎参数注册表（默认/覆写/线程隔离/端到端，39 项）
    pair_mem_estimate_selfcheck.py 两体峰值内存粗估（13 项）
    ai_code_freshness_selfcheck.py 代码新鲜度（28 项）
    solve_max_step_hard_selfcheck.py 步长硬上限（13 项）
    port_file_ownership_selfcheck.py .mcp-port 归属权（6 项）
    ai_h_snapshot_guard_selfcheck.py H(t) 快照尺寸闸（9 项）
    ai_job_artifacts_selfcheck.py  作业成果自动落盘（json 摘要 + npz，10 项）
    run_store_selfcheck.py         作业登记表（38 项）
    ai_result_card_selfcheck.py    结果摘要 + 模型卡（含单体路径真断言，14 项）
    mw_pair_keep_selfcheck.py      微波对累积相位判据 + 三光子梯子（24 项）
    ai_codeact_tools_selfcheck.py  CodeAct 工具面（命名空间/参数/错误路径，10 项）
    ai_semantic_state_selfcheck.py 语义状态面（set_groups/list_levels/初态，28 项）
    ai_superposition_selfcheck.py  逐原子叠加态持久真值（22 项）
    ai_pair_superposition_selfcheck.py 两体纠缠初态 spec + 三引擎贯穿（51 项）
    gui_diag_summary_test.py       检视器摘要头与 core.diag 同源（6 项）
    gui_ai_job_indicator_test.py   AI 作业指示灯（7 项）
    gui_engine_lamps_test.py       engine.* 指示灯 + 双向绑定（14 项）
    gui_ellip_row_test.py          椭圆偏参数行（11 项）
    gui_cost_dialog_test.py        运行前成本对话框（8 项）
    gui_worker_lifetime_test.py    QThread 生命周期（10 项）
    gui_params_page_test.py        引擎参数页接线（13 项）
    gui_mode_switch_test.py        模式切换 + 检视器常驻 + 双向绑定（16 项）
    ai_diag_test.py                只读观测通道 get_diag（16 项）
    evolve_dissipative_selfcheck.py Lindblad 耗散引擎（18 项）
    mixed_arb_mw_selfcheck.py      mixed 任意波形微波（12 项）

**slow 档（7 个脚本）**——重演化 / 全窗口 / 长梯子（实测 2026-09-14，串行）：
    evolve_exact_selfcheck.py      单原子 Sambe（光子块 + 包络自适应，11 项，~93s）
    sublevel_trim_selfcheck.py     逐子能级裁剪（A–G 七段，33 项，~83s）
    mixed_nolight_selfcheck.py     mixed 无光场退化契约（10 项，~73s）
    ui_selftest.py                 主窗口/编辑器/TTL/动态页/存档恢复（57 项，~66s）
    two_field_pair_selfcheck.py    两场两体引擎（11 项，~39s）
    evolve_exact_pair_selfcheck.py 两体第一性引擎（11 项，~33s）
    pair_evolve_selfcheck.py       两体动态（pairinteraction 权威参考；未装则 SKIP）

## 并行（--jobs N）

脚本之间**无共享状态**（各自起进程、各自建窗口/引擎），故可并行；但每个重脚本
峰值内存可达 1 GB 量级（两体 dim 数千），**N 只在你确认机器空闲内存 ≥ 2 GB×N 时用**。
默认 N=1（串行）：长脚本的输出要实时可读，且**人的软件可能正在跑实验**。

    ⚠ 自检一律带 `EL_NO_AI_SERVER=1`（2026-09-12 事故后加，见 docs/AI协同-分工与请示.md
    §六 #6）：**绝不抢活实例的 8765 端口 / 覆盖 .mcp-port**。需要真 MCP 端点的
    `ai_selftest / ai_compute_test / ai_codeact_test / ai_chain_selftest /
    ai_params_slim_test / ai_runstore_mcp_test` 因此**不在本聚合里**，
    要跑它们请在**没有活实例**时手动单跑（脚本自己会 SKIP 或起独立端口）。

不在聚合里、按需单跑的独立自检：
    hard_vdd_pack_selfcheck.py     hard 的 V_dd 打包 + Θ 判据语义（9 项）
    route_auto_selfcheck.py        route_auto 判据（6 项）
    basis_convergence_selfcheck.py **T2(E1) 验收工具：观测量收敛**（慢 + 修好前预期
                                   FAIL）→ 刻意不进聚合
    gui_arb_mixed_wiring_test.py   GUI 接线（任意波形 × 三引擎，~140s；动
                                   ui_dynamic / evolve_worker 后请跑）
    e1_pair_forensics.py           E1（光学对丢失）取证/回归锚（7 项）
    ai_*（上列 6 个 MCP 端点自检）
    lindblad/其它 → `checks/_archive/`（需要时单跑）

## 结果表

每个脚本跑完打印一行 `—— <脚本> 完成，<秒>s（PASS/FAIL/SKIP）`，最后给一张
按档分组的汇总表 + 一行机器可读的 `##SELFTEST_SUMMARY ...`（供脚本/日志解析）。
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

# (脚本, 档) —— 顺序 = 时间序（快→慢），串行时可尽早看到红的
_PLAN = [
    ('checks_sentinel.py', 'fast'),
    ('journal_size_cap_selfcheck.py', 'fast'),
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
    ('gui_engine_lamps_test.py', 'fast'),
    ('gui_ellip_row_test.py', 'fast'),
    ('gui_cost_dialog_test.py', 'fast'),
    ('gui_worker_lifetime_test.py', 'fast'),
    ('gui_params_page_test.py', 'fast'),
    ('gui_mode_switch_test.py', 'fast'),
    ('ai_diag_test.py', 'fast'),
    ('evolve_dissipative_selfcheck.py', 'fast'),
    ('mixed_arb_mw_selfcheck.py', 'fast'),
    ('evolve_exact_selfcheck.py', 'slow'),
    ('sublevel_trim_selfcheck.py', 'slow'),
    ('ui_selftest.py', 'slow'),
    ('two_field_pair_selfcheck.py', 'slow'),
    ('mixed_nolight_selfcheck.py', 'slow'),
    ('evolve_exact_pair_selfcheck.py', 'slow'),
    ('pair_evolve_selfcheck.py', 'slow'),
]

_RE_SUM = re.compile(r'\[(\w+)\]\s+PASS=(\d+)\s+FAIL=(\d+)\s+SKIP=(\d+)')
_RE_NN = re.compile(r'(\d+)/(\d+)\s+PASS')
_RE_OK = re.compile(r'\[PASS\]|\[ ok \]|^\s*OK\b|结果：全部通过', re.M)
_RE_BAD = re.compile(r'\[FAIL\]|^\s*FAIL\b|结果：\d+ 项失败', re.M)
_LOCK = threading.Lock()


def _counts(text):
    """脚本输出 → (pass, fail, skip)；认不出来就给 (-1, -1, -1)。

    三级认法：① 脚本自报的 `[name] PASS=x FAIL=y SKIP=z`；② `N/M PASS`；
    ③ 逐条标记（`[PASS]` / `[ ok ]` / `OK` / `结果：全部通过`）——第三条是为了
    让"只打 OK/FAIL 不报总数"的老脚本（如 ai_h_snapshot_guard）也进汇总表，
    不然汇总会**系统性少算**断言数。
    """
    m = _RE_SUM.search(text)
    if m:
        return int(m.group(2)), int(m.group(3)), int(m.group(4))
    nn = _RE_NN.findall(text)
    if nn:
        p, t = max(((int(a), int(b)) for a, b in nn), key=lambda x: x[1])
        return p, max(0, t - p), 0
    n_ok = len(_RE_OK.findall(text))
    n_bad = len(_RE_BAD.findall(text))
    if n_ok or n_bad:
        return n_ok, n_bad, 0
    return -1, -1, -1


def _run_one(script, env, quiet):
    """跑一个脚本：实时回显（串行）或整段回显（并行）；→ (rc, sec, counts, tail)."""
    path = os.path.join(_HERE, script)
    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, '-u', path], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding='utf-8', errors='replace',
                         bufsize=1)
    lines = []
    for ln in p.stdout:
        lines.append(ln)
        if not quiet:
            with _LOCK:
                sys.stdout.write(ln)
                sys.stdout.flush()
    rc = p.wait()
    sec = time.perf_counter() - t0
    return rc, sec, _counts(''.join(lines)), lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['fast', 'full'], default='full')
    ap.add_argument('--jobs', type=int, default=1,
                    help='并行档数（默认 1=串行；见模块 docstring 的内存警告）')
    ap.add_argument('--only', default='', help='只跑名字含该子串的脚本')
    a = ap.parse_args()
    plan = [(s, t) for s, t in _PLAN
            if (a.tier == 'full' or t == 'fast') and (not a.only or a.only in s)]
    if not plan:
        print('没有匹配的脚本'); return 2
    jobs = max(1, min(a.jobs, len(plan)))
    env = dict(os.environ)
    env.setdefault('EL_NO_AI_SERVER', '1')
    env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    print('跑 %d 个脚本（tier=%s, jobs=%d）' % (len(plan), a.tier, jobs))

    results = [None] * len(plan)
    t_all = time.perf_counter()

    def _work(i):
        script = plan[i][0]
        if jobs == 1:
            print(f'\n===== {script} =====')
        t0 = time.perf_counter()
        rc, sec, cnt, lines = _run_one(script, env, quiet=(jobs > 1))
        if jobs > 1:
            with _LOCK:
                print(f'\n===== {script} =====（{sec:.0f}s, rc={rc}）')
                sys.stdout.write(''.join(lines))
                sys.stdout.flush()
        p_, f_, s_ = cnt
        print(f'—— {script} 完成，{sec:.0f}s'
              + (f'（PASS={p_} FAIL={f_} SKIP={s_}）' if p_ >= 0 else '（无计数行）'))
        results[i] = (script, plan[i][1], rc, sec, cnt)

    if jobs == 1:
        for i in range(len(plan)):
            _work(i)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(_work, range(len(plan))))

    tot = time.perf_counter() - t_all
    print('\n==== 汇总（tier=%s, jobs=%d）====' % (a.tier, jobs))
    print('%-38s %-6s %7s %6s %6s %6s' % ('script', 'tier', 'sec', 'pass',
                                          'fail', 'skip'))
    tp = tf = ts = 0
    rc_all = 0
    for script, tier, rc, sec, (p_, f_, s_) in results:
        rc_all |= rc
        tp += max(0, p_)
        tf += max(0, f_)
        ts += max(0, s_)
        print('%-38s %-6s %7.0f %6s %6s %6s'
              % (script, tier, sec,
                 p_ if p_ >= 0 else '-', f_ if f_ >= 0 else '-',
                 s_ if s_ >= 0 else '-'))
    print('##SELFTEST_SUMMARY scripts=%d pass=%d fail=%d skip=%d seconds=%.0f '
          'tier=%s jobs=%d rc=%d'
          % (len(results), tp, tf, ts, tot, a.tier, jobs, rc_all))
    print('==== 回归完成，%.0fs，%s ===='
          % (tot, '全部通过 ✅' if rc_all == 0 else '有失败 ❌'))
    return rc_all


if __name__ == '__main__':
    sys.exit(main())
