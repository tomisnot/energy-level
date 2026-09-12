# -*- coding: utf-8 -*-
"""精简回归入口：核心自检（约 4 分钟）。

    core_selfcheck.py              单原子静态（能级/偶极/AC 位移，22 项）
    evolve_dissipative_selfcheck.py  Lindblad 耗散引擎（解析衰变/时变/退相/
                                     ac 约定回归/numba≡numpy，18 项）
    evolve_exact_selfcheck.py      第一性原理含时引擎（Sambe 光子块 + 包络
                                   自适应，解析/参考/封存权威交叉，11 项）
    evolve_exact_pair_selfcheck.py 两体第一性引擎（完整 V_dd 无 secular，
                                   交换/直积极限/vdW/耗散/收敛，11 项）
    two_field_pair_selfcheck.py    两场两体引擎（11 项）
    mw_pair_keep_selfcheck.py      微波对累积相位判据 + 三光子梯子回归锚
                                   （24 项，~5s；t79 加的防"静默零物理"闸）
    params_registry_selfcheck.py   引擎参数注册表（默认/覆写/线程隔离/
                                   端到端覆写生效 + 链路契约，39 项）
    mixed_nolight_selfcheck.py     mixed 无光场退化契约（10 项，~80s）
    mixed_arb_mw_selfcheck.py      mixed 任意波形微波（12 项）
    gui_params_page_test.py        引擎参数页接线（页面↔快照↔引擎，11 项）
    gui_mode_switch_test.py        模式切换 + 检视器常驻 + 参数双向绑定（16 项）
    gui_engine_lamps_test.py       engine.* 指示灯 + 双向绑定 + 假覆写防护（14 项）
    ai_diag_test.py                只读观测通道 get_diag（摘要/明细/错误路径，16 项）
    gui_cost_dialog_test.py        运行前成本对话框（进度框遮挡回归，8 项）
    gui_ellip_row_test.py          椭圆偏参数行（平时隐藏/选后出现，10 项）
    ui_selftest.py                 UI 回归（主窗口/编辑器/TTL/动态页/存档恢复）
    pair_evolve_selfcheck.py       两体动态（pairinteraction 权威参考；
                                    未装 pairinteraction 时自动 SKIP）

不在聚合里、按需单跑的独立自检：
    hard_vdd_pack_selfcheck.py     hard 的 V_dd 打包 + Θ 判据语义（9 项）
    route_auto_selfcheck.py        route_auto 判据（6 项）
    gui_arb_mixed_wiring_test.py   GUI 接线（任意波形 × 三引擎，10 项，~140s
                                   ——慢，按需跑；但它能抓"只在 GUI 路径出现"
                                   的错，改动 ui_dynamic / evolve_worker 后请跑）
    ai_selftest / ai_compute_test / ai_codeact_test   AI 协同三件套
                                   （需 mcp/uvicorn；当前被 mcp 2.2 的
                                   stateless-HTTP 行为阻塞）

其余自检（两体 pair/unified、STIRAP/拉曼、pulse 模块、Floquet 交叉验证、
单原子 evolve）已归档到 checks/_archive/，需要时单独运行，不进日常回归。
（2026-09-10 清理：`hard_numba_selfcheck` / `two_field_selfcheck` /
`evolve_matrix` / `evolve_rwa_bench` / `acceptance_state` 因导入或 API 断裂
（引用 `src.core.evolve`、`src.pulse` 等已删路径）或所基准的 floquet/pwc
算法已封存，一并迁入 `_archive/`；判断依据 = 实际执行退出码。
⚠ `mw_bench_firstprinciples.py` **保留在 checks/**——它不是独立自检，而是
`ui_selftest` 的场景构造依赖（`mb.build_s1`）。）

用法：
  python checks\\selftest_all.py       # 全量（约 4 分钟）
"""
import os
import sys
import time
import subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = ['checks_sentinel.py',        # 存活哨兵（秒级，最先跑：防脚本腐烂）
            'core_selfcheck.py', 'evolve_dissipative_selfcheck.py',
            'evolve_exact_selfcheck.py', 'evolve_exact_pair_selfcheck.py',
            'two_field_pair_selfcheck.py',
            'mw_pair_keep_selfcheck.py',
            'params_registry_selfcheck.py',
            'mixed_nolight_selfcheck.py',
            'mixed_arb_mw_selfcheck.py',
            'gui_params_page_test.py',
            'gui_mode_switch_test.py',
            'ai_codeact_tools_selfcheck.py',
    'ai_semantic_state_selfcheck.py',
    'gui_ai_job_indicator_test.py',
    'ai_result_card_selfcheck.py',
    'gui_basis_presets_test.py',
            'gui_engine_lamps_test.py',
            'ai_diag_test.py',
            'gui_cost_dialog_test.py',
            'gui_ellip_row_test.py',
            'ui_selftest.py',
            'pair_evolve_selfcheck.py']


def main():
    rc = 0
    t_all = time.perf_counter()
    for s in _SCRIPTS:
        print(f'\n===== {s} =====')
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, '-u', os.path.join(_HERE, s)])
        rc |= r.returncode
        print(f'—— {s} 完成，{time.perf_counter() - t0:.0f}s')
    print(f'\n==== 回归完成，{time.perf_counter() - t_all:.0f}s，'
          f'{"全部通过 ✅" if rc == 0 else "有失败 ❌"} ====')
    sys.exit(rc)


if __name__ == '__main__':
    main()
