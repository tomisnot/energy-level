# -*- coding: utf-8 -*-
"""检查公共设施（checklib）：供 core_selfcheck / pair_selfcheck 等复用。"""
import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class Checker:
    """PASS/FAIL/SKIP 计数器 + 汇总（退出码 = FAIL 数）。"""

    def __init__(self, title='selftest'):
        self.title = title
        self.n_pass = 0
        self.n_fail = 0
        self.n_skip = 0

    def check(self, name, cond, detail=''):
        if cond:
            self.n_pass += 1
            print(f'  [PASS] {name}')
        else:
            self.n_fail += 1
            print(f'  [FAIL] {name}   {detail}')

    def close(self, name, ours, ref, rtol=1e-6, atol=0.0):
        if abs(ours - ref) <= atol + rtol * abs(ref):
            self.n_pass += 1
            print(f'  [PASS] {name}  ({ours:.6g} vs {ref:.6g})')
        else:
            self.n_fail += 1
            print(f'  [FAIL] {name}  ours={ours:.8g} ref={ref:.8g}')

    def skip(self, name, why=''):
        self.n_skip += 1
        print(f'  [SKIP] {name}   {why}')

    def summary(self):
        print(f'[{self.title}]  PASS={self.n_pass}  FAIL={self.n_fail}'
              f'  SKIP={self.n_skip}')
        return self.n_fail == 0


def require_ai_server(port, why=''):
    """AI 接口没起来 → SKIP 退出（退出码 0），返回 port 便于链式使用。

    为什么要有这个：需要 MCP 端点的自检（`checks/ai_*test*.py`）过去一律
    `port = win._init_ai()` 真起 server —— 在**人的软件正在跑实验**时会去抢
    8765 / 覆盖 `.mcp-port`（2026-09-12 事故，见 docs/AI协同-分工与请示.md §六 #6）。
    现在 `start_server` 有 `EL_NO_AI_SERVER` 与"端口被活实例占用则拒启"两道闸，
    这些自检拿到 None 时应 **SKIP**（而不是把"没起来"误报成 FAIL）。
    """
    if port:
        return port
    import sys as _sys
    print('  [SKIP] AI 接口未启动（EL_NO_AI_SERVER 置位 / .mcp-port 指向活实例 / '
          '无 mcp SDK）→ 跳过本自检 %s' % why)
    _sys.exit(0)
