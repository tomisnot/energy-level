# -*- coding: utf-8 -*-
"""检查公共设施（checklib）：供 core_selfcheck / pair_selfcheck 等复用。

## 计数模型（2026-09-14 重设计：**不变式包**）
一个常被拆成十几条"碎片断言"的东西其实是**一个不变式**（例："存档往返后 `ms`
不丢"很容易写成"组数对 + 每组 ms 对 + 颜色对 + 引擎侧子能级数对…"）。
逐条计数会让汇总看起来有 677 项，而实际不变式只有 ~150 个（用户 2026-09-14：
"cheks 677 项太离谱了…重点在设立高效的检测，一个测出好几项"）。

于是 `Checker` 提供两种登记方式，**都算 1 项**：
  · `check(name, cond, detail)`    —— 单条断言
  · `group(name, subs, detail)`    —— **不变式包**：`subs = [(label, cond), ...]`，
    整包算 1 项；包内子条件（sub）单独累计，供诊断与"覆盖面没缩水"核对。

默认摘要打印 `PASS/FAIL/SKIP`（= 项数）与 `SUB`（= 子条件总数）；
`SELFTEST_VERBOSE=1` 时展开每个子条件。**失败时无论 verbose 都展开该包**
（否则诊断力会丢）。
"""
import os
import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class Checker:
    """PASS/FAIL/SKIP 计数器 + 不变式包 + 汇总（退出码 = FAIL 数）。"""

    def __init__(self, title='selftest', verbose=None):
        self.title = title
        self.n_pass = 0          # 项数（单条 check + 包，各算 1）
        self.n_fail = 0
        self.n_skip = 0
        self.n_sub = 0           # 子条件总数（覆盖面指标，不是项数）
        self.verbose = (bool(os.environ.get('SELFTEST_VERBOSE'))
                        if verbose is None else bool(verbose))

    # ---------- 登记 ----------
    def check(self, name, cond, detail=''):
        self.n_sub += 1
        if cond:
            self.n_pass += 1
            print(f'  [PASS] {name}')
        else:
            self.n_fail += 1
            print(f'  [FAIL] {name}   {detail}')

    def group(self, name, subs, detail=''):
        """**不变式包**：`subs = [(label, cond), ...]` → 整包算 1 项。

        子条件全真才算过；不过时**必然展开**（不依赖 verbose），便于定位。
        """
        subs = list(subs)
        bad = [(lab, c) for lab, c in subs if not c]
        self.n_sub += len(subs)
        if not bad:
            self.n_pass += 1
            print(f'  [PASS] {name}（{len(subs)} 子条件）')
        else:
            self.n_fail += 1
            print(f'  [FAIL] {name}（{len(bad)}/{len(subs)} 子条件不过）   '
                  f'{detail}')
        if bad or self.verbose:
            for lab, c in subs:
                print(f'      {"ok  " if c else "FAIL"} {lab}')

    def close(self, name, ours, ref, rtol=1e-6, atol=0.0):
        self.check(name, abs(ours - ref) <= atol + rtol * abs(ref),
                   f'ours={ours:.8g} ref={ref:.8g}')

    def skip(self, name, why=''):
        self.n_skip += 1
        print(f'  [SKIP] {name}   {why}')

    def summary(self):
        print(f'[{self.title}]  PASS={self.n_pass}  FAIL={self.n_fail}'
              f'  SKIP={self.n_skip}  SUB={self.n_sub}')
        return self.n_fail == 0


def require_ai_server(port, why=''):
    """AI 接口没起来 → SKIP 退出（退出码 0），返回 port 便于链式使用。

    为什么要有这个：需要 MCP 端点的自检（`checks/ai_*test*.py`）过去一律
    `port = win._init_ai()` 真起 server —— 在**人的软件正在跑实验**时会去抢
    8765 / 覆盖 `.mcp-port`（2026-09-12 事故，见 内部记录（未随仓发布，存档在仓外） §六 #6）。
    现在 `start_server` 有 `EL_NO_AI_SERVER` 与"端口被活实例占用则拒启"两道闸，
    这些自检拿到 None 时应 **SKIP**（而不是把"没起来"误报成 FAIL）。
    """
    if port:
        return port
    import sys as _sys
    print('  [SKIP] AI 接口未启动（EL_NO_AI_SERVER 置位 / .mcp-port 指向活实例 / '
          '无 mcp SDK）→ 跳过本自检 %s' % why)
    _sys.exit(0)
