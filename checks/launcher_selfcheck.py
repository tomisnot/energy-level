# -*- coding: utf-8 -*-
"""看门人（launcher.py）状态机判据：交接决策的纯逻辑，不起真进程、不弹窗。

守"两模式收拢"的编排核心——**进程交接的时序正确性**（这是活体最容易翻车、
又最难手工复现的一段：terminate→等租约释放→spawn，顺序错就 writer_lease_held）：

  A. _read_request：请求文件的解析——无文件/非法值→None（不误触发），
     gui/ai/大小写/空白→归一化到 'gui'/'ai'。
  B. _write_state：当前模式落 .mode-state（供人/判据看），覆盖写。
  C. _pick_dsh_port：返回可绑定回环端口，且**永不选 3080**（官方 dsh 的端口）。
  D. _terminate_group + _await_exit：交接时序——先 terminate 每个存活进程、
     再等它真退出（poll 从 None 变非 None）才返回。**致死性**：若 _await_exit
     不等退出就返回（旧写法），spawn 新权威会撞 writer_lease_held——本组用
     mock 进程钉死"必须等到 poll 非 None"。

不起真 app_entry 子进程（那要建 cache、十几秒且需 EL 根）——本判据只测编排
**决策逻辑**；真进程链路由活体验收（交用户）。
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
import launcher                                            # noqa: E402

C = Checker('launcher')


class _FakeProc:
    """假进程：poll() 前 wait_none 次返回 None（还在跑），之后返回 exit_code。"""

    def __init__(self, wait_none=0, exit_code=0):
        self._left = wait_none
        self._code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        if self._left > 0:
            self._left -= 1
            return None
        return self._code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def main():
    tmp = tempfile.mkdtemp(prefix='launcher_')
    # 把看门人的路径常量指到临时目录（不污染仓根、判据间隔离）
    launcher.MODE_REQUEST = os.path.join(tmp, '.mode-request')
    launcher.MODE_STATE = os.path.join(tmp, '.mode-state')

    # ---- A. _read_request 解析 ----
    launcher._clear_request()
    a_none = launcher._read_request()
    with open(launcher.MODE_REQUEST, 'w', encoding='utf-8') as f:
        f.write('gui')
    a_gui = launcher._read_request()
    with open(launcher.MODE_REQUEST, 'w', encoding='utf-8') as f:
        f.write('  AI \n')                     # 大小写 + 空白
    a_ai = launcher._read_request()
    with open(launcher.MODE_REQUEST, 'w', encoding='utf-8') as f:
        f.write('banana')                       # 非法值
    a_bad = launcher._read_request()
    C.group('A _read_request：合法值归一化、非法/缺失不误触发', [
        ('无请求文件 → None', a_none is None),
        ('"gui" → gui', a_gui == 'gui'),
        ('"  AI " → ai（大小写+空白归一）', a_ai == 'ai'),
        ('非法值 "banana" → None（不切换）', a_bad is None),
    ])
    launcher._clear_request()
    C.check('A _clear_request 删请求文件（交接后不重复触发）',
            launcher._read_request() is None
            and not os.path.exists(launcher.MODE_REQUEST))

    # ---- B. _write_state ----
    launcher._write_state('ai')
    got = open(launcher.MODE_STATE, encoding='utf-8').read()
    launcher._write_state('gui')               # 覆盖写
    got2 = open(launcher.MODE_STATE, encoding='utf-8').read()
    C.group('B _write_state：当前模式落文件、覆盖写', [
        ('写 ai 生效', got.strip() == 'ai'),
        ('覆盖为 gui（不留旧值）', got2.strip() == 'gui'),
    ])

    # ---- C. _pick_dsh_port ----
    p = launcher._pick_dsh_port(preferred=4100, tries=20)   # 从 4100 起找可绑的
    # 验"永不选 3080"：从 3080 起应跳过它返回 3081+
    p2 = launcher._pick_dsh_port(preferred=3080, tries=3)
    C.group('C _pick_dsh_port：可绑定且永不选 3080（避开官方 dsh）', [
        ('返回有效端口（>1024）', isinstance(p, int) and p > 1024),
        ('从 3080 起也跳过 3080（返回 3081+）', p2 != 3080 and p2 >= 3081),
    ])

    # ---- D. 交接时序：terminate → 等退出（租约释放）----
    # 致死性核心：_await_exit 必须等 poll() 变非 None 才返回。用 wait_none=3 的
    # 假进程模拟"terminate 后还要几拍才真退出"，断言返回时它已"退出"。
    procs = [_FakeProc(wait_none=3), _FakeProc(wait_none=1), None]  # 含 None 防崩
    t0 = time.time()
    launcher._terminate_group(procs)
    term_ok = procs[0].terminated and procs[1].terminated
    launcher._await_exit(procs, timeout=5)
    waited = time.time() - t0
    # 都退出后再 await 应立即返回（不等满 timeout）
    t1 = time.time()
    launcher._await_exit(procs, timeout=5)
    immediate = time.time() - t1

    C.group('D 交接时序：先 terminate 再等真退出（防租约没放干净就 spawn）', [
        ('terminate 对每个存活进程都调了（None 跳过不崩）', term_ok),
        ('_await_exit 等到进程真退出才返回（耗时反映等待）', waited > 0.1),
        ('已退出的进程再 await 立即返回（不空等 timeout）', immediate < 0.5),
    ])
    # 超时兜底：永不退出的进程 → await 到 timeout 后 kill，不无限挂
    stuck = _FakeProc(wait_none=10 ** 9)
    launcher._terminate_group([stuck])
    t2 = time.time()
    launcher._await_exit([stuck], timeout=1)
    C.check('D 卡死进程：await 到 timeout 后 kill 兜底（不无限挂住基线）',
            stuck.killed and 0.8 < time.time() - t2 < 3.0)

    ok = C.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
