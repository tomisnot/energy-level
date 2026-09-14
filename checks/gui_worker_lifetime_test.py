# -*- coding: utf-8 -*-
"""自检：QThread 生命周期——收尾时**不得**丢掉仍在运行的 worker 的最后一根引用。

## 事故（2026-09-12，实测证据）
AI 提交的两体 job20（13 组、dim=7056、t_max=150/npts=301）跑完时进程**静默消失**
（Windows：`Application Error` … faulting module Qt5Core.dll，exception **0xc0000409**
= fail-fast；WER **BEX64**）；应用 stderr 里最后一条 Qt 原生消息是
```
QThread: Destroyed while thread is still running
```
`.mcp-port` 未被重写 ⇒ `stop_server()` 从未被调用 ⇒ 不是正常关窗路径，是崩溃。
结果随进程一起丢（"成果全丢"）。

## 根因（代码级）
`done` 是在 worker 线程的 `run()` **内部** emit 的（queued 投递到主线程）：
```python
# evolve_worker.EvolveWorkerPair.run()
res = evolve_mixed_pair(...)
self.done.emit(res, None)        # ← 主线程此刻可能已经开始处理这个槽
```
主线程处理 `_on_pair_evolve_done` → 第一句就是 `self._finish_run_ui()`，而它原来
`self._worker = None` —— 那是 QThread 的**唯一**强引用（worker 以 `parent=None` 创建，
没有任何别处持有）⇒ 引用归零 → QThread 包装被销毁，而此时 worker 线程还在退出
（要退出 `_P.override` 上下文、释放大结果数组）→ Qt 打印上面那句并 **fail-fast**。
**结果越大的作业越容易输掉这场竞速**（释放 ~100 MB 数组要时间），这解释了为什么
20:03–20:09 的 19 个单体小作业都没事、而 dim=7056 的两体 job20 死了。

## 修法
`_finish_run_ui` 在丢掉引用前先看 `isRunning()`：还在跑 → 放进
`_retired_workers`（强引用），并接 `finished` → `_sweep_retired_workers()`；
只有线程**确实退出**后才丢引用。

## 本自检（2026-09-14 重写：**与负载无关**）
旧版用 `sleep(0.4)` + 「轮询等 2 s / 5 s」猜时序 ⇒ 在 `--jobs 2`（或机器上有别的
重进程时）会偶发假红：主线程可能在"看到 `isRunning()==True`"与"调 `_finish_run_ui()`"
之间被抢占超过 0.4 s，于是线程**已经退出**，第 3 条断言（"仍在跑的 worker 被强引用住"）
就挂了——**那是自检的时序假设错了，不是代码坏了**。

现在合成 worker 的 `run()` **阻塞在一个 `threading.Event` 上**，结束时刻由测试自己
放行 ⇒ "调用 `_finish_run_ui()` 时线程仍在 run() 里"是**结构性成立**的，与负载无关；
所有等待都改成**等可判定条件**（`started` 事件 / `finished` 已投递 / `isFinished()`），
超时给到 30 s 只用于"真坏了就别挂死"，不再当判据。

断言（13 条）：
  0  面板有 `_retired_workers` 与 `_sweep_retired_workers`（修已就位）
  1  线程已启动且**已进入 `run()`**（`entered` 事件，非轮询猜测）
  2  `_finish_run_ui()` 后 `_worker` 已清空（UI 语义不变）
  3  仍在跑的 worker 被 `_retired_workers` 强引用住
  3b 此刻它**确实**还在跑（`isRunning()`）——证明第 3 条不是空转
  4  Qt 消息流里**没有** `Destroyed while thread is still running`
  5  放行后线程真的退出（`finished` 已投递 + `isFinished()` + `isRunning()==False`）
  6  退出后引用在一次 `_sweep_retired_workers()` 后被释放（= 生产里的机制：
     `finished` 槽与"下次运行开始"都会调它；**必须 `isRunning()==False` 才丢**）
  6b **反向守卫**：把"仍在跑"的 worker 放进 `_retired_workers` 再 sweep ⇒ 必须**保留**
     （引用一丢就是 0xc0000409 那类崩溃；这条守住"只丢确实不跑的"）
  7  全程无 QThread 销毁告警
  8a 第二轮也已启动（`entered`）
  8  连续两轮后 `_retired_workers` 已清空（无堆积）
  9  第二轮退出后**仍无**销毁告警（两轮都覆盖）

用法：python checks/gui_worker_lifetime_test.py（offscreen；构窗失败/MemoryError → SKIP）
⚠ 本脚本被 `selftest_all.py` 标为 `_SERIAL_ONLY`：`--jobs N` 时**先串行跑它**。
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402

C = Checker('gui_worker_lifetime')
# ------------------- 计数模型：不变式包（见 checks/checklib.py）-------------------
# group(...) 开一个包，其后相邻的 check(...) 全部变成该包的**子条件**：整包只计 1
# 项，子条件逐个进 SUB=（覆盖面不缩水：SUB == 改前逐条计数的项数）。原断言名与判据
# 表达式一律原样保留——只是不再各自计 1 项；不过时整包自动展开子条件。
_PKG = None


class _Pkg:
    """一个不变式包（先缓冲子条件，关闭时一次性交给 C.group 登记）。"""

    def __init__(self, name, detail=''):
        self.name, self.detail, self.subs = name, detail, []

    def flush(self):
        C.group(self.name, self.subs, self.detail)


def group(name, detail=''):
    """开新的不变式包：此后 check() 的断言都记入它（直到下一个 group）。"""
    global _PKG
    if _PKG is not None:
        _PKG.flush()
    _PKG = _Pkg(name, detail)


def flush():
    """关闭最后一个包（本文件收尾）。"""
    global _PKG
    if _PKG is not None:
        _PKG.flush()
        _PKG = None


def check(name, cond, detail=''):
    """薄包装：判据表达式原样，只登记为**当前包的子条件**（detail 作现场值）。"""
    if _PKG is None:
        raise RuntimeError(f'check() 出现在任何 group() 之前：{name}')
    _PKG.subs.append((name + (('  | ' + str(detail)) if detail else ''),
                      bool(cond)))
    return bool(cond)
_MSGS = []
_WAIT_S = 30.0                  # 只用于"真坏了别挂死"，不是判据


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    try:
        from PyQt5.QtCore import QThread, qInstallMessageHandler  # noqa: E402
        from PyQt5.QtWidgets import QApplication                  # noqa: E402
        import app_entry as M                                     # noqa: E402
    except Exception as e:                                        # noqa: BLE001
        print('  SKIP 环境不可用：%s' % e)
        return 0
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        win = M.MainWindow()
    except MemoryError as e:
        print('  SKIP 内存不足（软件可能正跑着）：%s' % e)
        return 0
    except Exception as e:                                        # noqa: BLE001
        print('  SKIP 构窗失败：%s' % e)
        return 0

    def _handler(mtype, ctx, msg):
        _MSGS.append(str(msg))

    old = qInstallMessageHandler(_handler)

    class _HoldWorker(QThread):
        """合成 worker：`run()` 阻塞到测试放行 ⇒ **结束时刻由测试决定**。

        这是本自检"与负载无关"的关键：不再用 `sleep(0.4)` 猜"调用收尾时线程还在不在"，
        而是让 `run()` 根本不返回，直到测试明确 `release.set()`。
        """

        def __init__(self):
            super().__init__(None)          # 与真实 worker 一致：parent=None
            self.entered = threading.Event()   # 已进入 run()
            self.release = threading.Event()   # 测试放行
            self.finished_seen = threading.Event()

        def run(self):
            self.entered.set()
            self.release.wait(_WAIT_S)
            return

    def _wait(pred, why, timeout_s=_WAIT_S):
        """等**可判定条件**成立（不是等固定时长）；超时只用于防挂死。"""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            app.processEvents()
            if pred():
                return True
            time.sleep(0.001)
        app.processEvents()
        return bool(pred())

    def _mk(worker):
        """起一个 worker 并等它**确实进了 run()**（事件驱动，非轮询猜测）。"""
        worker.finished.connect(worker.finished_seen.set)
        worker.start()
        return _wait(lambda: worker.entered.is_set() and worker.isRunning(),
                     '进入 run()')

    def _has_warn():
        return any('Destroyed while thread is still running' in m for m in _MSGS)

    def _running(x):
        """`isRunning()` 的容错版（C++ 对象已删 → RuntimeError）。"""
        try:
            return x is not None and x.isRunning()
        except RuntimeError:
            return False

    dyn = win.dynamic
    w = w2 = None
    try:
        group('A 修已就位 + 首轮 worker 已进入 run()（entered 事件）')
        check('0 面板有 _retired_workers 与 _sweep_retired_workers（修已就位）',
                hasattr(dyn, '_retired_workers')
                and callable(getattr(dyn, '_sweep_retired_workers', None)))

        w = _HoldWorker()
        dyn._worker = w                     # 与真实路径一样：面板持有唯一强引用
        check('1 线程已启动且已进入 run()（entered 事件，非轮询猜测）', _mk(w))

        # 【关键】模拟 done 槽在 run() 还没返回时执行收尾
        #   —— run() 阻塞在 release 上 ⇒ "此刻仍在跑"是结构性事实
        dyn._finish_run_ui()
        app.processEvents()
        group('B 收尾竞速：仍在跑的 worker 被 _retired_workers 强引用住')
        check('2 收尾后 _worker 已清空（UI 语义不变）', dyn._worker is None)
        check('3 仍在跑的 worker 被 _retired_workers 强引用住（不丢最后一根引用）',
                w in (dyn._retired_workers or []),
                str(len(dyn._retired_workers or [])))
        check('3b 此刻它确实还在跑（第 3 条不是空转）', w.isRunning())
        check('4 没有打印 "QThread: Destroyed while thread is still running"',
                not _has_warn(), str(_MSGS[-3:]))

        # 放行 → 线程真的结束（等 finished 投递 + isFinished()，与负载无关）
        w.release.set()
        ok_done = _wait(lambda: w.finished_seen.is_set() and not w.isRunning()
                        and w.isFinished(), '线程退出')
        group('C 退出后才释放引用（含反向守卫：只丢确实不跑的）+ 无销毁告警')
        check('5 放行后线程真的退出（finished 已投递 + isFinished）',
                ok_done and not w.isRunning() and w.isFinished(),
                'seen=%s running=%s finished=%s'
                % (w.finished_seen.is_set(), w.isRunning(), w.isFinished()))
        check('6 退出后引用在一次 _sweep_retired_workers() 后被释放'
                '（= 生产机制：finished 槽 / 下次运行开始都会调）',
                (dyn._sweep_retired_workers() or True)
                and w not in (dyn._retired_workers or []),
                str([type(x).__name__ for x in (dyn._retired_workers or [])]))

        # 反向守卫：还在跑的绝不能丢（丢了就是 0xc0000409 那类崩溃）
        w3 = _HoldWorker()
        dyn._worker = w3
        _mk(w3)
        dyn._retired_workers = [w3]
        dyn._sweep_retired_workers()
        check('6b 反向守卫：仍在跑的 worker 在 sweep 后**必须保留**'
                '（只丢确实不跑的）', w3 in (dyn._retired_workers or []),
                str([type(x).__name__ for x in (dyn._retired_workers or [])]))
        check('7 全程无 QThread 销毁告警', not _has_warn())
        w3.release.set()
        _wait(lambda: not w3.isRunning(), 'w3 收尾')
        dyn._sweep_retired_workers()
        dyn._worker = None

        # 再跑一轮：连续两轮不残留
        w2 = _HoldWorker()
        dyn._worker = w2
        group('D 连续两轮无堆积、全程无 QThread 销毁告警')
        check('8a 第二轮也已启动（entered）', _mk(w2))
        dyn._finish_run_ui()
        w2.release.set()
        _wait(lambda: w2.finished_seen.is_set() and not w2.isRunning(), 'w2 退出')
        dyn._sweep_retired_workers()
        app.processEvents()
        check('8 连续两轮后 _retired_workers 已清空（无堆积）',
                not (dyn._retired_workers or []),
                str(len(dyn._retired_workers or [])))
        check('9 第二轮退出后仍无 QThread 销毁告警（两轮都覆盖）',
                not _has_warn())
        flush()
    finally:
        # 无论成败都放行并等退出，避免"进程退出时线程还在跑"污染结论
        for x in (w, w2):
            try:
                if x is not None:
                    x.release.set()
            except Exception:                                     # noqa: BLE001
                pass
        t0 = time.time()
        while time.time() - t0 < 5.0 and any(
                _running(x) for x in (w, w2)):
            app.processEvents()
            time.sleep(0.005)
        qInstallMessageHandler(old)
        dyn._worker = None
        dyn._retired_workers = []

    ok = C.summary()
    if not ok:
        print('\n提示：2/3/4 是本次事故的直接回归锚（3b 保证第 3 条不是空转，'
              '6b 守住"只丢确实不跑的"）；4/7/9 失败意味着 Qt 会 fail-fast'
              '（整进程消失+结果全丢）。')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
