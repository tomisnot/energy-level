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

## 本自检
用**合成 QThread**（run() 里 sleep 0.4 s 再返回）复现"done 槽先于 run() 返回"的时序：
调 `_finish_run_ui()` 时线程仍在跑，然后断言
  1. `_worker` 已清空（UI 语义不变）；
  2. worker 被 `_retired_workers` 强引用住；
  3. 线程退出后自动释放（不泄漏）；
  4. **Qt 消息流里没有** `Destroyed while thread is still running`（经
     `qInstallMessageHandler` 捕获；旧代码会在这里打出该消息并 fail-fast）。

用法：python checks/gui_worker_lifetime_test.py（offscreen；构窗失败/MemoryError → SKIP）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                              # noqa: E402

C = Checker('gui_worker_lifetime')
_MSGS = []


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

    class _SlowWorker(QThread):
        """合成 worker：模拟"done 槽先于 run() 返回"的时序（sleep 后返回）。"""

        def __init__(self, hold_s=0.4):
            super().__init__(None)          # 与真实 worker 一致：parent=None
            self.hold_s = float(hold_s)

        def run(self):
            t0 = time.time()
            while time.time() - t0 < self.hold_s:
                time.sleep(0.01)
            return

    dyn = win.dynamic
    try:
        C.check('0 面板有 _retired_workers 与 _sweep_retired_workers（修已就位）',
                hasattr(dyn, '_retired_workers')
                and callable(getattr(dyn, '_sweep_retired_workers', None)))

        w = _SlowWorker(0.4)
        dyn._worker = w                     # 与真实路径一样：面板持有唯一强引用
        w.start()
        # ⚠ `isRunning()` 在 `start()` 刚返回时可能**短暂为 False**（Qt 在新线程内部
        #   才置 `d->running`）→ 必须等一下再断言（本自检第一版就在这里误报了）。
        t0 = time.time()
        while not w.isRunning() and time.time() - t0 < 2.0:
            app.processEvents()
            time.sleep(0.005)
        C.check('1 线程已启动', w.isRunning())

        # 【关键】模拟 done 槽在 run() 还没返回时执行收尾
        dyn._finish_run_ui()
        app.processEvents()
        C.check('2 收尾后 _worker 已清空（UI 语义不变）', dyn._worker is None)
        C.check('3 仍在跑的 worker 被 _retired_workers 强引用住（不丢最后一根引用）',
                w in (dyn._retired_workers or []), str(len(dyn._retired_workers or [])))
        C.check('4 没有打印 "QThread: Destroyed while thread is still running"',
                not any('Destroyed while thread is still running' in m
                        for m in _MSGS), str(_MSGS[-3:]))

        # 等线程真的结束 → 引用自动释放
        t0 = time.time()
        while w.isRunning() and time.time() - t0 < 5.0:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        C.check('5 线程已退出', not w.isRunning())
        C.check('6 退出后引用被释放（不泄漏）',
                w not in (dyn._retired_workers or []),
                str([type(x).__name__ for x in (dyn._retired_workers or [])]))
        C.check('7 全程无 QThread 销毁告警',
                not any('Destroyed while thread is still running' in m
                        for m in _MSGS))

        # 再跑一次：连续两轮不残留
        w2 = _SlowWorker(0.2)
        dyn._worker = w2
        w2.start()
        t0 = time.time()
        while not w2.isRunning() and time.time() - t0 < 2.0:
            app.processEvents()
            time.sleep(0.005)
        C.check('8a 第二轮也已启动', w2.isRunning())
        dyn._finish_run_ui()
        t0 = time.time()
        while w2.isRunning() and time.time() - t0 < 5.0:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        C.check('8 连续两轮后 _retired_workers 已清空（无堆积）',
                not (dyn._retired_workers or []),
                str(len(dyn._retired_workers or [])))
    finally:
        qInstallMessageHandler(old)
        dyn._worker = None
        dyn._retired_workers = []

    ok = C.summary()
    if not ok:
        print('\n提示：2/3/4 是本次事故的直接回归锚；4 失败意味着 Qt 会 fail-fast'
              '（整进程消失+结果全丢）。')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
