# -*- coding: utf-8 -*-
"""主线程派发器：把 MCP 线程的调用安全地搬到 Qt 主线程执行。

背景：Qt 状态（store/控件）只允许主线程读写。MCP server 跑在独立线程的
asyncio 事件循环里，工具 handler 若直接碰 store 会崩（PyQt qFatal 无回溯）。

机制（草案 §1"Qt 线程亲和"）：
  worker 线程调 run_on_main(fn, timeout) → 通过一个主线程 QObject 上的
  信号（QueuedConnection）投递闭包 → 主线程执行 → threading.Event 回传
  结果/异常。主线程若正阻塞（长计算无事件泵），超时返回错误而非死锁。

单例：init(app) 在 MainWindow 构造后调用一次；get() 取派发器。
"""
import threading

from PyQt5.QtCore import QObject, QCoreApplication, Qt, pyqtSignal


class _Dispatcher(QObject):
    _invoke = pyqtSignal(object)   # 携带 callable（闭包自带结果槽）

    def __init__(self):
        super().__init__()
        # 关键：QueuedConnection 保证跨线程 emit 时槽在主线程事件循环执行
        self._invoke.connect(self._on_invoke, Qt.QueuedConnection)
        self._main_tid = threading.get_ident()

    def _on_invoke(self, fn):
        """主线程槽：执行投递来的闭包（闭包内部自行回传结果）。"""
        try:
            fn()
        except BaseException as e:      # noqa: BLE001 - 回传给调用线程
            self._last_error = e


_DISP = None


def init():
    """在 Qt 主线程（MainWindow 构造期）调用，安装派发器单例。"""
    global _DISP
    if _DISP is None:
        app = QCoreApplication.instance()
        if app is None:
            raise RuntimeError('QApplication 尚未创建，无法安装主线程派发器')
        _DISP = _Dispatcher()
    return _DISP


def get():
    return _DISP


def is_main_thread():
    return _DISP is not None and threading.get_ident() == _DISP._main_tid


def run_on_main(fn, timeout=30.0):
    """在 Qt 主线程执行 fn()，阻塞等待并返回其结果。

    已在主线程则直接调用；否则投递 + Event 等待。超时抛 TimeoutError
    （fn 可能稍后仍在主线程执行，故 fn 应幂等或可容忍迟到）。
    """
    d = _DISP
    if d is None:
        raise RuntimeError('主线程派发器未初始化（先调用 ai.dispatch.init()）')
    if threading.get_ident() == d._main_tid:
        return fn()

    box = {}
    ev = threading.Event()

    def _task():
        try:
            box['result'] = fn()
        except BaseException as e:      # noqa: BLE001
            box['error'] = e
        finally:
            ev.set()

    d._invoke.emit(_task)
    if not ev.wait(timeout):
        raise TimeoutError(f'主线程 {timeout}s 内未响应（可能正阻塞在长任务）')
    if 'error' in box:
        raise box['error']
    return box.get('result')
