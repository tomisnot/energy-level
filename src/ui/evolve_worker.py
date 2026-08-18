# -*- coding: utf-8 -*-
"""演化计算后台线程（独立于 UI，供动态页复用）。

阻塞式 evolve.evolve 放 QThread 后台，主线程轮询 progress_arr 更新进度/取消。
不依赖任何 UI 类；done/log_msg 信号由使用方连接。
"""
from PyQt5.QtCore import QThread, pyqtSignal

from ..core import evolve


class EvolveWorker(QThread):
    """演化计算线程：阻塞式 evolve 放后台，主线程轮询进度/可取消。"""

    done = pyqtSignal(object, object)       # (result_dict, error_str 或 None)
    log_msg = pyqtSignal(str)               # 算法提示/警告 → 状态栏

    def __init__(self, states, fields, psi0, t_grid, method, backend,
                 progress_arr, decays=None, parent=None):
        super().__init__(parent)
        self.states = states
        self.fields = fields
        self.psi0 = psi0
        self.t_grid = t_grid
        self.method = method
        self.backend = backend
        self.progress_arr = progress_arr
        self.decays = list(decays or [])

    def run(self):
        try:
            # ac_shift 已内嵌各 field（随包络时变），无需另传
            res = evolve.evolve(self.states, self.fields, self.psi0,
                                self.t_grid, method=self.method,
                                backend=self.backend,
                                progress_arr=self.progress_arr,
                                log=self.log_msg.emit,
                                decays=self.decays)
            self.done.emit(res, None)
        except RuntimeError as e:
            # 用户取消：progress_arr[1]≠0 → 不算错误，静默
            if self.progress_arr[1] != 0.0:
                self.done.emit(None, 'cancelled')
            else:
                self.done.emit(None, str(e))
        except Exception as e:
            self.done.emit(None, str(e))
