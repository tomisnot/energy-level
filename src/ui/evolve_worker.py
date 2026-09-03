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
        self.fields = fields          # method='hard' 时为两场 meta（build_two_field）
        self.psi0 = psi0
        self.t_grid = t_grid
        self.method = method
        self._cache_ref = None
        self.backend = backend
        self.progress_arr = progress_arr
        self.decays = list(decays or [])


    def run(self):
        try:
            if self.method == 'hard':
                # t20/t24：hard 硬积分（numba 加速）——fields 为两场 meta；
                # t24：decays 传透（落点=丢失 NonHermitian，此前静默忽略）
                from ..core.evolve_twofield import evolve_hard
                res = evolve_hard(
                    self.states, self.fields, self.psi0, self.t_grid,
                    rtol=1e-8, atol=1e-10, numba=True,
                    progress_arr=self.progress_arr, log=self.log_msg.emit,
                    decays=self.decays)
                # t28：收尾阶段标记（JIT 装载/准备 5% → 积分 5-95% → 收尾 95%）
                if len(self.progress_arr) > 2:
                    self.progress_arr[2] = 2
                    self.progress_arr[0] = 1.0
            elif self.method == 'mixed':
                # t62：混合引擎（光 Sambe 光子块 + 微波当场，φ-帧单 DOP853）
                # ——fields 为 kind='exact' 场列表（build_fields_exact 输出，
                # 与 sambe 同构——正式版已含 far_sos AC 化注入物理 E，ae04d11）
                from ..core.evolve_mixed import evolve_mixed
                res = evolve_mixed(
                    self.states, self.fields, self.psi0, self.t_grid,
                    tol=1e-8,
                    progress_arr=self.progress_arr, log=self.log_msg.emit)
            else:
                # ac_shift 已内嵌各 field（随包络时变），无需另传
                # t25：显示名 'sambe' → 引擎 'exact'（Sambe 光子块）
                res = evolve.evolve(self.states, self.fields, self.psi0,
                                    self.t_grid,
                                    method={'sambe': 'exact'}.get(
                                        self.method, self.method),
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




class EvolveWorkerPair(QThread):
    """双原子演化计算线程（纯数值：不访问 ARC/SQLite）。

    V_dd 由调用方在主线程预计算（复用主线程 ArcCache，见 ui_dynamic），
    以 vdd_matrix 传入——ARC 的 SQLite 连接不能跨线程使用
    （sqlite3: objects created in a thread can only be used in that
    same thread），工作线程内不再触碰 cache。
    """

    done = pyqtSignal(object, object)
    log_msg = pyqtSignal(str)

    def __init__(self, states, fields_A, fields_B, psi0_A, psi0_B,
                 t_grid, progress_arr, vdd_matrix=None, parent=None,
                 decays_A=None, decays_B=None, method='ip4',
                 per_freq_M=None, frozen_h_ns=None):
        super().__init__(parent)
        # t25/t26：显示名 'sambe' → 引擎 'exact'（Sambe 光子块；分支调度统一）
        self.method = {'sambe': 'exact'}.get(method, method)
        self.states = states
        self.fields_A = fields_A
        self.fields_B = fields_B
        self.psi0_A = psi0_A
        self.psi0_B = psi0_B
        self.t_grid = t_grid
        self.progress_arr = progress_arr
        self.vdd_matrix = vdd_matrix
        self.decays_A = decays_A
        self.decays_B = decays_B
        # t56：全频段两体 Sambe 可调参数（逐载频光子数 + 纯冻结段长）
        self.per_freq_M = per_freq_M
        self.frozen_h_ns = frozen_h_ns
        self._cache_ref = None

    def run(self):
        try:
            if self.method == 'hard':
                # t21：pair hard（fields_A/B 为 build_two_field meta）
                from ..core.hard import evolve_hard_pair
                res = evolve_hard_pair(
                    self.states, self.fields_A, self.fields_B,
                    self.psi0_A, self.psi0_B,
                    t_grid=self.t_grid,
                    vdd_matrix=self.vdd_matrix,
                    decays_A=self.decays_A, decays_B=self.decays_B,
                    progress_arr=self.progress_arr,
                    log=self.log_msg.emit)
                # t28：收尾阶段标记（同单原子 hard）
                if len(self.progress_arr) > 2:
                    self.progress_arr[2] = 2
                    self.progress_arr[0] = 1.0
            if self.method == 'exact':
                from ..core.evolve_exact_pair import evolve_exact_pair
                res = evolve_exact_pair(
                    self.states, self.fields_A, self.fields_B,
                    self.psi0_A, self.psi0_B,
                    t_grid=self.t_grid,
                    vdd_matrix=self.vdd_matrix,
                    decays_A=self.decays_A, decays_B=self.decays_B,
                    progress_arr=self.progress_arr,
                    log=self.log_msg.emit,
                    frozen_h_ns=self.frozen_h_ns,
                    per_freq_M=self.per_freq_M)
            elif self.method != 'hard':
                # hard 分支已在上方赋值 res；此处排除，避免 pair_evolve 覆盖
                from ..core import pair_evolve
                res = pair_evolve.pair_evolve(
                    self.states, self.fields_A, self.fields_B,
                    self.psi0_A, self.psi0_B,
                    t_grid=self.t_grid, method='ip4',
                    vdd_matrix=self.vdd_matrix,
                    decays_A=self.decays_A, decays_B=self.decays_B,
                    progress_arr=self.progress_arr,
                    log=self.log_msg.emit)
            self.done.emit(res, None)
        except RuntimeError as e:
            if self.progress_arr[1] != 0.0:
                self.done.emit(None, 'cancelled')
            else:
                self.done.emit(None, str(e))
        except Exception as e:
            self.done.emit(None, str(e))
