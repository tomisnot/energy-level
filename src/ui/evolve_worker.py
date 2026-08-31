# -*- coding: utf-8 -*-
"""演化计算后台线程（独立于 UI，供动态页复用）。

阻塞式 evolve.evolve 放 QThread 后台，主线程轮询 progress_arr 更新进度/取消。
不依赖任何 UI 类；done/log_msg 信号由使用方连接。
"""
from PyQt5.QtCore import QThread, pyqtSignal
import os
import numpy as np

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
                # t20：hard 硬积分（numba 加速）——fields 为两场 meta
                from ..core.evolve_twofield import evolve_hard
                res = evolve_hard(
                    self.states, self.fields, self.psi0, self.t_grid,
                    rtol=1e-8, atol=1e-10, numba=True,
                    progress_arr=self.progress_arr, log=self.log_msg.emit)
            else:
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



def _export_request_for_julia(req_path, states, fields_A, fields_B,
                               waves_A, waves_B, psi0_A_idx, psi0_B_idx,
                               t_grid, vdd_matrix, decays_A, decays_B):
    """从 GUI 数据结构导出 Julia NPZ 请求（schema v1.1）。"""
    import numpy as np
    import _julia_bridge as _jb

    N = len(states)
    N2 = N * N
    mF_states = [float(s.get('m', s.get('mJ', s.get('mF', 0.0))))
                 for s in states]

    # 从 exact fields 构建载频分组
    groups = []
    fields_out = []
    for side, flds, wvs in (('A', fields_A, waves_A),
                            ('B', fields_B, waves_B)):
        I_N = np.eye(N)
        for fi, f in enumerate(flds):
            A = np.asarray(f['A'], float)
            K2 = np.kron(A, I_N) if side == 'A' else np.kron(I_N, A)
            w = float(f['omega_L_mhz'])
            hit = None
            for gi, g in enumerate(groups):
                if abs(g['omega'] - w) <= 1e-9 * max(abs(w), 1.0):
                    hit = g
                    break
            if hit is None:
                groups.append(dict(omega=w, K2=K2, pols=[],
                                   envs=[], waves=[]))
                hit = groups[-1]
            hit['K2'] = hit['K2'] + K2
            hit['pols'].append(wvs[fi].pol if fi < len(wvs) else 'π')
            hit['waves'].append(wvs[fi] if fi < len(wvs) else None)

    # 光子荷
    charges = []
    blockable = True
    for g in groups:
        qs = set()
        for p in g['pols']:
            q = 1.0 if p == 'σ+' else (-1.0 if p == 'σ-' else
                                       (0.0 if p == 'π' else float('nan')))
            qs.add(q)
        if len(qs) != 1 or np.isnan(next(iter(qs))):
            blockable = False
            charges.append(0.0)
        else:
            charges.append(next(iter(qs)))

    # 场列表（扁平化：每场 → (group_idx, kind, p1..4)）
    fields_flat = []
    samp_count = 0
    for gi, g in enumerate(groups):
        for wi, wv in enumerate(g['waves']):
            if wv is None:
                continue
            e = wv.envelope
            if hasattr(e, 'center_ns'):
                kind = 0
                p1 = e.center_ns + wv.offset_ns
                p2 = e.sigma_ns
                p3 = e.peak
                p4 = 0.0
            elif hasattr(e, 'start_ns'):
                kind = 1
                p1 = e.start_ns + wv.offset_ns
                p2 = e.edge_ns
                p3 = e.height
                p4 = (e.fall_start_ns if e.fall_start_ns is not None
                      else e.start_ns + 2 * e.edge_ns) + wv.offset_ns
            else:
                continue
            fields_flat.append((gi, kind, (p1, p2, p3, p4), None, None))

    # E_pair
    E = [float(s['E_mhz']) for s in states]
    E_pair = [E[i] + E[j] for i in range(N) for j in range(N)]

    _jb.write_request(req_path, N=N, E_pair=E_pair, vdd=vdd,
                      groups=[(g['omega'], g['K2']) for g in groups],
                      fields=fields_flat,
                      psi0=(psi0_A_idx, psi0_B_idx),
                      t_grid=t_grid, tol=1e-6,
                      gamma_A=None, gamma_B=None,
                      m_max=6, dense_max=700, sparse_backend=1,
                      block_mf=blockable,
                      skip_probe=True)



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
                 waves_A=None, waves_B=None, use_julia=False):
        super().__init__(parent)
        self.waves_A = waves_A
        self.waves_B = waves_B
        self.use_julia = use_julia
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
        self.method = method
        self._cache_ref = None

    def _run_julia(self):
        """Julia 后端：导出 NPZ → 子进程 → 读取结果 → 转标准返回格式。"""
        try:
            import checks._julia_bridge as jb
        except ImportError:
            try:
                import _julia_bridge as jb
            except ImportError:
                self.log_msg.emit('[Julia] 桥接模块不可用，回退 Python')
                return None
        exe = jb.find_julia()
        if exe is None:
            self.log_msg.emit('[Julia] 未找到 julia.exe，回退 Python')
            return None
        self.log_msg.emit(f'[Julia] 后端: {exe}')
        work = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), '..', 'julia', 'work')
        os.makedirs(work, exist_ok=True)
        req = os.path.abspath(os.path.join(work, 'gui_req.npz'))
        resp = os.path.abspath(os.path.join(work, 'gui_resp.npz'))
        cancel = os.path.abspath(os.path.join(work, 'gui_cancel.flag'))

        # ---- 导出请求 ----
        # fields_A/B 已在主线程构建（kind='exact'），直接使用
        fields_eA, fields_eB = self.fields_A, self.fields_B

        try:
            _export_request_for_julia(
                req, self.states, fields_eA, fields_eB,
                self.waves_A or [], self.waves_B or [],
                self.psi0_A, self.psi0_B, self.t_grid,
                self.vdd_matrix, self.decays_A, self.decays_B)
        except Exception as e:
            self.log_msg.emit(f'[Julia] 导出失败: {e}，回退 Python')
            return None

        # ---- 运行 ----
        self.log_msg.emit('[Julia] 启动计算…')
        try:
            code, out = jb.run_cli(req, resp, cancel_file=cancel, timeout=14400)
        except Exception as e:
            self.log_msg.emit(f'[Julia] 运行异常: {e}')
            return None
        if code == 2:
            raise RuntimeError('用户取消')
        if code != 0:
            self.log_msg.emit(f'[Julia] 退出码 {code}')
            return None
        d = dict(np.load(resp))
        if int(d['ok'][0]) != 1:
            err = bytes(d.get('err_msg', b'')).decode('utf-8', 'replace')
            self.log_msg.emit(f'[Julia] 错误: {err}')
            return None

        # ---- 转标准返回格式 ----
        t = d['t']
        N2 = d['pops_2d'].shape[1]
        N = int(round(np.sqrt(N2)))
        pops_A = d['pops_A']
        pops_B = d['pops_B']
        loss_A = d['loss_A']
        loss_B = d['loss_B']
        states_2d = [dict(label=f'({self.states[i]["label"]}) x '
                                f'({self.states[j]["label"]})',
                          E_mhz=self.states[i]['E_mhz'] + self.states[j]['E_mhz'],
                          ket_i=i, ket_j=j)
                     for i in range(N) for j in range(N)]
        self.log_msg.emit(f'[Julia] 完成: {int(d["n_segments"][0])} 段, '
                          f'M={int(d["M"][0])}, '
                          f'耗时 {d["timing"][0]:.0f}s')
        return dict(
            t=t, pops_2d=d['pops_2d'], pops_A=pops_A, pops_B=pops_B,
            loss_A=loss_A, loss_B=loss_B, states_2d=states_2d,
            method='exact', backend='julia-wosc',
            error_report=dict(
                n_segments=int(d['n_segments'][0]),
                max_richardson_diff=float(d['max_diff'][0]),
                dt_min_hit=bool(d['dt_min_hit'][0]),
                max_norm_dev=float(d['max_norm_dev'][0]),
                photon_blocks={}, tol=0.0))


    def run(self):
        try:
            if self.method == 'exact' and self.use_julia:
                res = self._run_julia()
                if res is not None:
                    self.done.emit(res, None)
                    return
            if self.method == 'exact':
                from ..core.evolve_exact_pair import evolve_exact_pair
                res = evolve_exact_pair(
                    self.states, self.fields_A, self.fields_B,
                    self.psi0_A, self.psi0_B,
                    t_grid=self.t_grid,
                    vdd_matrix=self.vdd_matrix,
                    decays_A=self.decays_A, decays_B=self.decays_B,
                    progress_arr=self.progress_arr,
                    log=self.log_msg.emit)
            else:
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
