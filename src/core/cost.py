# -*- coding: utf-8 -*-
"""成本与内存粗估（**无 Qt**）：两体耗时/峰值内存标定 + 可用内存。

原本长在 `src/ai/sim_state.py` 里（那是 AI 门面，导入即拉起 pm_setup →
PyQt5 → param_manager）。并行扫描要在 **headless 的 runner 子进程**里按
内存定并行度，不能为了一个内存估算把 GUI 栈拖进去 ⇒ 2026-09-15 搬到这里。
`sim_state` 仍按原名再导出，既有调用方（server / checks）零改动。

**这两个估算是"粗估"不是承诺**：只用来说"要不要走异步 / 要不要降规模"，
判据与锚点见下。
"""
import os

__all__ = ['PAIR_CAL', 'PAIR_MEM_CAL', 'estimate_pair_cost',
           'estimate_pair_peak_gb', 'avail_ram_gb']

# 两体**耗时**标定：实测（2026-09-12）N=36 态 → 1296 两体态、t_max=300 ns
# 单次 ≈120 s（nfev≈4.3e5、dt≈0.002 ns，被 GHz 载波钉死）。两体 RHS 成本
# ∝ dim²，dim=N² → 成本 ∝ N⁴；对 t_max 近似线性。
PAIR_CAL = {'N': 36, 't_ns': 300.0, 'sec': 120.0}

# 两体**峰值内存**标定：使用者实测 12 组（N=68 → dim=4624）作业，进程私有
# 内存 **1.36 GB**（运行中稳定）；近似按 dim² 外推（V_dd 装配的稠密中间量
# 与 CSR/COO 拷贝占主导，y_out/pops 等线性项在小 dim 时次要）。
PAIR_MEM_CAL = {'dim': 4624, 'gb': 1.36}


def estimate_pair_cost(basis_n, t_max_ns, npts=201):
    """两体运行耗时的**粗估**（秒）——按实测标定外推。"""
    N = max(int(basis_n), 1)
    s = PAIR_CAL['sec'] * (N / PAIR_CAL['N']) ** 4
    s *= max(float(t_max_ns), 1e-9) / PAIR_CAL['t_ns']
    s *= max(int(npts), 2) / 201.0
    return float(s)


def estimate_pair_peak_gb(dim, factor=1.0):
    """两体运行峰值内存粗估（GB）；dim 非法 → None（调用方据此跳过限档）。"""
    if not dim or float(dim) <= 0:
        return None
    return float(PAIR_MEM_CAL['gb']) * (float(dim) / PAIR_MEM_CAL['dim']) ** 2 \
        * float(factor)


def avail_ram_gb():
    """当前**可用**物理内存（GB）；取不到 → None（绝不因此报错）。"""
    try:                                    # 首选 psutil（若装了）
        import psutil
        return float(psutil.virtual_memory().available) / 1e9
    except Exception:                       # noqa: BLE001
        pass
    if os.name != 'nt':
        return None
    try:                                    # Windows 原生
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return float(st.ullAvailPhys) / 1e9
    except Exception:                       # noqa: BLE001
        return None
