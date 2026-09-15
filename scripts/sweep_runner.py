# -*- coding: utf-8 -*-
"""并行扫描 runner 入口（独立进程的 `__main__`）。

**故意做得极轻**：Windows 的 spawn 会在每个 worker 子进程里重新导入本文件，
所以这里只准放"设环境变量 + 加 sys.path + 转调"三件事。任何 Qt 或重依赖的
导入都会让 W 个 worker 各付一次代价（而本进程的存在意义之一就是不带 Qt）。

用法（一般由 GUI/MCP 的 `src.sweep.registry` 代起，也可手动）：
    python scripts/sweep_runner.py --spec runtime/sweeps/<sid>/spec.json
"""
import os
import sys

# ⚠ 必须早于 numpy/引擎导入：BLAS/numba 的线程数是在**导入期**读的。
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 继承提交方的 sys.path：软件本体可能依赖**项目之外的包**（本项目的
# `param_manager` 就是同级目录里的另一个仓库，靠调用方的 sys.path 才可见）。
# 不继承的话 runner 子进程会 import 失败，而且错误发生在远处的 worker 里、
# 很难归因。提交侧见 src/sweep/registry.py 的 EL_SWEEP_SYS_PATH。
for _p in reversed((os.environ.get('EL_SWEEP_SYS_PATH') or '').split(os.pathsep)):
    if _p and _p not in sys.path:
        sys.path.append(_p)

from src.sweep.runner import main  # noqa: E402


if __name__ == '__main__':
    sys.exit(main())
