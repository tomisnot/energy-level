# -*- coding: utf-8 -*-
"""核心计算包：物理计算（无 UI 依赖）。

- static_atom.py    单原子静态（能级 / AC/DC-Stark / 偶极矩阵元 / 谱线 / 偏振）
- static_pair.py    两体静态（pair 哈密顿 / vdW / 交换耦合 / 耦合强度雷达图）
- atomic_levels.py  能级构造（make_levels / gaussian / square / step）
- field_build.py    场构造（field/group/real/arbitrary_field + build_fields）
- evolve_rwa.py     单原子 RWA 含时演化（ip4 / Lindblad）
- evolve_arbitrary.py 任意波形演化（实验室系，无 RWA）
- evolve_sambe.py   单原子 Sambe 光子块演化（原 evolve_exact）
- evolve_sambe_pair.py 两体 Sambe（原 evolve_exact_pair）
- evolve_twofield.py 两场引擎（光旋转系 + 微波，原 evolve_two_field）
- hard.py           hard 硬算单/两体（原 hard_numba + hard_pair）
- waveforms.py      波形数据层（Wave / 包络 / build_fields）
- dop853.py / sambe.py / tensor.py / driver.py  核心层与驱动（命名已功能化）
- pair_evolve.py    两体 RWA 演化（保留旧名，t19 未列入改动）

兼容导出：旧文件名 → 新模块 re-export（src 内部互引与外部 UI/checks 零改动）。
每个别名在对应新模块导入后**立即**注册——后续模块的内部旧名引用
（如 evolve_sambe_pair 的 `from .evolve_exact import …`）即可解析。
"""
import sys as _sys

# ---- 兼容层：旧文件名 → 新模块（t19 重命名，零外部改动；增量注册） ----
from . import static_atom as core
_sys.modules[__name__ + '.core'] = core

from . import static_pair as pair_core
_sys.modules[__name__ + '.pair_core'] = pair_core

from . import evolve_sambe as evolve_exact
_sys.modules[__name__ + '.evolve_exact'] = evolve_exact

from . import evolve_sambe_pair as evolve_exact_pair
_sys.modules[__name__ + '.evolve_exact_pair'] = evolve_exact_pair

from . import evolve_twofield as evolve_two_field
_sys.modules[__name__ + '.evolve_two_field'] = evolve_two_field

from . import evolve_mixed as evolve_mixed       # 混合引擎（单原子）
_sys.modules[__name__ + '.evolve_mixed'] = evolve_mixed

from . import evolve_mixed_pair as evolve_mixed_pair   # 混合引擎（两体）
_sys.modules[__name__ + '.evolve_mixed_pair'] = evolve_mixed_pair

from . import hard as hard_numba      # hard_numba.py + hard_pair.py → hard.py
_sys.modules[__name__ + '.hard_numba'] = hard_numba
_sys.modules[__name__ + '.hard_pair'] = hard_numba

del _sys
