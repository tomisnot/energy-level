# -*- coding: utf-8 -*-
"""核心计算包：物理计算（无 UI 依赖）。

模块地图（t74/t74b 后终态）：
- static_atom.py    单原子静态（能级 / AC/DC-Stark / 偶极矩阵元 / 谱线 / 偏振）
- static_pair.py    两体静态（pair 哈密顿 / vdW / 交换耦合 / 耦合强度雷达图）
- atomic_levels.py  能级构造（make_levels / gaussian / square / step）
- field_build.py    场构造（field/group/real + build_fields_exact）
- waveforms.py      波形数据层（Wave / 包络 / 序列化）
- sambe.py          Sambe 引擎（核心层 + 单体 + 两体，t74 合一）
- evolve_mixed.py   混合引擎（单/两体 + acize，t74/t74b 合一）
- hard.py           hard 引擎单/两体 + build_two_field
- evolve_twofield.py 两场引擎家族
- evolve_reference.py 参考积分器（H0 IP + Lindblad BDF）
- evolve_rwa.py     ip4/Lindblad 遗留引擎（数值权威参照）
- dop853.py / coupling.py / project.py / tensor.py / tensor_fields.py /
  params.py / evolve_util.py / evolve_engine.py / route.py  公共层与调度

兼容导出：旧名 → 新模块（供 checks/scripts 历史脚本零改动运行）。
"""
import sys as _sys

# ---- 兼容导出：旧名 → 新模块（增量注册） ----
from . import static_atom as core
_sys.modules[__name__ + '.core'] = core

from . import static_pair as pair_core
_sys.modules[__name__ + '.pair_core'] = pair_core

from .sambe import evolve_sambe as evolve_exact
_sys.modules[__name__ + '.evolve_exact'] = evolve_exact

from .sambe import evolve_sambe_pair as evolve_exact_pair
_sys.modules[__name__ + '.evolve_exact_pair'] = evolve_exact_pair

from . import evolve_twofield as evolve_two_field
_sys.modules[__name__ + '.evolve_two_field'] = evolve_two_field

from . import evolve_mixed as evolve_mixed       # 混合引擎（单/两体，t74 合并）
_sys.modules[__name__ + '.evolve_mixed'] = evolve_mixed
_sys.modules[__name__ + '.evolve_mixed_pair'] = evolve_mixed

from . import hard as hard_numba      # hard_numba.py + hard_pair.py → hard.py
_sys.modules[__name__ + '.hard_numba'] = hard_numba
_sys.modules[__name__ + '.hard_pair'] = hard_numba

del _sys
