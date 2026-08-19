# -*- coding: utf-8 -*-
"""Energy Level 软件包（入口为顶层 能级.py）。

包结构：
- src.core/   核心计算（无 UI 依赖）：core / pair_core / driver / evolve / waveforms
- src.ui/     界面：fields / ui_theme / ui_single / ui_radar / ui_pair /
              ui_waves / ui_dynamic / evolve_worker / app_state / serialization / pages

兼容导出：`from src import core, evolve, ...` 继续工作（委托到 src.core.* / src.ui.*），
历史脚本（checks/scripts）无需改动。

导入本包即自动把 ARC（Alkali Rydberg Calculator）的数据目录重定向到项目内
.arcdata/（见 _patch_arc_datafolder）：ARC 的 Rubidium87() 实例化会在其数据
目录写 sqlite 缓存（rb87_precalculated.db 等），默认位于用户主目录 ~/.arc-data
（沙箱/受限环境下无写权限会启动即崩）。本项目自带 Rb-87 所需缓存子集，
重定向后所有读写都落在项目内，无需外部写权限，也保证结果可复现。
"""
import os

_ARC_DATA_LOCAL = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), '.arcdata')
_ARC_PATCHED = False


def _patch_arc_datafolder():
    """把 arc.alkali_atom_functions.AlkaliAtom.dataFolder 指向项目 .arcdata/。

    dataFolder 是类属性，实例化（Rubidium87()）时动态查找，故导入 arc 后、
    实例化前打补丁即可生效；不修改 arc 源码。若 .arcdata/ 不存在（未运行
    scripts/setup_arcdata.py），保持 arc 默认路径（~/.arc-data）。
    """
    global _ARC_PATCHED
    if _ARC_PATCHED or not os.path.isdir(_ARC_DATA_LOCAL):
        return
    try:
        import arc.alkali_atom_functions as _aaf
    except Exception:
        return
    _aaf.AlkaliAtom.dataFolder = _ARC_DATA_LOCAL
    _ARC_PATCHED = True


_patch_arc_datafolder()

# ---------------- 兼容导出（from src import X 委托到 core./ui.） ----------------
from .core import core            # noqa: E402,F401
from .core import pair_core       # noqa: E402,F401
from .core import driver          # noqa: E402,F401
from .core import evolve          # noqa: E402,F401
from .core import waveforms       # noqa: E402,F401
from .ui import fields            # noqa: E402,F401
from .ui import ui_theme          # noqa: E402,F401
from .ui import ui_single         # noqa: E402,F401
from .ui import ui_radar          # noqa: E402,F401
from .ui import ui_pair           # noqa: E402,F401
from .ui import ui_waves          # noqa: E402,F401
from .ui import ui_dynamic        # noqa: E402,F401
from .ui import evolve_worker     # noqa: E402,F401
from .ui import app_state         # noqa: E402,F401
from .ui import serialization     # noqa: E402,F401
#from .ui import pages             # noqa: E402,F401
