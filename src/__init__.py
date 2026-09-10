# -*- coding: utf-8 -*-
"""Energy Level 软件包（入口为顶层 能级.py）。

包结构：
- src.core/   核心计算（无 UI 依赖）：core / pair_core / waveforms
              （driver 关注跃迁引擎已归档 _archive，t73b）
- src.ui/     界面：fields / ui_theme / ui_single / ui_radar /
              ui_waves / ui_dynamic / evolve_worker / app_state / serialization / pages

兼容导出：`from src import waveforms, pair_core, ui_theme, ...` 继续工作
（委托到 src.core.* / src.ui.*），历史脚本（checks/scripts）无需改动。
⚠ 例外：`core` **不再从 src 层导出**（会遮蔽同名子包 `src.core`，见文件末尾注释）——
旧写法 `from src import core` 请改为 `from src.core import core`。

导入本包即自动把 ARC（Alkali Rydberg Calculator）的数据目录重定向到项目内
.arcdata/（见 _patch_arc_datafolder）：ARC 的 Rubidium87() 实例化会在其数据
目录写 sqlite 缓存（rb87_precalculated.db 等），默认位于用户主目录 ~/.arc-data
（沙箱/受限环境下无写权限会启动即崩）。本项目自带 Rb-87 所需缓存子集，
重定向后所有读写都落在项目内，无需外部写权限，也保证结果可复现。

打包模式（PyInstaller，sys.frozen）：程序运行在只读/临时目录，不能直接写
包内 .arcdata —— 首次启动把随包携带的 `.arcdata` 种子复制到
%LOCALAPPDATA%/EnergyLevel/.arcdata（可写、持久、跨版本复用），再重定向到那里。
"""
import os
import shutil
import sys

_ARC_FROZEN = bool(getattr(sys, 'frozen', False))
_ARC_DATA_LOCAL = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), '.arcdata')
_ARC_PATCHED = False


def _copy_missing(src_dir, dst_dir):
    """把种子 src_dir 中缺失的文件递归复制到 dst_dir（二次启动即跳过，<1 s）。"""
    if not os.path.isdir(src_dir):
        return
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        tgt = dst_dir if rel == '.' else os.path.join(dst_dir, rel)
        try:
            os.makedirs(tgt, exist_ok=True)
        except OSError:
            continue
        for fn in files:
            s = os.path.join(root, fn)
            d = os.path.join(tgt, fn)
            if not os.path.exists(d):
                try:
                    shutil.copy2(s, d)
                except OSError:
                    pass


def _arc_data_target():
    """ARC 数据目录（确保存在并返回其路径）。

    - 开发模式（非 frozen）：项目根 .arcdata/（沿用现状；不存在时不强制创建，
      保持 ARC 默认路径行为，见 _patch_arc_datafolder）；
    - 打包模式（frozen）：把随包种子 `.arcdata` 一次性复制到
      %LOCALAPPDATA%/EnergyLevel/.arcdata（可写、持久）；
      复制失败则回退项目内目录，再不行就用种子目录本身。
    """
    if not _ARC_FROZEN:
        return _ARC_DATA_LOCAL
    seed = os.path.join(
        getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)), '.arcdata')
    root = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    dest = os.path.join(root, 'EnergyLevel', '.arcdata')
    try:
        os.makedirs(dest, exist_ok=True)
        _copy_missing(seed, dest)
        return dest
    except OSError:
        try:
            os.makedirs(_ARC_DATA_LOCAL, exist_ok=True)
            _copy_missing(seed, _ARC_DATA_LOCAL)
            return _ARC_DATA_LOCAL
        except OSError:
            return dest if os.path.isdir(dest) else seed


def _patch_arc_datafolder():
    """把 arc.alkali_atom_functions.AlkaliAtom.dataFolder 指向 ARC 数据目录。

    dataFolder 是类属性，实例化（Rubidium87()）时动态查找，故导入 arc 后、
    实例化前打补丁即可生效；不修改 arc 源码。若目标目录不存在（开发模式且
    未运行 scripts/setup_arcdata.py），保持 arc 默认路径（~/.arc-data）。
    """
    global _ARC_PATCHED
    if _ARC_PATCHED:
        return
    data_dir = _arc_data_target()
    if not os.path.isdir(data_dir):
        return
    try:
        import arc.alkali_atom_functions as _aaf
    except Exception:
        return
    _aaf.AlkaliAtom.dataFolder = data_dir
    _ARC_PATCHED = True


_patch_arc_datafolder()

# ---------------- 兼容导出（from src import X 委托到 core./ui.） ----------------
# ⚠ `core` **故意不在此列**（t78 修）：`from .core import core` 会给 src 绑定一个
#   与子包同名的属性 `core` → 遮蔽真正的 `src.core` 子包，导致
#   `import src.core.<x> as <别名>` 报 `cannot import name '<x>' from
#   src.core.static_atom`（CPython 对带 as 的点号导入走父模块属性遍历）。
#   旧写法 `from src import core` 的调用方请改用 `from src.core import core`
#   （两者是同一个 static_atom 对象），语义不变。
from .core import pair_core       # noqa: E402,F401
from .core import evolve_exact    # noqa: E402,F401
from .core import waveforms       # noqa: E402,F401
from .ui import fields            # noqa: E402,F401
from .ui import ui_theme          # noqa: E402,F401
from .ui import ui_single         # noqa: E402,F401
from .ui import ui_radar          # noqa: E402,F401
from .ui import ui_waves          # noqa: E402,F401
from .ui import ui_dynamic        # noqa: E402,F401
from .ui import evolve_worker     # noqa: E402,F401
from .ui import app_state         # noqa: E402,F401
#from .ui import pages             # noqa: E402,F401