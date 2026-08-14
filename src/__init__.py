# -*- coding: utf-8 -*-
"""Energy Level 软件核心与 UI 组件包（入口为顶层 能级.py）。

- core.py       ：单原子物理核心（能级/AC-Stark/DC-Stark/频谱）
- pair_core.py  ：两体静态（偶极-偶极相互作用/pair 哈密顿）
- driver.py     ：统一驱动引擎（build_system/drive/线性响应谱）
- ui_pair.py    ：响应谱画布 + 统一响应谱计算
- fields.py     ：场定义面板 + 光场上层处理（磁场/电场/光场表格/关注跃迁/初态/扫描光强/导出频率）

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
