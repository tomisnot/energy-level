# -*- coding: utf-8 -*-
"""Energy Level 软件包根（Re0 引擎沙盒 + 旧仓惰性 UI 导出，合并版）。

- **引擎别名**（惰性）：`src.core` / `src.pair_core` / `src.waveforms` → `src.core.*`
  （`src/core/__init__.py` 已把旧名注册进 sys.modules；这里让它们从 src 层也拿得到）。
- **UI 惰性导出**（PEP 562）：`from src import fields/ui_dynamic/...` → `src.ui.*`。
  惰性 ⇒ `import src.core.X` 不连带拉起 PyQt5，并行扫描 runner/worker 保持 headless。
- **ARC 数据补丁**：若仓内 `.arcdata/` 存在，导入时把 ARC 的数据目录重定向到仓内，
  避免把 SQLite literatureDME 写到用户目录；没有仓内数据时保留 ARC 默认路径。
"""

_ENGINE_ALIASES = ('core', 'pair_core', 'waveforms')

# UI 惰性导出（`from src import X` → `src.ui.X`）；迁移期逐步接回 mecha
_LAZY_UI = ('fields', 'ui_theme', 'ui_single', 'ui_radar', 'ui_waves',
            'ui_dynamic', 'evolve_worker', 'app_state', 'ui_group',
            'ui_spectrum', 'hamiltonian_viewer', 'solver_inspector',
            'sweep_panel')


def _configure_arc_data():
    """把 ARC 的可写缓存优先放进当前 EL 仓库。

    ARC 在 `Rubidium87()` 构造时会初始化/更新 SQLite。这个补丁必须发生在
    原子对象创建前；缺少 ARC 或 `.arcdata` 时静默保留 ARC 默认行为，避免
    `import src` 变成 ARC 的硬依赖。
    """
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            '.arcdata')
    if not os.path.isdir(data_dir):
        return
    try:
        from arc.alkali_atom_functions import AlkaliAtom
    except Exception:
        return
    AlkaliAtom.dataFolder = data_dir
    globals()['_ARC_DATA_FOLDER'] = data_dir


_configure_arc_data()


def __getattr__(name):
    import importlib
    if name in _ENGINE_ALIASES:
        mod = importlib.import_module('src.core.%s' % name)
        globals()[name] = mod          # 命中一次后转常规属性
        return mod
    if name in _LAZY_UI:
        mod = importlib.import_module('src.ui.%s' % name)
        globals()[name] = mod
        return mod
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(list(globals()) + list(_ENGINE_ALIASES) + list(_LAZY_UI))
