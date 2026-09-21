# -*- coding: utf-8 -*-
"""Energy Level 软件包根（Re0 引擎沙盒 + 旧仓惰性 UI 导出，合并版）。

- **引擎别名**（惰性）：`src.core` / `src.pair_core` / `src.waveforms` → `src.core.*`
  （`src/core/__init__.py` 已把旧名注册进 sys.modules；这里让它们从 src 层也拿得到）。
- **UI 惰性导出**（PEP 562）：`from src import fields/ui_dynamic/...` → `src.ui.*`。
  惰性 ⇒ `import src.core.X` 不连带拉起 PyQt5，并行扫描 runner/worker 保持 headless。
- **ARC 数据补丁**：暂不装（沿用 Re0：走 `~/.arc-data`，与判据一致、导入快）。
  打包/发布（Phase E）时再装 `.arcdata` 重定向补丁，届时重跑基线核对物理不变。
"""

_ENGINE_ALIASES = ('core', 'pair_core', 'waveforms')

# UI 惰性导出（`from src import X` → `src.ui.X`）；迁移期逐步接回 mecha
_LAZY_UI = ('fields', 'ui_theme', 'ui_single', 'ui_radar', 'ui_waves',
            'ui_dynamic', 'evolve_worker', 'app_state', 'ui_group',
            'ui_spectrum', 'hamiltonian_viewer', 'solver_inspector',
            'sweep_panel')


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
