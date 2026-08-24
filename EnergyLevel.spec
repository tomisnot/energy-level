# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 配置（onedir 首发版）。

- 入口：main.py（ASCII 名，内部转发 能级.py）
- 数据：
    .arcdata 种子（68 MB）——随包分发，首次启动复制到
      %LOCALAPPDATA%/EnergyLevel/.arcdata（见 src/__init__.py 的 sys.frozen 分支）；
    arc 包自带 NIST 数据（collect_data_files('arc')，量子缺陷/矩阵元数据）。
- 精简：排除 tkinter / torch / pairinteraction / qutip / pytest（软件本体不需要）；
  numba 必须保留（ip4/lindblad 演化内核）。
"""
from PyInstaller.utils.hooks import collect_data_files

# ── Qt 路径修正 ─────────────────────────────────────────────────────────
# PyQt5 钩子经 QLibraryInfo 取插件/DLL/翻译路径；项目路径含中文（GBK 代码页）
# 时该路径偶发被编坏（GBK 字节按 Latin-1 解码 → isdir=False 抛异常）。
# 直接从 Python 侧按 PyQt5 wheel 布局重算，消除对 QLibraryInfo 的依赖。
import pathlib
import PyQt5 as _pyqt5
from PyInstaller.utils.hooks.qt import pyqt5_library_info as _qt_info
_qt_base = pathlib.Path(_pyqt5.__file__).resolve().parent / 'Qt5'
for _k, _v in {
    'PrefixPath': _qt_base,
    'BinariesPath': _qt_base / 'bin',
    'PluginsPath': _qt_base / 'plugins',
    'TranslationsPath': _qt_base / 'translations',
}.items():
    _qt_info.location[_k] = str(_v)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    # 数据：.arcdata 种子（68 MB，首启复制到 %LOCALAPPDATA%/EnergyLevel/.arcdata，
    #   见 src/__init__.py 的 sys.frozen 分支）+ arc 包 NIST 数据。
    # 注：软件本体在 app_entry.py（随 main.py 依赖图编译进包），
    #   不把 能级.py 作为磁盘脚本执行，无需加入 datas。
    datas=[('.arcdata', '.arcdata')] + collect_data_files('arc'),
    hiddenimports=[
        'arc.alkali_atom_functions',
        'arc.alkali_atom_data',
        'arc.wigner',
        'arc.calculations_atom_single',
        'arc.calculations_atom_pairstate',
        'arc._database',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'torch', 'pairinteraction', 'qutip', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── 手动补收 Qt 平台/样式/图像插件 ──────────────────────────────────────
# PyInstaller 的 Qt 钩子靠"插件关联表"自动收集；中文路径下该表偶发加载不全
# （plugins 一个都不收，Qt 启动即 "Could not find the Qt platform plugin"）。
# 显式按 PyQt5 wheel 布局收进包（落在 Qt5Core.dll 相对前缀可自动发现的
# PyQt5/Qt5/plugins/<type>），并去重防冲突。
import glob as _glob
import os
import posixpath as _ppath
_plugin_root = _qt_info.location.get('PluginsPath', str(_qt_base / 'plugins'))
for _pt in ('platforms', 'imageformats', 'styles', 'accessible', 'iconengines'):
    _dest = _ppath.join('PyQt5', 'Qt5', 'plugins', _pt)
    for _f in sorted(_glob.glob(os.path.join(_plugin_root, _pt, '*.dll'))):
        if not any(os.path.normcase(b[1]) == os.path.normcase(_f) for b in a.binaries):
            a.binaries.append((_dest, _f, 'BINARY'))

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EnergyLevel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='EnergyLevel',
)