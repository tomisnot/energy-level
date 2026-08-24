# -*- mode: python ; coding: utf-8 -*-
"""控制台调试版 spec（专用于打包后排错）：与 EnergyLevel.spec 完全相同的
Analysis/datas/excludes，仅 console=True 且产物名 EnergyLevel-console。
控制台版把 Python 报错直接打印到终端（发行版 windowed 模式只弹框，
无桌面环境下无法观察）。排查完用发行 spec 重新打包即可。
"""
from PyInstaller.utils.hooks import collect_data_files

# ── Qt 路径修正（中文路径下 QLibraryInfo 偶发编坏路径，见 EnergyLevel.spec 注释）──
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
    runtime_hooks=['_sim_windowed.py'],
    excludes=['tkinter', 'torch', 'pairinteraction', 'qutip', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── 手动补收 Qt 平台/样式/图像插件（自动收集在中文路径下偶发遗漏，见发行 spec）──
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
    name='EnergyLevel-console',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
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
    name='EnergyLevel-console',
)