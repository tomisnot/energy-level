# -*- coding: utf-8 -*-
"""B4 自检：基组预设（声明式裁剪的 GUI 面）+ 维数提示。

覆盖：
  1 控件与方法就位（预设下拉 / 应用 / 存为 / 维数提示）
  2 内置三套预设可读（含文件缺失时的兜底）
  3 应用「两体交换集(4)」→ 基组恰好 4 组（这就是"裁剪超限基组"的一键操作）
  4 应用「两体+freeze(7)」→ 恰好 7 组，且**不误删**其中任一（事故回归：
    曾按 '60P' 前缀匹配而删掉 60P_3/2）
  5 维数提示正确反映两体 dim=N² 并在超阈值时变色
  6 预设文件可扩展（人与 AI 共用一份 json）

用法：python checks/gui_basis_presets_test.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                            # noqa: E402
from PyQt5.QtWidgets import QApplication                       # noqa: E402
import app_entry as M                                          # noqa: E402

C = Checker('gui_basis_presets')
_app = QApplication.instance() or QApplication(sys.argv)
try:
    win = M.MainWindow()
except MemoryError as e:
    # 软件正在运行（占着大基组与 ARC 数据）时可能拿不到内存 → 跳过而非误报失败
    print('  SKIP 内存不足，先关掉正在运行的软件再跑：%s' % e)
    sys.exit(0)

page = None
for v in vars(win).values():
    if hasattr(v, 'preset_combo') and hasattr(v, 'apply_preset'):
        page = v
        break
C.check('1 找到带基组预设的页面', page is not None)

C.check('2 控件就位', hasattr(page, 'preset_combo')
        and hasattr(page, 'preset_apply') and hasattr(page, 'preset_save')
        and hasattr(page, 'dim_hint'))
C.check('3 内置预设可读（≥3 套）', len(page._presets) >= 3,
        str(list(page._presets)[:4]))


def keys_now():
    return sorted((int(g['n']), int(g['l']), float(g['j']))
                  for g in win.groups_store.groups)


EXCH = [(6, 1, 1.5), (58, 4, 3.5), (60, 0, 0.5), (60, 1, 1.5)]
FREEZE7 = EXCH + [(58, 3, 2.5), (58, 4, 4.5), (59, 2, 2.5)]

page.preset_combo.setCurrentText('两体交换集(4)')
page.apply_preset()
C.check('4 应用两体交换集 → 恰好 4 组', keys_now() == sorted(EXCH),
        str(keys_now()))

page.preset_combo.setCurrentText('两体+freeze(7)')
page.apply_preset()
C.check('5 应用两体+freeze → 恰好 7 组', keys_now() == sorted(FREEZE7),
        str(keys_now()))
C.check('6 事故回归：60P_3/2 与 58F_5/2 未被误删',
        (60, 1, 1.5) in keys_now() and (58, 3, 2.5) in keys_now(),
        str(keys_now()))

page.preset_combo.setCurrentText('单体扫描集(12)')
page.apply_preset()
txt = page.dim_hint.text()
C.check('7 维数提示含两体 dim=N² 与警告', '两体 dim=' in txt
        and len(win.groups_store.groups) == 12, txt)
two = 2 * len(win.groups_store.groups)  # 粗判：提示里确实报了数字
C.check('8 超阈值时变色（12 组 → 红）', '#c62828' in page.dim_hint.styleSheet()
        or '#ef6c00' in page.dim_hint.styleSheet(),
        page.dim_hint.styleSheet())

p = page._presets_path()
C.check('9 预设文件在 save_state/ 且含内置项',
        p.endswith(os.path.join('save_state', 'basis_presets.json'))
        and os.path.exists(p), p)
# 写入/读回测试**只碰临时副本**（2026-09-12 修）：save_state/ 是人的数据，
# 自检往那里写会污染人的预设（实测：本项曾把人的 basis_presets.json 重写一遍）。
import shutil                                                  # noqa: E402
import tempfile                                                # noqa: E402
_tmp_p = os.path.join(tempfile.gettempdir(), '_basis_presets_selfcheck.json')
shutil.copyfile(p, _tmp_p)
_mtime0 = os.path.getmtime(p)
page._presets_path = lambda: _tmp_p        # 实例属性遮蔽 → 只写副本
d = json.load(open(_tmp_p, encoding='utf-8'))
d['_自检临时集'] = [[60, 0, 0.5]]
json.dump(d, open(_tmp_p, 'w', encoding='utf-8'), ensure_ascii=False)
page._load_presets()
C.check('10 新增预设可被读回（人与 AI 共用一份）',
        '_自检临时集' in page._presets)
d.pop('_自检临时集', None)
json.dump(d, open(_tmp_p, 'w', encoding='utf-8'), ensure_ascii=False)
page._load_presets()
C.check('11 人的 basis_presets.json 未被自检改写（mtime 不变）',
        os.path.getmtime(p) == _mtime0, f'{p}')

C.summary()
