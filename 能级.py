# -*- coding: utf-8 -*-
"""开发入口（功能与旧版 能级.py 完全一致，软件本体已移至 app_entry.py）。

本文件只做转发：
- 开发运行：python 能级.py
- 打包运行：main.py（PyInstaller 入口）→ app_entry.main()
checks/ui_selftest.py、ui_shot.py 以 `import 能级 as main_mod` 方式引用
MainWindow，故在此一并再导出。
"""
import sys

from app_entry import MainWindow, main

if __name__ == '__main__':
    sys.exit(main())