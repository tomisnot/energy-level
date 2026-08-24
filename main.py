# -*- coding: utf-8 -*-
"""PyInstaller 打包入口：冻结模块方式启动 app_entry.main()。

历史：曾用 runpy.run_path 执行 能级.py——冷冻环境下磁盘脚本的 import
解析与常规模块不同（PyQt5 等解析失败），已弃用。现改为普通 import，
走 PyInstaller 标准机制；app_entry.py 会随 main.py 的依赖图被编译进包。
"""
import sys

from app_entry import main

sys.exit(main())