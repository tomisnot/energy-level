# -*- coding: utf-8 -*-
"""PyInstaller runtime hook（仅 console 排错版使用）：
设置环境变量 ENERGYLEVEL_SIM_WINDOWED=1 时，把 sys.stdout/stderr 置为
None，复刻发行版（windowed/console=False）的"无控制台"运行条件——
用于在无桌面的环境里可靠验证 windowed 特有的 stdio=None 崩溃
（console 版报错会直接打印并以非零码退出，存活即表示 windowed 路径健康）。
"""
import os
import sys

if os.environ.get('ENERGYLEVEL_SIM_WINDOWED'):
    sys.stdout = None
    sys.stderr = None