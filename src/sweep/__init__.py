# -*- coding: utf-8 -*-
"""并行扫描（CPU 多进程）子包。

模块地图：
  spec.py      声明式轴展开 + 登记册路径寻址（纯函数，无 Qt）
  runner.py    **worker 侧**执行器：进程池、共享内存进度、逐点落盘
  registry.py  **提交侧**（GUI / MCP）：建目录、写规格、起 runner、查状态/结果/取消

分层原则：worker 子进程只依赖 `spec` + `src.core.*`，**不 import PyQt5**；
GUI 只经 `registry` 与 runner 交互（进程边界 + 文件），不共享内存对象。
"""
