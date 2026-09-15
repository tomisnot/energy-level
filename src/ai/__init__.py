# -*- coding: utf-8 -*-
"""AI 协同适配层（Energy Level 软件即 Skill）。

模块地图：
  dispatch.py   主线程派发器：MCP 线程 → 主线程执行 + 回传结果
  journal.py    操作日志（Op）：谁改的/改哪/前后值/原因，支持 undo
  sim_state.py  状态门面：读写经现有 store/序列化路径，人机等效
  server.py     进程内 MCP server（streamable HTTP），暴露语义工具
  codeact.py    CodeAct 通道：受限 run_python，一次调用完成多步编排
  run_store.py  CodeAct 运行记录落盘（runtime/ai_results/）：算完不丢、断线可取回

设计依据见 docs/AI协同-项目目标与原则.md、docs/上帝模式-AI接入设计.md
（原《AI协同-实现方案草案.md》2026-09-14 已并入 docs/历史-任务报告汇编.md）。
"""
