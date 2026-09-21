# -*- coding: utf-8 -*-
"""驾驶舱（cockpit）——**独立只读**的 Web 投影（纲领 §2 投影层 / T5 黑匣子）。

一个独立进程（`python -m src.cockpit.server`），读磁盘 journal、投影成 HUD 归因灯 /
告警灯 / 飞行记录仪。**无自己的真值、无写回通道**（I2；AST 守卫看住）。

    backend.py   只读 journal + 告警规则（例外驱动 M5，每条可被构造点亮）
    server.py    独立 Starlette/uvicorn 进程（static + /api/* 只读 + SSE；可远程鉴权）
    static/      浏览器前端（HUD / 告警 / 飞行记录仪；diff-since-seq 增量）
"""

__all__ = ['backend', 'server']
