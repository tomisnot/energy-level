# -*- coding: utf-8 -*-
"""EL 的控制面**接线** —— 机制在框架，本文件只给**项目知识**。

⚠ **本文件不许再出现机制实现**（判据 `checks/control_channel_selfcheck.py` 钉着）：
三个操作的工具定义、端点启动、端口回读、端口文件、`bad_mode`/`no_manual_surface`
的响亮失败 —— 全在框架 `mecha.software_control`。⭐ 本项目只交
**3 个回调 + 2 个值**（`authority` · `switch` · `audit` · `open_surface` · `how_to_open`
· `server_name`）。⚠ 别抄一份回来：EL 与 GP 各手写 ≈134 行同构实现，正是要消灭的东西。

| 交给框架的 | EL 的值 | 依据 |
|---|---|---|
| `authority` | `bridge.authority` | 只读自省要探**真闸门**（框架用 `gate(side)` 纯查询） |
| `switch(v)` | `bridge.switch_mode(v, actor="human")` | 人侧切写权（裸值；⚠ `Mode` 是 str-Enum，一律 `.value`） |
| `audit(op, val, why)` | `bridge.record_audit(op, val, actor="human", reason=why)` | ⚠ **只记史、不过写权门**（`Gate.record`）—— 切写权本身就改门，过门会**自锁**（EL 踩过） |
| `open_surface()` | `app_entry._open_on_qt_thread`（**搬到 Qt 主线程**那条路） | Qt 要求 GUI 对象/事件循环属于创建 `QApplication` 的那条线程 |
| `how_to_open` | `HOW_TO_OPEN`（本项目文案） | 框架**不写死**它：怎么开手动面是项目知识 |
| `server_name` | `CONTROL_SERVER_NAME` | 框架里不许有项目名 |
"""

from __future__ import annotations

#: ⭐ 从框架转出（单一事实：端点端口文件名由框架拥有）。
from mecha.software_control import CONTROL_PORT_FILE, build_control_registry

#: 控制端点服务名（⚠ 只是服务名，不是给 AI 的工具前缀——这条端点不导出给 AI）。
CONTROL_SERVER_NAME = "re0-mecha-control"

#: ⭐ 由**本项目**给的「怎么切到人工」说明（框架不写死；出现在 `read_write_authority()`
#: 的回执里，给**被拒的一方**指路）。
HOW_TO_OPEN = (
    "面板（常驻控制台）上该软件的「切到手动」按钮：按一下写权即归人"
    "（本机可信、无口令），再按「切回 AI」交还模型；"
    "纯本地用户不经 mecha，直接 `python app_entry.py --gui`（独立进程、不起端点）。"
)


def wire_control(bridge, *, open_surface=None):
    """把 EL 的**回调**接进框架的控制面（机制、注册、返回字段都在框架里）。"""
    return build_control_registry(
        bridge.authority,
        switch=lambda v: bridge.switch_mode(v, actor="human"),          # noqa: E731
        audit=lambda op, val, why: bridge.record_audit(op, val, actor="human",
                                                       reason=why),     # noqa: E731
        open_surface=open_surface,
        how_to_open=HOW_TO_OPEN,
    )


__all__ = ["CONTROL_PORT_FILE", "CONTROL_SERVER_NAME", "HOW_TO_OPEN", "wire_control"]
