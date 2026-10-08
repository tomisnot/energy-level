# -*- coding: utf-8 -*-
"""控制通道（**软件侧**）：第二条 MCP 端点 —— **人侧专用**（三层架构 §7.1.1 契约）。

## 契约（照 `docs/v2/软件-交互-写权-三层架构.md` §7.1.1 实现，两端别各写一套）

| 项 | 约定 |
|---|---|
| 端口文件 | ``<svc_root>/.control-port``（与 ``.mcp-port`` **并列、同名族**） |
| 传输 | MCP over streamable-http —— ⭐ **复用框架 ``McpEndpoint``**，不发明第二种协议 |
| 谁连 | **只有框架/面板/人手腿** ⇒ ⚠ **不导出给 AI** |
| 操作 | ``set_write_authority(mode)`` · ``open_manual_surface()`` |
| 入参 | ``mode`` 是**裸值** ``locked``/``human``/``ai``/``open`` |
| 记账 | 两个操作都**记同一本账**（§4 的纪律） |
| 失败 | 未知 mode / 没有手动面 ⇒ **响亮失败** |

## ⚠ 为什么必须**另起一条端点**（不能"给 AI 那条加过滤"）

``build_mcp_server`` 吃**整个 registry**（``for schema in registry.schemas()``）
⇒ **一条端点 = 一个面**，同一个 registry 给不出两套可见面。⇒ 可见性由**端点的存在性**
保证，**不靠记得加过滤**（忘了过滤会**静默**发生）。

## ⚠ 记账为什么不走 ``Gate.set``

``set_write_authority`` 改的东西**就是"谁能写"**：若它自己也过一次写权门，就会出现
"切到 ``ai`` 之后人侧记不了账、也再切不回来"的**自锁**。⇒ 用
``bridge.record_audit``（``Gate.record``：**只记史、不改快照、不过写权门**），
账仍然是**同一本**（§4 要的是"同一本账"，不是"同一道门禁"）。

## ⚠ 地址不许外泄

``.control-port`` 只被框架读。本文件里**任何日志/工具输出都不打印端口号**
（§7.1.1 明写：不许把 ``:control-port`` 写进任何给 AI 的文本/工具输出）。
"""

from __future__ import annotations

from mecha.authority import Mode
from mecha.errors import MechaError
from mecha.tools import ToolRegistry, define_tool

#: 控制端点的端口文件（契约 §7.1.1：与 `.mcp-port` 并列、同名族，写在软件根）。
CONTROL_PORT_FILE = ".control-port"

#: 端点服务名（⚠ 只是服务名，**不是**给 AI 的工具前缀——这条端点不导出给 AI）。
CONTROL_SERVER_NAME = "re0-mecha-control"

#: 端点说明（框架会校验体积；⚠ **不写地址、不写端口**）。
CONTROL_INSTRUCTIONS = (
    "本端点供**人侧**（框架 / 面板 / 人手腿）控制这份软件：切写权、按需开手动面。"
    "它不是给模型用的能力面——能力面在另一条端点上，两者共用同一份门与同一本账。"
)


def build_control_registry(bridge, *, open_surface=None) -> ToolRegistry:
    """建控制面的工具表（两个操作）。

    ``open_surface`` 是宿主注入的"开手动面"可调用件（见 ``app_entry``：它负责把这件事
    搬到 **Qt 主线程**去做）。``None`` ⇒ 本进程没有手动面能力 ⇒ 调用时**响亮失败**。
    """
    reg = ToolRegistry()

    def _set_write_authority(mode):
        raw = str(mode).strip().lower()
        try:
            m = Mode(raw)
        except ValueError:
            raise MechaError(
                "未知写权模式：%r" % (mode,), kind="bad_mode",
                hint="裸值只认 %s（⚠ 不是 `str(Mode.AI)` 那种 'Mode.AI' 形状）"
                     % "/".join(x.value for x in Mode))
        # ⚠ 一律 `.value`：`Mode` 是 **str-Enum**（`str(Mode.AI)` 给 'Mode.AI' 不是 'ai'）。
        bridge.switch_mode(m.value, actor="human")
        bridge.record_audit("control.write_authority", m.value, actor="human",
                            reason="人侧切写权（控制端点）")
        return {"ok": True, "mode": m.value}

    def _open_manual_surface():
        if open_surface is None:
            raise MechaError(
                "本进程没有可开的「手动面」（未注入 open_surface）",
                kind="no_manual_surface",
                hint="只有带 GUI 的形态才有人手腿（如 `--mcp --with-gui`）")
        status = str(open_surface())
        bridge.record_audit("control.open_manual_surface", status, actor="human",
                            reason="人侧按需开窗（控制端点）")
        return {"ok": True, "status": status}

    reg.register(define_tool(
        name="set_write_authority",
        description=("切换这份软件的写权归属（人侧专用）。切完**立刻生效**："
                     "下一次写权判定就用新值（不允许'等一拍'）。"),
        parameters={"mode": {"type": "string",
                             "description": "裸值：locked / human / ai / open"}},
        output_schema={"type": "object", "required": ["ok", "mode"]},
        execute=lambda mode: _set_write_authority(mode),
    ))
    reg.register(define_tool(
        name="open_manual_surface",
        description=("在本进程内打开「手动面」（GUI 窗口，**复用已装配的那份桥**）。"
                     "已经开着 ⇒ 幂等返回 already-open（不重复开、不报错）。"),
        parameters={},
        output_schema={"type": "object", "required": ["ok", "status"]},
        execute=lambda: _open_manual_surface(),
    ))
    return reg


def start_control_endpoint(registry, *, port_file, host="127.0.0.1", log=None):
    """起控制端点（框架 ``McpEndpoint``；**端口回读**，与能力面同款）。

    返回 ``(endpoint, actual_port)``。⚠ 端口号**不要**打印/回显给 AI（§7.1.1）：
    它只写进 ``port_file`` 给框架读。
    """
    from mecha.providers.mcp import McpEndpoint
    ep = McpEndpoint(registry, server_name=CONTROL_SERVER_NAME,
                     instructions=CONTROL_INSTRUCTIONS, host=host, port=0,
                     port_file=str(port_file), log=log)
    return ep, ep.start()


__all__ = ["CONTROL_PORT_FILE", "CONTROL_SERVER_NAME", "build_control_registry",
           "start_control_endpoint"]
