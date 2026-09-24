# -*- coding: utf-8 -*-
"""EL ↔ mecha v2 的接线层：两模式收拢后的装配点（v1 门面已退役）。

app_entry 用它装配 V2Bridge.for_shared。**两模式**（用户裁决 2026-09-24）：
- GUI 模式：`start_services=False`——纯本地软件，不起 MCP/监控端点，AI 物理
  看不见（写权恒 human）。
- AI 模式：headless 权威（run_authority）用默认 True——起 MCP 服务器线程 +
  只读监控端点，写权自动落 AI（看门人代理人意图开闸）。

无论哪种模式，所有写都经**同一道 Gate、同一份 History**。cache 经 CacheProxy
汇流到引擎线程（ARC SQLite 认线程，B-1①）。

监控端点是**只读**的（/status + /activity）：给 dsh 右侧栏面板轮询"谁改了什么"。
控制权切换不经它——人的夺权/交权由看门人（launcher.py）编排进程交接；"AI 不能
自授权"的护栏靠 Authority 的 side 判定（AI 通道改不了 mode），不靠端点鉴权。

## Known Limitations and Deferred Work

- 依赖 mecha v2 已安装（editable 安装 mecha==0.2）。v1/v2 同名 mecha，
  **同进程只能有一个**——本层假设 v2 已生效，不兼容同进程 import v1 的
  mecha.ops/session 等。
- gc（数据回收）在 v2 未实现（D-9 保留策略等第二项目）；GUI 的回收按钮在
  v2 下会显示"未接线"而非静默，见 load_artifact 旁的 gc 垫片。
- undo/redo 未接：v2 History 是 append-only 事件流，撤销按"反向 set 记一条
  新事件"实现（决策见 mecha 仓 ADR `EL能力面取舍` 第 3 条；语义与 v1 的
  session.undo 不同），实现挂起至接 GUI 撤销按钮（真实调用方出现）。
"""
from __future__ import annotations

import http.server
import json
import os
import pickle
import threading
from pathlib import Path


def load_artifact(root, aid):
    """v2 artifact 读取（模块函数，兼容 GUI 的 `load_artifact(ops.root, aid)`）。

    返回 (labels, arrays, manifest) 三元组——GUI 取 [1]（arrays）绘图，与 v1
    mecha.ops.load_artifact 的元组形状对齐。
    """
    path = Path(root) / f"{aid}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        labels, arrays = pickle.load(f)
    return (labels, arrays, {})


def gc_report(root, *, dry_run=True, **_kw):
    """v2 数据回收垫片：未实现，如实报告（不静默）。GUI 回收按钮走这里。"""
    return {"ok": False, "dry_run": dry_run,
            "note": "mecha v2 未接 gc（保留策略等第二项目，见 D-9）；本目录未清理"}


# ---------------- 监控端点（只读：dsh 侧边栏面板轮询用）----------------
#
# 两模式收拢（2026-09-24）后，人的控制权切换改由**看门人**（launcher.py）编排，
# 本端点**降级为只读监控面**：只暴露"当前写权模式 + AI 最近做了什么"，供 dsh
# 右侧栏面板轮询。写操作 / 切模式的 POST 路由与 token 鉴权一并退役——
# "AI 不能自授权"的护栏不靠这个端点（靠 Authority 的 side 判定：AI 通道改不了
# mode，见 mecha/authority.py）。loopback-only（绑 127.0.0.1），读的是操作流水。

class _MonitorHandler(http.server.BaseHTTPRequestHandler):
    bridge = None

    def do_GET(self):
        if self.path == "/status":
            return self._json(200, {"ok": True, "mode": type(self).bridge.mode})
        if self.path.startswith("/activity"):
            # 增量拉事件流：?since_seq=N 返回 seq>=N 的写事件（dsh 面板轮询用）。
            since = 0
            q = self.path.partition("?")[2]
            for kv in q.split("&"):
                k, _, v = kv.partition("=")
                if k == "since_seq" and v.isdigit():
                    since = int(v)
            events = type(self).bridge.activity(since)
            return self._json(200, {"ok": True, "events": events,
                                    "mode": type(self).bridge.mode})
        # 退役面如实 404：无网页控制台、无写/切模式路由（控制权走看门人）
        self._json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self):
        # 只读端点：一切写/切模式 POST 退役（404）。人的控制权经看门人交接。
        self._json(404, {"ok": False, "error": "read-only monitor endpoint"})

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                              # 静音
        pass


def start_monitor_endpoint(bridge, host="127.0.0.1", port=8767):
    """起只读监控端点（/status + /activity?since_seq=）。返回端口。

    绑回环、无鉴权——读的是"谁改了什么"的操作流水，非域数据覆写面；写与切
    模式不经这里（控制权归看门人）。dsh 右侧栏面板按 seq 增量轮询。
    """
    _MonitorHandler.bridge = bridge
    srv = http.server.ThreadingHTTPServer((host, port), _MonitorHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


# ---------------- 装配：给 app_entry 用 ----------------

def make_v2_bridge(el_root, *, data_root=None, data_dir_name=".mecha-v2",
                   start_services=True, mcp_port_file=None):
    """构造共享 Gate 的 V2Bridge；可选起 MCP 服务器 + 只读监控端点。

    app_entry 调它：GUI 模式传 `start_services=False`（纯本地，不起任何端点）；
    headless 权威（run_authority）用默认 True（起 MCP + 监控端点，供 dsh 连与轮询）。
    数据根优先级：显式参数 > `RE0_V2_DATA_ROOT`（判据/测试用临时目录，
    不污染仓内权威 journal）> el_root。`EL_NO_AI_SERVER` 置位 → 不起后台
    服务（判据 in-process 用，避免与活实例抢端口/抢 .mcp-port——2026-09-12
    事故的同款防线）。
    """
    from examples.energy_level.bridge import V2Bridge
    from mecha.authority import Mode
    data_root = (data_root or os.environ.get("RE0_V2_DATA_ROOT") or el_root)
    if os.environ.get("EL_NO_AI_SERVER"):
        start_services = False
    bridge = V2Bridge.for_shared(el_root=el_root, data_root=data_root,
                                 data_dir_name=data_dir_name, mode=Mode.HUMAN)
    bridge._mcp_thread = None
    bridge._mcp_server = None
    if start_services:
        pf = mcp_port_file or str(Path(el_root) / ".mcp-port")
        _t, server, port = bridge.serve_mcp(port_file=pf)
        bridge._mcp_thread, bridge._mcp_server, bridge._mcp_port = _t, server, port
        mon_port = start_monitor_endpoint(bridge)
        bridge._monitor_port = mon_port
        print(f"[v2_bind] MCP 服务器 :{port}/mcp（工具经同一道 Gate）；"
              f"只读监控 :{mon_port}/activity（dsh 面板轮询）")
    return bridge
