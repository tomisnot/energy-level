# -*- coding: utf-8 -*-
"""EL ↔ mecha v2 的接线层（Phase 5 第 2 步：GUI 与 AI 过同一道 v2 门）。

设 `RE0_V2=1` 时 app_entry 用它取代 v1 的 AppBridge：一个 V2Bridge.for_shared
同时服务 GUI（主线程直接读写）与 AI（后台 MCP 服务器线程），**共一 Gate、一份
History**。cache 经 CacheProxy 汇流到引擎线程（ARC SQLite 认线程，B-1①）。

另起一个**人工控制端点**（镜像 v1 src/adapter/serve 的形状）：人的夺权/交权/
写走它、actor 由服务端固定为 human，AI 面（MCP）改不了 mode 也伪造不了 human
归因——这两条与 v1 同源，是"AI 不能自授权"的护栏。

## Known Limitations and Deferred Work

- 依赖 mecha v2 在 sys.path 上（app_entry 的 v2 分支负责插入；或由 editable
  安装 mecha==0.2 顶掉 v1）。v1/v2 同名 mecha，**同进程只能有一个**——本层
  假设 v2 已生效，不兼容同进程 import v1 的 mecha.ops/session 等。
- gc（数据回收）在 v2 未实现（D-9 保留策略等第二项目）；GUI 的回收按钮在
  v2 下会显示"未接线"而非静默，见 load_artifact 旁的 gc 垫片。
- undo/redo 未接：v2 History 是 append-only 事件流，撤销要按"反向 set 记一条
  新事件"实现（语义与 v1 的 session.undo 不同），等接 GUI 撤销按钮时写 ADR。
"""
from __future__ import annotations

import http.server
import json
import os
import pickle
import secrets
import sys
import threading
from pathlib import Path

# v2 仓根：环境变量优先；否则试两个已知布局（并列目录 / code-nosync）
def _find_v2_root() -> str:
    env = os.environ.get("MECHA_V2_ROOT")
    if env:
        return env
    candidates = [
        str(Path(__file__).resolve().parents[2] / "mecha"),      # .../python/mecha
        r"D:\code-nosync\mecha",
    ]
    for c in candidates:
        if (Path(c) / "mecha" / "gate.py").exists():
            return c
    return candidates[0]


_MECHA_V2 = _find_v2_root()


def _ensure_v2_path() -> None:
    if _MECHA_V2 and _MECHA_V2 not in sys.path:
        sys.path.insert(0, _MECHA_V2)


_ensure_v2_path()


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


# ---------------- 人工控制端点（镜像 v1，actor 服务端固定 human）----------------

class _HumanHandler(http.server.BaseHTTPRequestHandler):
    bridge = None
    token = None

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json(400, {"ok": False, "error": "bad json"})
        if body.get("token") != type(self).token:
            return self._json(403, {"ok": False, "error": "bad token"})
        br = type(self).bridge
        reason = body.get("reason", "human write")
        try:
            if self.path == "/switch_mode":
                br.switch_mode(body.get("mode"), actor="human")
                return self._json(200, {"ok": True, "mode": br.mode})
            if self.path == "/write":
                rec = br.write(body.get("key"), body.get("value"),
                               reason=reason, actor="human")
                return self._json(200, rec)
            if self.path == "/write_many":
                br.write_many(body.get("patch") or {}, reason=reason, actor="human")
                return self._json(200, {"ok": True})
            return self._json(404, {"ok": False, "error": "unknown endpoint"})
        except Exception as e:                              # noqa: BLE001
            kind = getattr(e, "kind", "")
            hint = getattr(e, "hint", "")
            return self._json(400, {"ok": False, "error": {
                "message": str(e), "kind": kind, "hint": hint}})

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                              # 静音
        pass


def start_human_control(bridge, host="127.0.0.1", port=8767, token_file=".human-token"):
    """起人工控制端点；token 写文件（人夺权用）。返回 (port, token)。"""
    token = os.environ.get("RE0_HUMAN_TOKEN") or secrets.token_urlsafe(16)
    _HumanHandler.bridge = bridge
    _HumanHandler.token = token
    srv = http.server.ThreadingHTTPServer((host, port), _HumanHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Path(token_file).write_text(token, encoding="utf-8")
    return srv.server_address[1], token


# ---------------- 装配：给 app_entry 用 ----------------

def make_v2_bridge(el_root, *, data_root=None, data_dir_name=".mecha-v2",
                   start_services=True, mcp_port_file=None):
    """构造共享 Gate 的 V2Bridge；可选起 MCP 服务器 + 人工控制端点。

    app_entry 的 v2 分支调它，拿到的对象即 v1 AppBridge 的 drop-in。
    """
    from examples.energy_level.bridge import V2Bridge
    from mecha.authority import Mode
    data_root = data_root or el_root
    bridge = V2Bridge.for_shared(el_root=el_root, data_root=data_root,
                                 data_dir_name=data_dir_name, mode=Mode.HUMAN)
    bridge._mcp_thread = None
    bridge._mcp_server = None
    if start_services:
        pf = mcp_port_file or str(Path(el_root) / ".mcp-port")
        _t, server, port = bridge.serve_mcp(port_file=pf)
        bridge._mcp_thread, bridge._mcp_server, bridge._mcp_port = _t, server, port
        hc_port, token = start_human_control(
            bridge, token_file=str(Path(el_root) / ".human-token"))
        bridge._human_port = hc_port
        print(f"[v2_bind] MCP 服务器 :{port}/mcp（工具经同一道 Gate）；"
              f"人工控制 :{hc_port}/switch_mode（token 在 .human-token）")
    return bridge
