# -*- coding: utf-8 -*-
"""MCP streamable-http 客户端（stdlib urllib，零新依赖）——GUI 作 Hub 客户端的传输层。

Hub（`src.adapter.serve`）把 mecha 暴露成 MCP server（stateful：强制 Mcp-Session-Id）。
本客户端做最小 JSON-RPC over HTTP：initialize 拿 session id → tools/call 调工具。
响应可能是 JSON 或 SSE（text/event-stream），两者都解析。

**定位**：GUI Hub 客户端化（Phase 1）的传输层；也是任何"外部程序想当 mecha 客户端"
的参考实现。只读/写都经 Hub 的 choke point（与 AI 同 session、同事件总线）。
"""
import json
import urllib.error
import urllib.request

__all__ = ['HubClient', 'HubError']


class HubError(RuntimeError):
    """Hub 连接/调用失败（带可教学消息）。"""


def _post(url, body, headers=None, timeout=60.0):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json',
                 'Accept': 'application/json, text/event-stream',
                 **(headers or {})},
        method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = (r.headers.get('Content-Type') or '').split(';')[0].strip()
        sid = r.headers.get('Mcp-Session-Id')
        raw = r.read().decode('utf-8')
    if ctype == 'text/event-stream':
        return _parse_sse(raw), sid
    return json.loads(raw) if raw.strip() else None, sid


def _parse_sse(raw):
    """取 SSE 流里最后一条 data: 的 JSON-RPC 响应。"""
    last = None
    for line in raw.splitlines():
        if line.startswith('data:'):
            chunk = line[5:].strip()
            if chunk:
                try:
                    last = json.loads(chunk)
                except ValueError:
                    pass
    return last


class HubClient:
    """连一个 mecha Hub（MCP）。with 语义：close() 结束 session。"""

    def __init__(self, url, timeout=60.0):
        self.url = url.rstrip('/')
        if not self.url.endswith('/mcp'):
            self.url += '/mcp'
        self.timeout = timeout
        self.sid = None
        self._rid = 0

    # ---------------- 生命周期 ----------------
    def initialize(self):
        body, sid = _post(self.url, {
            'jsonrpc': '2.0', 'id': self._next(), 'method': 'initialize',
            'params': {'protocolVersion': '2025-03-26',
                       'capabilities': {},
                       'clientInfo': {'name': 'mecha-gui-client', 'version': '0.1'}}},
            timeout=self.timeout)
        self.sid = sid
        # notified initialized（无响应体）
        try:
            _post(self.url, {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                  headers={'Mcp-Session-Id': self.sid} if self.sid else None,
                  timeout=self.timeout)
        except Exception:                                  # noqa: BLE001
            pass
        return body

    def close(self):
        self.sid = None

    def _next(self):
        self._rid += 1
        return self._rid

    # ---------------- 调用 ----------------
    def call(self, name, **args):
        """调一个 MCP 工具 → 解析后的 content（dict/str）。失败抛 HubError（可教学）。"""
        body, _ = _post(self.url, {
            'jsonrpc': '2.0', 'id': self._next(), 'method': 'tools/call',
            'params': {'name': name, 'arguments': args}},
            headers={'Mcp-Session-Id': self.sid} if self.sid else None,
            timeout=self.timeout)
        return self._unwrap(body, name)

    def list_tools(self):
        body, _ = _post(self.url, {
            'jsonrpc': '2.0', 'id': self._next(), 'method': 'tools/list',
            'params': {}},
            headers={'Mcp-Session-Id': self.sid} if self.sid else None,
            timeout=self.timeout)
        return (body or {}).get('result', {}).get('tools', [])

    @staticmethod
    def _unwrap(body, name):
        if body is None:
            raise HubError('工具 %s 无响应' % name)
        if 'error' in body:
            e = body['error']
            raise HubError('工具 %s 失败：%s' % (name, e.get('message', e)))
        res = (body or {}).get('result', {})
        content = res.get('content') or []
        # 取第一段 text 的 JSON（mecha 工具回程是 JSON 字符串）
        for c in content:
            if c.get('type') == 'text':
                try:
                    return json.loads(c['text'])
                except ValueError:
                    return c['text']
        return res


def discover_hub(port_file='.mcp-port'):
    """读端口文件 → Hub base url；无则 None（GUI 退回 in-process）。"""
    import os
    try:
        with open(port_file, encoding='utf-8') as f:
            port = int(f.read().strip())
        return 'http://127.0.0.1:%d' % port
    except (OSError, ValueError):
        return None
