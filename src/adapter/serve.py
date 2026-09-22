# -*- coding: utf-8 -*-
"""MCP 写通道 launcher（领域侧）——注入 `EngineAdapter` 后起 `mecha.server`。

人运行：`python -m src.adapter.serve`。mecha 不认领域，故"构造引擎"的 launcher 住在
adapter（唯一可同时 import core 与 mecha 的层）。写通道仅 localhost（T6）；harness 靠
`.mcp-port` 端口文件发现并 attach（T4：人启动 = 权威，harness 绝不拉起）。
"""
import argparse
import asyncio
import atexit
import http.server
import json
import os
import secrets
import sys
import threading

from ..mecha import registry, runtime
from ..mecha.ops import Ops
from ..mecha.server import (create_server, find_free_port, run_streamable_http,
                            write_port_file)
from ..mecha.session import Session
from . import EngineAdapter

__all__ = ['main']


class _HumanControl(http.server.BaseHTTPRequestHandler):
    """人工控制端点（非 MCP）：switch_mode 是**人的动作**（AI 不能自授权），故不经 MCP，
    走独立 token 鉴权的 HTTP 端点。GUI 作 Hub 客户端时经此夺权。"""
    session = None
    token = None

    def do_POST(self):
        if self.path != '/switch_mode':
            return self._json(404, {'ok': False, 'error': 'unknown endpoint'})
        n = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(n) or b'{}')
        if body.get('token') != type(self).token:
            return self._json(403, {'ok': False, 'error': 'bad token'})
        mode = body.get('mode')
        reason = body.get('reason', 'human switch')
        try:
            type(self).session.authority.switch(
                mode, actor='human', reason=reason, in_flight=0)
            self._json(200, {'ok': True, 'mode': mode})
        except Exception as e:                              # noqa: BLE001
            self._json(400, {'ok': False, 'error': str(e)})

    def _json(self, code, obj):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):                            # 静音
        pass


def start_human_control(session, host='127.0.0.1', port=8767):
    """起人工控制端点；token 写 .human-token（GUI 读它夺权）。返回 (port, token)。"""
    token = os.environ.get('RE0_HUMAN_TOKEN') or secrets.token_urlsafe(16)
    _HumanControl.session = session
    _HumanControl.token = token
    srv = http.server.ThreadingHTTPServer((host, port), _HumanControl)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with open('.human-token', 'w', encoding='utf-8') as f:
        f.write(token)
    return srv.server_address[1], token


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m src.adapter.serve')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--atom', default='Rubidium87')
    ap.add_argument('--stdio', action='store_true',
                    help='用 stdio 传输（默认 streamable-http + 端口文件）')
    ap.add_argument('--allow-remote-write', action='store_true',
                    help='远程写（绑 0.0.0.0 + token；仅仿真 L2；硬件 L3 仍需本地互锁+人确认）')
    a = ap.parse_args(argv)

    if os.environ.get('RE0_MECHA_NO_SERVER'):
        print('RE0_MECHA_NO_SERVER 置位 → 不启动 MCP 写通道')
        return 0

    # Windows：用 SelectorEventLoop 避开 Proactor 在客户端突然断连（dsh 自愈桥重连/
    # 浏览器关 SSE）时 _call_connection_lost 刷 WinError 10054 的无害噪声。本进程不在
    # 事件循环里 spawn 子进程，切 Selector 安全。
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    runtime.set_engine(EngineAdapter(atom=a.atom))   # 注入领域引擎
    session = Session()
    # 落一条 registry_schema 事件：驾驶舱配置态结构树的 schema 源（需求②）。cockpit 只读
    # journal 即可富渲染、不 import mecha（黑匣子纯净 T5）；大 schema 被 I4 体积闸截成
    # 指针，cockpit 端顺 where 读回原文。
    session.bus.append(
        actor='system', kind='registry_schema', target='registry',
        after=registry.list_params(include_internal=True,
                                    page_size=100000)['params'],
        reason='配置态 schema（供驾驶舱结构树）',
        provenance='adapter.serve', reversibility='none')
    ops = Ops(session)
    hc_port, _hc_token = start_human_control(session)
    print('人工控制端点：http://%s:%d/switch_mode（token 在 .human-token）'
          % (a.host, hc_port), flush=True)
    srv, _ = create_server(session, ops)

    if a.stdio:
        print('MCP 写通道：stdio 传输', flush=True)
        asyncio.run(srv.run_stdio_async())
        return 0

    port = find_free_port(a.port)
    pf = write_port_file(port)
    atexit.register(lambda: os.path.exists(pf) and os.remove(pf))
    host = '0.0.0.0' if a.allow_remote_write else a.host
    if a.allow_remote_write:
        tok = secrets.token_urlsafe(16)
        with open('.mcp-token', 'w', encoding='utf-8') as f:
            f.write(tok)
        print('⚠ 远程写已开（绑 0.0.0.0，token 在 .mcp-token）。仅仿真 L2 可远程；'
              '硬件 L3 仍需本地互锁+人确认（铁律二）。', flush=True)
    print('MCP 写通道：http://%s:%d/mcp（端口文件 %s）' % (host, port, pf),
          flush=True)
    print('写通道仅 localhost（T6）；驾驶舱只读订阅端点另起。', flush=True)
    try:
        asyncio.run(run_streamable_http(srv, host, port))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
