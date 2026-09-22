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

from mecha import registry, runtime
from mecha.ops import Ops
from mecha.server import (create_server, find_free_port, run_streamable_http,
                            write_port_file)
from mecha.session import Session
from . import EngineAdapter

__all__ = ['main']


class _HumanControl(http.server.BaseHTTPRequestHandler):
    """**人的面**（非 MCP）：人的动作不经 MCP，走独立 token 鉴权的 HTTP 端点。

    为什么人的写不能走 MCP（P2-4，主代理裁定）：MCP 是 **AI 的面**；若给 MCP 工具加
    `actor` 参数，AI 就能传 `actor='human'` —— **同时**伪造人类归因（破 M1/M4）
    **并**绕过写权门（`authority.gate`）。与"AI 不能给自己授权"（switch_mode 不经 MCP）同源。

    端点（`POST`，body 里带 `token`）：
      · `/switch_mode`  —— 夺权/交权（actor 固定 `human`）
      · `/write`        —— 写一个键（actor 固定 `human`），回执与 `session.set` 同形
      · `/write_many`   —— 批量写（原子），回执为 set 回执列表
      · `/undo`         —— 撤销最近一次写（actor 固定 `human`）
    `actor` **由服务端固定为 `human`**，请求体里给什么都不认——这才堵得住伪造。
    """
    session = None
    token = None

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except ValueError:
            return self._json(400, {'ok': False, 'error': 'bad json'})
        if body.get('token') != type(self).token:
            return self._json(403, {'ok': False, 'error': 'bad token'})
        sess = type(self).session
        reason = body.get('reason', 'human write')
        try:
            if self.path == '/switch_mode':
                sess.authority.switch(body.get('mode'), actor='human',
                                      reason=reason, in_flight=0)
                return self._json(200, {'ok': True, 'mode': body.get('mode')})
            if self.path == '/write':
                rec = sess.set(body.get('key'), body.get('value'),
                               actor='human', reason=reason)
                return self._json(200, rec)
            if self.path == '/write_many':
                recs = sess.set_many(body.get('patch') or {}, actor='human',
                                     reason=reason)
                return self._json(200, {'ok': True, 'recs': recs})
            if self.path == '/undo':
                return self._json(200, sess.undo(actor='human', reason=reason))
            return self._json(404, {'ok': False, 'error': 'unknown endpoint'})
        except Exception as e:                              # noqa: BLE001
            # 可教学错误体（与 MCP 面同形），让人在 GUI 里看得到原因
            err = e.to_dict() if hasattr(e, 'to_dict') else {'message': str(e)}
            return self._json(400, {'ok': False, 'error': err})

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


def build_session(root, *, mode='human'):
    """**装配**一个 Hub 用的 Session（控制权门 + 数据根 + 引擎）。

    ★ 为什么把它抽成**具名函数**（主代理 2026-09-22）：原来这段埋在 `main()` 里，
    于是判据只能"自己造 session"去验端点机制，**从没验过启动器的装配**——
    结果 `main()` 里传的是 `Session(root=root)`（**没有 authority**），
    `authority.gate` 整段被跳过 ⇒ **Hub 进程里"两模式写权"这条不变量根本不存在**，
    而判据 `human_control_selfcheck` 却 7/0 全绿。**单元被测了，装配没被测。**
    抽出来之后，装配点变成**可被测的东西**（判据直接调本函数断言门是活的）。

    `mode='human'`（人先开车，与 `AppBridge` 默认一致）；交给 AI 由人经
    `/switch_mode`（独立 token，AI 不能自授权）。
    """
    from mecha.authority import Authority
    from mecha.bus import Bus
    bus = Bus(root=root)
    return Session(bus=bus, authority=Authority(bus=bus, mode=mode))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m src.adapter.serve')
    ap.add_argument('--root', default=None,
                    help='项目根：数据落 <root>/.mecha、端口文件写 <root>/.mcp-port'
                         '（默认 = 当前工作目录；也可用环境变量 MECHA_ROOT）')
    ap.add_argument('--port', type=int, default=None,
                    help='MCP 端口（默认让 OS 分配；**不再默认抢 8765**）')
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

    # 数据根**显式**取当前工作目录（launcher 约定从项目根跑），并注入环境变量，
    # 让同进程内其它组件（驾驶舱子进程等）拿到同一个根——不再从 __file__ 倒推。
    root = os.path.abspath(a.root or os.environ.get('MECHA_ROOT') or os.getcwd())
    os.environ['MECHA_ROOT'] = root

    # Windows：用 SelectorEventLoop 避开 Proactor 在客户端突然断连（dsh 自愈桥重连/
    # 浏览器关 SSE）时 _call_connection_lost 刷 WinError 10054 的无害噪声。本进程不在
    # 事件循环里 spawn 子进程，切 Selector 安全。
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    runtime.set_engine(EngineAdapter(atom=a.atom))   # 注入领域引擎
    # ★ 装配走 `build_session`（**可被测的具名装配点**）——`Session(authority=None)`
    #   会让写权门整段失效（A1 的深层缺陷，见该函数 docstring）。
    session = build_session(root, mode='human')
    print('数据根：%s（journal/artifacts/payloads/jobs）；控制权模式：human'
          % session.bus.root, flush=True)
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
    # MCP 是 **AI 的面**（actor='ai'，P2-4）；身份由**领域侧**给（P1-9e：
    # 库不再写死 're0-mecha' / "Re0 物理引擎…"——那是控诉书罪四）。
    _eng = runtime.engine()
    srv, _tools = create_server(
        session, ops, actor='ai',
        name=getattr(_eng, 'server_name', lambda: None)() or None,
        instructions=getattr(_eng, 'server_instructions', lambda: None)() or None)
    print('MCP 服务名：%s；工具数：%d'
          % (getattr(_eng, 'server_name', lambda: 'mecha')(), len(_tools)),
          flush=True)

    if a.stdio:
        print('MCP 写通道：stdio 传输', flush=True)
        asyncio.run(srv.run_stdio_async())
        return 0

    port = find_free_port(a.port)
    # 端口文件与数据根同址：<root>/.mcp-port（harness/GUI 靠它发现 Hub；T4）。
    pf = write_port_file(port, root=os.path.dirname(session.bus.root))
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
