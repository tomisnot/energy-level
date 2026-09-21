# -*- coding: utf-8 -*-
"""MCP 写通道 launcher（领域侧）——注入 `EngineAdapter` 后起 `mecha.server`。

人运行：`python -m src.adapter.serve`。mecha 不认领域，故"构造引擎"的 launcher 住在
adapter（唯一可同时 import core 与 mecha 的层）。写通道仅 localhost（T6）；harness 靠
`.mcp-port` 端口文件发现并 attach（T4：人启动 = 权威，harness 绝不拉起）。
"""
import argparse
import asyncio
import atexit
import os
import sys

from ..mecha import registry, runtime
from ..mecha.ops import Ops
from ..mecha.server import (create_server, find_free_port, run_streamable_http,
                            write_port_file)
from ..mecha.session import Session
from . import EngineAdapter

__all__ = ['main']


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m src.adapter.serve')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--atom', default='Rubidium87')
    ap.add_argument('--stdio', action='store_true',
                    help='用 stdio 传输（默认 streamable-http + 端口文件）')
    a = ap.parse_args(argv)

    if os.environ.get('RE0_MECHA_NO_SERVER'):
        print('RE0_MECHA_NO_SERVER 置位 → 不启动 MCP 写通道')
        return 0

    runtime.set_engine(EngineAdapter(atom=a.atom))   # 注入领域引擎
    session = Session()
    # 落一条 registry_schema 事件：驾驶舱配置态结构树的 schema 源（需求②）。cockpit 只读
    # journal 即可富渲染、不 import mecha（黑匣子纯净 T5）；大 schema 被 I4 体积闸截成
    # 指针，cockpit 端顺 where 读回原文。
    session.bus.append(
        actor='system', kind='registry_schema', target='registry',
        after=registry.list_params(include_internal=True)['params'],
        reason='配置态 schema（供驾驶舱结构树）',
        provenance='adapter.serve', reversibility='none')
    ops = Ops(session)
    srv, _ = create_server(session, ops)

    if a.stdio:
        print('MCP 写通道：stdio 传输', flush=True)
        asyncio.run(srv.run_stdio_async())
        return 0

    port = find_free_port(a.port)
    pf = write_port_file(port)
    atexit.register(lambda: os.path.exists(pf) and os.remove(pf))
    print('MCP 写通道：http://%s:%d/mcp（端口文件 %s）' % (a.host, port, pf),
          flush=True)
    print('写通道仅 localhost（T6）；驾驶舱只读订阅端点另起。', flush=True)
    try:
        asyncio.run(run_streamable_http(srv, a.host, port))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
