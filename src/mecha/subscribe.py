# -*- coding: utf-8 -*-
"""只读订阅端点（读通道）——`mecha.bus` 的网络投影（纲领 T1/T2/T6）。

- **T1 两条通道一个真值源**：写走 MCP（localhost），读走本端点；两者都接在
  **同一条事件总线**上（本端点只 `bus.since/subscribe`，不镜像 MCP 流量）。
- **T2 读总线不读 MCP 镜像**：这里承载**所有 actor** 的事件（AI / 人 / 引擎），
  因为它读的就是 bus 本身。
- **T6 读写端点分家**：本端点**只读**、可（鉴权后）远程暴露；写端点（MCP）永不
  出本机。物理上本模块**没有任何写回 bus 的路径**（只调 since/subscribe/orientation）。

供**远程驾驶舱**在项目存活时低延迟 tail；本地独立驾驶舱另有 `src/cockpit`（读
journal 文件，项目死了也在，T5）。
"""
import asyncio
import json

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

__all__ = ['create_app']


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)


def create_app(bus, token=None):
    """把一个 `Bus` 投影成只读 Starlette app。

    token：非 None 时，所有请求须带 `?token=<token>` 或 `Authorization: Bearer`
    （远程只读鉴权，T6）。**无写路由**。
    """
    def _authorized(request):
        if not token:
            return True
        if request.query_params.get('token') == token:
            return True
        auth = request.headers.get('authorization', '')
        return auth == 'Bearer %s' % token

    async def history(request):
        if not _authorized(request):
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        since = int(request.query_params.get('since_seq', 0))
        evs = bus.since(since)
        return JSONResponse({'since_seq': since, 'count': len(evs),
                             'last_seq': bus.last_seq, 'events': evs})

    async def orientation(request):
        if not _authorized(request):
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        since = int(request.query_params.get('since_seq', 0))
        return JSONResponse(bus.orientation(since_seq=since))

    async def stream(request):
        """SSE：先补发 since_seq 之后的存量，再实时推新事件（diff-since-seq）。"""
        if not _authorized(request):
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        since = int(request.query_params.get('since_seq', 0))

        async def gen():
            q = asyncio.Queue()
            unsub = bus.subscribe(lambda ev: q.put_nowait(ev))
            try:
                for ev in bus.since(since):           # 存量补发
                    yield 'data: %s\n\n' % _dump(ev)
                yield 'event: ready\ndata: %s\n\n' % _dump(
                    {'last_seq': bus.last_seq})
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield 'data: %s\n\n' % _dump(ev)
                    except asyncio.TimeoutError:
                        yield ': heartbeat\n\n'        # 保活注释帧
            finally:
                unsub()
        return StreamingResponse(gen(), media_type='text/event-stream')

    return Starlette(routes=[
        Route('/history', history),
        Route('/orientation', orientation),
        Route('/events', stream),
    ])
