# -*- coding: utf-8 -*-
"""驾驶舱（`src/cockpit`）——**独立只读**的 Web 投影（纲领 §2 投影层 / T5 黑匣子）。

它是一个**独立进程**（`python -m src.cockpit.server`），物理上独立于项目机身：
读的是磁盘上的 append-only journal（`backend.read_events`），**项目死了它还在**，
仍能显示死前最后状态 + 完整事件史（T5：飞行记录仪必须独立于机身）。

三条纪律：
- **只读**（I2）：无任何写回项目/总线的通道——由 `cockpit_selfcheck` 的 AST 守卫
  致死性看住。驾驶舱无自己的真值，只投影总线；本地只留视图偏好。
- **读总线不读 MCP 镜像**（T2）：数据源是 journal（写入也走的那条总线的持久化
  形态），承载所有 actor（AI / 人 / 引擎）。
- **读写端点分家**（T6）：默认只绑 localhost；`--host 0.0.0.0 --token <t>` 才远程
  只读暴露（鉴权）；写端点（MCP）永不出本机，本进程根本不含写路由。
"""
import argparse
import asyncio
import json
import os
import sys

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

from . import backend

__all__ = ['create_app', 'main']

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


def _default_root():
    """默认 journal 根 `<repo>/.mecha`——**按路径约定**自算，不 import mecha
    （驾驶舱完全独立于项目机身，只读磁盘文件；T5 黑匣子）。"""
    here = os.path.dirname(os.path.abspath(__file__))   # src/cockpit
    repo = os.path.dirname(os.path.dirname(here))       # → src → repo
    return os.path.join(repo, '.mecha')


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)


def create_app(root=None, token=None):
    """只读驾驶舱 app（static 前端 + /api/* 读接口 + SSE）。**无写路由**。"""
    root = root or _default_root()

    def _authorized(request):
        if not token:
            return True
        if request.query_params.get('token') == token:
            return True
        return request.headers.get('authorization', '') == 'Bearer %s' % token

    def _guard(request):
        return None if _authorized(request) else JSONResponse(
            {'error': 'unauthorized'}, status_code=401)

    async def api_history(request):
        d = _guard(request)
        if d:
            return d
        since = int(request.query_params.get('since_seq', 0))
        limit = request.query_params.get('limit')
        evs = backend.read_events(root, since,
                                  limit=int(limit) if limit else None)
        return JSONResponse({'since_seq': since, 'count': len(evs),
                             'last_seq': backend.last_seq(root), 'events': evs})

    async def api_orientation(request):
        d = _guard(request)
        if d:
            return d
        return JSONResponse(backend.orientation(root))

    async def api_alerts(request):
        d = _guard(request)
        if d:
            return d
        evs = backend.read_events(root, 0)
        al = backend.evaluate_alerts(evs)
        return JSONResponse({'level': backend.top_level(al), 'alerts': al,
                             'as_of_seq': backend.last_seq(root)})

    async def api_config_tree(request):
        """配置态结构树（需求②）：schema join 现值，纯读 journal。"""
        d = _guard(request)
        if d:
            return d
        return JSONResponse(backend.config_tree(root))

    async def api_claim_measured(request):
        """M2 声称 vs 实测并排：每条 set/run 事件的 reason（AI 声称）| after
        （choke point 产出的 readback 实测）。AI 伪造不了右列（readback 不经 AI）。"""
        d = _guard(request)
        if d:
            return d
        since = int(request.query_params.get('since_seq', 0))
        evs = backend.read_events(root, since)
        pairs = [{'seq': e.get('seq'), 'actor': e.get('actor'),
                  'kind': e.get('kind'), 'target': e.get('target'),
                  'claim': e.get('reason'), 'measured': e.get('after')}
                 for e in evs if e.get('kind') in ('set', 'set_many', 'run')]
        return JSONResponse({'pairs': pairs[-100:],
                             'as_of_seq': backend.last_seq(root)})

    async def api_stream(request):
        """SSE：tail journal 文件，diff-since-seq 增量推（项目死了也能读到史）。"""
        d = _guard(request)
        if d:
            return d
        since = int(request.query_params.get('since_seq', 0))

        async def gen():
            cur = since
            idle = 0
            while True:
                if await request.is_disconnected():
                    break
                evs = backend.read_events(root, cur)
                for ev in evs:
                    yield 'data: %s\n\n' % _dump(ev)
                    cur = int(ev.get('seq', cur))
                if evs:
                    idle = 0
                else:
                    idle += 1
                    if idle % 20 == 1:
                        yield ': heartbeat\n\n'
                    await asyncio.sleep(0.5)
        return StreamingResponse(gen(), media_type='text/event-stream')

    async def healthz(request):
        return JSONResponse({'ok': True, 'root': root,
                             'last_seq': backend.last_seq(root),
                             'read_only': True})

    routes = [
        Route('/api/history', api_history, methods=['GET']),
        Route('/api/orientation', api_orientation, methods=['GET']),
        Route('/api/alerts', api_alerts, methods=['GET']),
        Route('/api/config-tree', api_config_tree, methods=['GET']),
        Route('/api/claim-measured', api_claim_measured, methods=['GET']),
        Route('/api/stream', api_stream, methods=['GET']),
        Route('/healthz', healthz, methods=['GET']),
    ]
    if os.path.isdir(_STATIC):
        routes.append(Mount('/', StaticFiles(directory=_STATIC, html=True),
                            name='static'))
    return Starlette(routes=routes)


def main(argv=None):
    """人运行的 launcher：起独立只读驾驶舱（默认 localhost）。"""
    ap = argparse.ArgumentParser(prog='python -m src.cockpit.server')
    ap.add_argument('--root', default=None, help='journal 根目录（默认 <repo>/.mecha）')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--token', default=None,
                    help='远程只读鉴权 token（绑非 localhost 时务必设）')
    a = ap.parse_args(argv)

    root = a.root or _default_root()
    if a.host not in ('127.0.0.1', 'localhost') and not a.token:
        print('⚠ 绑定非 localhost（%s）却没设 --token：远程只读**必须**鉴权（T6）。'
              % a.host, file=sys.stderr)
    app = create_app(root=root, token=a.token)
    import uvicorn
    print('驾驶舱（只读）：http://%s:%d/  root=%s' % (a.host, a.port, root),
          flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level='warning')
    return 0


if __name__ == '__main__':
    sys.exit(main())
