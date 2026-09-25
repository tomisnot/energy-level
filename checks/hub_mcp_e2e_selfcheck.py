# -*- coding: utf-8 -*-
"""Hub/Bridge 的 MCP **端到端**判据（R10：迁移后先问"我改的那条路径，谁在走它？"）。

## 为什么必须有这一条

2026-09-25 第 3 段把 MCP 投影上提框架时，EL 侧把 ``toolhost=stack["toolhost"]``
（= ``sw.toolhost``，``assemble()`` 建出的**空注册表**）传给了端点 ⇒
**工具列得出来（14 个，投影用的是真注册表）、但一个都调不动**（``unknown_tool``，
连带 journal 不落盘）。当时 ``--fast`` **30/30 全绿**——因为**没有任何自家判据真的
走"装配 → 起 MCP → 调用"这条路径**；抓到它的是 mecha 的跨仓判据 ``phase5_hub_selfcheck``。
⇒ 本判据补上这条路径：**真装配**（`build_stack` / `V2Bridge.for_shared`，真 ElEngine）
→ **起真 MCP**（`port=0`，streamable-http）→ **真调一个工具** → **查 journal 真落盘**。

## 两条腿（同一缺陷类的两个生产路径）

- **A 独立进程 Hub**：`hub.build_stack()` → `hub.make_host()` → 真 HTTP 调 `set_config`。
- **B 进程内同一道门**（EL 默认部署路径）：`V2Bridge.for_shared()` → `serve_mcp()`
  → 同一个客户端往返。

两条都断言：① 工具面非空且含 `set_config`（"列得出来"）；② 调用**成功**
（`isError` 假、正文无 `unknown_tool`、正文是 `{ok: true …}`）（"调得动"）；
③ **journal.jsonl 真出现那条写**（"真落盘"）。

## 非退化自证（R8）

打印并断言：列出的工具数 > 0、调用回执正文非空、journal 行数 > 0。否则"两边都空"
也会"看起来一致"。灵敏度证据（R7）：把接线改回修复前（`make_host` 传空注册表的
toolhost）⇒ 本判据必须红（`unknown_tool` + journal 无事件）。

隔离：数据根用临时目录（不碰仓内 ``.mecha-v2``）；端口 ``0``（OS 分配）+ 端口文件
写临时根；两条腿各自独立临时根（写租约不互抢）。

用法：python checks\\hub_mcp_e2e_selfcheck.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from mecha.authority import Mode                             # noqa: E402
from mecha.data_layout import resolve_data_dir               # noqa: E402

from mecha_v2.bridge import V2Bridge                         # noqa: E402
from mecha_v2.hub import build_stack, make_host              # noqa: E402

C = Checker('hub_mcp_e2e')
EL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRITE_KEY = 'field.B_gauss'
WRITE_VALUE = 21.7


async def _wire_roundtrip(url):
    """真 MCP 客户端往返（streamable-http）→ (工具名表, 调用正文, is_error)。

    ⚠ **字段名是 ``is_error``（snake_case）**——mcp 2.2 的 ``CallToolResult`` 只有这个名字
    （``isError`` 只是序列化别名）。本判据第一版读 ``isError`` ⇒ ``getattr`` 兜默认 False
    ⇒ 那一项**恒为假、是死的观测量**（全靠正文断言在抓）。这是 R9 的同族坑
    （观测量本身失效），故这里两种名字都读、并以正文为主证据。
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            res = await session.call_tool(
                "set_config", {"key": WRITE_KEY, "value": WRITE_VALUE,
                               "reason": "判据 E2E"})
            text = "".join(getattr(c, "text", "") or "" for c in (res.content or []))
            err = getattr(res, "is_error", None)
            if err is None:
                err = getattr(res, "isError", False)
            return [t.name for t in listed], text, bool(err)


def _journal(root, dir_name):
    """数据目录下的 journal 路径（真落盘证据）。"""
    return Path(resolve_data_dir(Path(root), dir_name).data_dir) / "journal.jsonl"


def _journal_lines(path):
    try:
        return [json.loads(l) for l in
                Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return []


def _leg(label, tools, text, is_error, journal_path, subs, stats):
    """一条腿的三条断言（A1/A2/A3）+ 非退化自证。"""
    stats['%s.工具数' % label] = len(tools)
    stats['%s.回执' % label] = (text or '')[:120]
    events = _journal_lines(journal_path)
    stats['%s.journal 行数' % label] = len(events)
    subs.append(('① %s 工具面：列出 %d 个且含 set_config（列得出来）'
                 % (label, len(tools)),
                 len(tools) > 0 and 'set_config' in tools))
    ok_call = (not is_error) and ('unknown_tool' not in (text or '')) \
        and ('"ok": true' in (text or '').replace('"ok":true', '"ok": true')
             or '"ok":true' in (text or ''))
    subs.append(('② %s 真调 set_config 成功（调得动；isError=%s，正文 %s）'
                 % (label, is_error, (text or '')[:70]), ok_call))
    hit = any(e.get('key') == WRITE_KEY for e in events)
    subs.append(('③ %s journal 真落了那条写（%d 行，命中=%s）'
                 % (label, len(events), hit), hit))


def main():
    subs, stats = [], {}

    # ---------------- A 独立进程 Hub：真装配 → 起 MCP → 真调 ----------------
    tmp_a = tempfile.mkdtemp(prefix='re0-e2e-hub-')
    stack = build_stack(EL_ROOT, tmp_a)
    stack['authority'].switch_mode(Mode.AI, side='human')      # = main() 的 --open-gate
    pf_a = str(Path(tmp_a) / '.mcp-port')
    host = make_host(stack, port=0, port_file=pf_a)
    try:
        host.start()
        stats['A.端口'] = host.port
        stats['A.端口文件'] = Path(pf_a).read_text(encoding='utf-8').strip() \
            if Path(pf_a).exists() else '（缺）'
        tools, text, is_error = asyncio.run(_wire_roundtrip(host.url))
    finally:
        host.stop()
    stats['A.stop 后端口文件'] = '已清' if not Path(pf_a).exists() else '仍在'
    _leg('A Hub', tools, text, is_error, _journal(tmp_a, '.mecha-v2'), subs, stats)

    # ---------------- B 进程内同一道门：for_shared → serve_mcp → 真调 ----------------
    tmp_b = tempfile.mkdtemp(prefix='re0-e2e-bridge-')
    br = V2Bridge.for_shared(el_root=EL_ROOT, data_root=tmp_b, mode=Mode.AI)
    try:
        bridge_host, port_b = br.serve_mcp(port=0)
        stats['B.端口'] = port_b
        tools_b, text_b, err_b = asyncio.run(_wire_roundtrip(bridge_host.url))
    finally:
        try:
            bridge_host.stop()
        except NameError:
            pass
    _leg('B 进程内', tools_b, text_b, err_b, _journal(tmp_b, '.mecha-v2'), subs, stats)

    print('[hub_mcp_e2e] 非退化自证：%s' % json.dumps(stats, ensure_ascii=False))
    C.group('A 真装配→起 MCP→真调 set_config→journal 落盘（独立进程 Hub 路径）', subs[:3])
    C.group('B 同一条路走进程内默认部署路径（for_shared → serve_mcp）', subs[3:])
    C.group('R8 非退化自证：工具面/回执/journal 都非空（防"两边都空"）', [
        ('A 工具数 %d > 0、回执非空、journal %d 行'
         % (stats.get('A Hub.工具数', 0), stats.get('A Hub.journal 行数', 0)),
         stats.get('A Hub.工具数', 0) > 0 and bool(stats.get('A Hub.回执'))
         and stats.get('A Hub.journal 行数', 0) > 0),
        ('B 工具数 %d > 0、回执非空、journal %d 行'
         % (stats.get('B 进程内.工具数', 0), stats.get('B 进程内.journal 行数', 0)),
         stats.get('B 进程内.工具数', 0) > 0 and bool(stats.get('B 进程内.回执'))
         and stats.get('B 进程内.journal 行数', 0) > 0),
    ])
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
