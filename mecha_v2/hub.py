"""mecha v2 MCP 端点（EL 侧**薄壳**）：投影与传输全在 ``mecha.providers.mcp``。

归属：本文件住在 EL 仓，但只保留 **EL 词汇与 EL 策略**——服务名、服务说明。
其余（三步投影、required 解析、null 垫片、失败归一化、uvicorn 后台线程与端口回读、
SDK 私有面垫片、instructions 体积守卫、端口文件的启停清理）**逐条**已是框架行为
（``mecha.providers.mcp``，mecha `4638519`），本文件不再有第二份实现。

## 迁移记录（第 3 段：MCP 侧薄壳化，2026-09-25）

迁移前本文件 **223 行**，自持 ``to_mcp`` / ``_register`` / ``_free_port`` 与
"工具面覆写"那整套机制（闭包签名推 required、`defaulted` 垫片、私有面写回）。
现按契约上提框架，EL 侧只剩默认值：

- **调用路径**：``tools.execute`` → 框架 handler 内的 ``toolhost.call``（远端边界那层
  失败归一化 ``toolhost_error`` / ``bad_host_result`` 留在路径上）⇒ 畸形回执从
  "透传成怪响应"变成**当场归一化的失败**（更严，不是行为回归）。
- **required**：框架按 ``注入来源 → 闭包签名`` 求；EL 的工具都是**命名 lambda**
  （签名可用）⇒ **不需要**注入 ``RequiredSource``。
- **null 垫片**：``defaulted`` 的唯一权威来源是**声明里的 ``default`` 键**（不读闭包
  签名）——本段把 EL 的 10 个"真默认非 None"的参数补齐了声明（见 ``tools.py`` 的
  R5 三分法），真默认就是 ``None`` 的 8 个一律不声明。
- **wire 三条**：``required`` **恒回 ``[]``**、无 ``type`` 声明 ⇒ 注解 ``Any``
  （``set_config.value`` 的任意 JSON 形态靠它）、**不声明 ``outputSchema``**。

**两种部署形态，别混**（审查 2.3 纠偏）：
- **独立进程 Hub（本文件的 ``main``）**：无 GUI 的 headless 部署（跑在 A 机、
  AI 从 B 机连）。写权威落 ``--data-dir``（默认 ``.mecha-v2-hub``）。
- **同一道门（``bridge.V2Bridge.serve_mcp``，EL 默认部署路径）**：GUI 进程内
  MCP 服务器线程与 GUI **共一 Gate**，一份真值一份史（v1 已退役，GUI 即默认）。
  两条路径都经 ``build_mcp_server`` 构造同一个投影。

用法：python -m mecha_v2.hub --root <EL根> [--port 0]
      [--data-dir .mecha-v2-hub] [--open-gate] [--port-file 路径]

## Known Limitations and Deferred Work

- 独立进程与 GUI **不共享内存快照**：两进程同起会各写各的 journal（写租约
  保证各目录单写者，但互不见对方的写）——headless 部署就**只起这一个**权威。
- 未接 dsh ctx.approval：AI 写权由 Authority 模式机决定（出厂 LOCKED，
  --open-gate 或人工端点开闸）；细粒度每次写问人待 approval Provider 接线。
- run_sim 只回摘要（pops/t 不过线）；完整数组走 Spill Provider（待接线）。
- 服务名/说明是**宿主词汇**（dsh 侧要求名形如 ``[A-Za-z0-9_-]{1,32}`` 且跨实例唯一，
  说明的字节上限由框架 ``check_instructions`` 在构造期守）——框架按设计不校验服务名。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

# —— EL 根注入：本文件可被直跑（python -m mecha_v2.hub 或判据 subprocess），
# 插 EL 根保证 `mecha_v2` 包与 `src.core.protocol` 都解析得到 ——
_EL_ROOT = Path(__file__).resolve().parents[1]
if str(_EL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EL_ROOT))

from mecha.authority import Mode  # noqa: E402
from mecha.data_layout import resolve_data_dir  # noqa: E402
from mecha.providers.mcp import McpEndpoint as _McpEndpoint  # noqa: E402
from mecha.providers.mcp import (  # noqa: E402
    build_mcp_server as _framework_build_mcp_server,
)

from mecha_v2.engine import ElEngine, make_validator  # noqa: E402
from mecha_v2.tools import build_tool_registry  # noqa: E402

#: MCP 服务名 → dsh 侧工具名形如 ``mcp__mecha-el__set_config``。
MCP_SERVER_NAME = "mecha-el"

#: 服务端说明（模型可见）。只含任务相关概念，不含 UI/传输/实现词汇；体积由框架的
#: ``check_instructions`` 在构造期守卫（dsh 侧 maxInstructionBytes 默认 32768，
#: 超出会让连接失败——报错离病因很远，所以要在构造期拦）。
INSTRUCTIONS = (
    "Rydberg 仿真软件的操作工具集。"
    "先 query_params 看可写键，set_config 写，estimate_run 估时，run_sim 跑。"
)


def build_stack(el_root: str | Path, data_root: str | Path,
                data_dir_name: str = ".mecha-v2") -> dict:
    """装配 Hub 侧的 v2 栈——**经 mecha.assembly.assemble() 具名装配点**
    （D-33：真实路径与判据同走 assemble，装配漂移在真引擎上即红）。

    返回里**只带真注册表**（``tools``），**不带** ``toolhost``：``assemble()`` 的
    ``sw.toolhost`` 包的是装配点那个**空** ``sw.tools``，而 EL 的 14 个工具注册在
    ``build_tool_registry`` 里、从不写进 ``sw.tools`` ⇒ 谁把它当 ``toolhost=`` 传进
    MCP 端点，就会"工具列得出来、调不动"（2026-09-25 实发，见 A11）。调用唯一路径
    仍是框架 handler 内的 ``ToolHost.call``；端点缺省
    ``LocalToolHost(本函数返回的 tools)`` 才是对的。
    """
    from mecha.assembly import assemble
    from mecha_v2.bridge import el_summarizer
    engine = ElEngine(el_root)
    sw = assemble(root=Path(data_root), data_dir_name=data_dir_name,
                  engine=engine, summarizer=el_summarizer,
                  validate=make_validator(engine), mode=Mode.LOCKED,
                  project="energy-level")
    ai = sw.channels["ai"]
    tools = build_tool_registry(engine, sw.gate, sw.surface, sw.authority, ai,
                                sw.jobs)
    return {"engine": engine, "history": sw.history, "authority": sw.authority,
            "gate": sw.gate, "surface": sw.surface, "ai": ai, "jobs": sw.jobs,
            "tools": tools}


def build_mcp_server(tools, *, toolhost=None, **kwargs):
    """按 EL 的默认值**纯构造** MCP 服务（不起线程、不碰端口；判据可直接调）。

    EL 的默认值只有两样：服务名与服务说明；其余（投影、required、null 垫片、
    失败归一化）走框架。``toolhost`` 缺省时框架用 ``LocalToolHost(tools)``。
    """
    return _framework_build_mcp_server(
        tools, server_name=MCP_SERVER_NAME, instructions=INSTRUCTIONS,
        toolhost=toolhost, **kwargs)


class McpHost:
    """在**权威进程后台线程**跑 MCP 服务（薄壳：委托框架 ``McpEndpoint``）。

    - ``start()`` 返回**实际**端口（``port=0`` 由 OS 分配后**回读**，无 TOCTOU
      窗口），并写 ``port_file`` 供发现；
    - ``stop()`` 干净退出：置 uvicorn 的 ``should_exit`` → join → **按内容**清端口
      文件（模式交接依赖"旧权威真退出、写租约释放、新权威再起"）；
    - ``server`` 暴露底层 MCPServer（判据/调试用；**不是**第二条调用通道）。
    """

    def __init__(self, tools, *, toolhost=None, host: str = "127.0.0.1",
                 port: int = 0, port_file=None, log=None) -> None:
        self._endpoint = _McpEndpoint(
            tools, server_name=MCP_SERVER_NAME, instructions=INSTRUCTIONS,
            toolhost=toolhost, host=host, port=port, port_file=port_file, log=log)

    # ------------------------------------------------------------ 事实面
    @property
    def port(self) -> int | None:
        return self._endpoint.port

    @property
    def url(self) -> str:
        return self._endpoint.url

    @property
    def server(self):
        return self._endpoint.server

    # ------------------------------------------------------------ 生命周期
    def build_server(self):
        return self._endpoint.build_server()

    def start(self, timeout: float = 30.0) -> int:
        return self._endpoint.start(timeout)

    def stop(self, timeout: float = 10.0) -> None:
        self._endpoint.stop(timeout)

    def __enter__(self) -> "McpHost":
        self.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False


def make_host(stack: dict, *, host: str = "127.0.0.1", port: int = 0,
              port_file=None, log=None) -> "McpHost":
    """由 ``build_stack`` 的产物造一台 MCP 端点壳（**真入口与判据同源**）。

    ⚠ 这里**不再**传 ``toolhost=stack["toolhost"]``：``assemble()`` 建出的
    ``sw.tools`` 是**空注册表**（EL 的工具注册在自己的 ``build_tool_registry`` 里，
    从不写进 ``sw.tools``）⇒ ``sw.toolhost`` 包的是空表，传它会让
    **工具列得出来、调不动**（`unknown_tool`，且 journal 不落盘——2026-09-25 实测，
    由 mecha 的 `phase5_hub_selfcheck` 抓到）。缺省让框架用
    ``LocalToolHost(stack["tools"])``，即绑**真的**那 14 个工具。
    """
    return McpHost(stack["tools"], host=host, port=port,
                   port_file=port_file, log=log)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="mecha v2 MCP Hub（EL）")
    ap.add_argument("--root", default=os.environ.get("MECHA_EL_ROOT", ""),
                    help="EL 项目根（含 src/core/protocol.py）")
    ap.add_argument("--data-root", default="", help="数据目录父（默认=EL 根）")
    ap.add_argument("--data-dir", default=".mecha-v2-hub",
                    help="数据目录名（注入；headless 独立进程写权威落此处。"
                         "与 GUI 的 .mecha-v2 分开——两进程各写各的 journal，"
                         "写租约各保各目录单写者，但互不见对方的写）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0=自动选空闲端口")
    ap.add_argument("--port-file", default="",
                    help="端口文件（默认 <root>/.mcp-port，与 dsh 插件 re0-mecha "
                         "的发现约定一致；勿与 GUI 进程同起——两者都写此文件会互踩）")
    ap.add_argument("--open-gate", action="store_true",
                    help="启动即开 ai 写权（演示/测试；默认 LOCKED）")
    args = ap.parse_args(argv)

    if not args.root:
        print("必须给 --root 或环境变量 MECHA_EL_ROOT（EL 项目根）")
        return 2
    data_root = args.data_root or args.root
    stack = build_stack(args.root, data_root, args.data_dir)
    if args.open_gate:
        stack["authority"].switch_mode(Mode.AI, side="human")
        print("[hub] 写权已开到 ai 模式（--open-gate）")

    port_file = args.port_file or str(Path(args.root) / ".mcp-port")
    host = make_host(stack, host=args.host, port=args.port, port_file=port_file,
                     log=print)
    port = host.start()
    print(f"[hub] {port_file} ← {port}；"
          f"工具 {len(stack['tools'].schemas())} 个；"
          f"journal={resolve_data_dir(Path(data_root), args.data_dir).data_dir}")
    try:
        threading.Event().wait()                     # 阻塞到 Ctrl+C / terminate
    except KeyboardInterrupt:
        print("\n[hub] 收到 Ctrl+C，收尾（按内容清端口文件）。")
    finally:
        host.stop()
    return 0


#: 公开面**只有 EL 自己的东西**：装配、默认值壳、端点壳。框架的
#: ``tool_input_schema`` / ``resolve_required`` / ``RequiredSource`` / ``MCP_PATH``
#: 等**不在这里转手导出**——它们没有本仓的消费者（判据要直接
#: ``from mecha.providers.mcp import …``），转一手只多一层要同步的转发面。
__all__ = ["INSTRUCTIONS", "MCP_SERVER_NAME", "McpHost", "build_mcp_server",
           "build_stack"]


if __name__ == "__main__":
    sys.exit(main())
