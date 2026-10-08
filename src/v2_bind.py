# -*- coding: utf-8 -*-
"""EL ↔ mecha v2 的接线层：两模式收拢后的装配点（v1 门面已退役）。

app_entry 用它装配 V2Bridge.for_shared。**两模式**（用户裁决 2026-09-24）：
- GUI 模式：`start_services=False`——纯本地软件，不起 MCP/监控端点，AI 物理
  看不见（写权恒 human）。
- AI 模式：headless 权威（run_authority）用默认 True——起 MCP 服务器线程 +
  只读监控端点，写权自动落 AI（看门人代理人意图开闸）。

无论哪种模式，所有写都经**同一道 Gate、同一份 History**。cache 经 CacheProxy
汇流到引擎线程（ARC SQLite 认线程，B-1①）。

监控端点是**只读**的（/status + /activity + /history + /config）：给 dsh 右侧栏
面板轮询"谁改了什么"。**实现已上提到框架 `mecha.cockpit`**（2026-09-25 第 1 段，
mecha `2722f58`）——本文件只留 EL 侧的两件事：把 `V2Bridge` 适配成框架的
`MonitorSource`、以及端口由宿主传参。控制权切换不经它——人的夺权/交权由看门人
（launcher.py）编排进程交接；"AI 不能自授权"的护栏靠 Authority 的 side 判定
（AI 通道改不了 mode），不靠端点鉴权。

## Known Limitations and Deferred Work

- 依赖 mecha v2 已安装（editable 安装 mecha==0.2）。v1/v2 同名 mecha，
  **同进程只能有一个**——本层假设 v2 已生效，不兼容同进程 import v1 的
  mecha.ops/session 等。
- gc（数据回收）在 v2 未实现（D-9 保留策略等第二项目）；GUI 的回收按钮在
  v2 下会显示"未接线"而非静默，见 load_artifact 旁的 gc 垫片。
- undo/redo 未接：v2 History 是 append-only 事件流，撤销按"反向 set 记一条
  新事件"实现（决策见 mecha 仓 ADR `EL能力面取舍` 第 3 条；语义与 v1 的
  session.undo 不同），实现挂起至接 GUI 撤销按钮（真实调用方出现）。
- **监控端点：动态端口 + 端口文件发现**（第 2 段）：`port=0`（OS 分配，库内零端口
  字面量），实际端口写进 `<EL 根>/.monitor-port`；dsh 插件 host 半经同源只读路由
  `/re0-mecha/monitor-url` 把它转成 `{base}` 给面板，面板**每轮现读**。清理规则归
  框架 `mecha.portfile`（启动前无条件清 / 收尾按内容清）。**要改端口请改 dsh 侧**，
  Python 侧不再有任何端口字面量。
- 监控端点的**传输契约**（四路由字段 / CORS / OPTIONS / POST 404 / 精确匹配）归框架
  `mecha.cockpit`，由框架 `checks/phase11_cockpit_selfcheck.py` 守；EL 侧只守
  "**自己的 schema/装配真的接进了树**"（`checks/human_control_selfcheck.py` 的 F 组）。
"""
from __future__ import annotations

import os
import pickle
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


# ---------------- 监控端点（只读；路由/投影/生命周期归 mecha.cockpit）----------------
#
# 2026-09-25 第 1 段（迁移）：本文件原来的 `_MonitorHandler` + `_history_records` +
# `_config_tree` 整体换成**框架实现**（`mecha.cockpit`，mecha 2722f58）——两家下游
# 各写一份逐字节同形的端点，现在归框架。EL 侧只剩：
#   ① `_BridgeMonitorSource`：把 `V2Bridge` 适配成框架的五成员 MonitorSource；
#   ② 端口与发现文件（第 2 段：**动态端口** + `mecha.portfile` 写 `.monitor-port`）。
# 顺带消失的还有旧实现的**类属性共享隐患**（`_MonitorHandler.bridge = bridge`：
# 同进程起两个端点会互相覆盖）——框架按实例派生处理器子类。
#
# 为什么用显式适配器、而不是 `MonitorSource.from_software`：`V2Bridge` 只公开
# `history`，没有公开 `gate`/`authority`（`_assembled` 里用过 assemble 的 Software
# 但没留引用），而判据还会手搓没有 software 的纯内存桥。显式适配器让"真装配"与
# "判据桥"走同一条路，不引入第二种形态；框架侧也明说鸭子类型即可。
#
# 契约（逐字段冻结）：/status、/activity(6 键)、/history(9 键，`before` 由**全史
# fold** 重建)、/config(族→前缀树 + orphans + n_keys + n_snapshot_keys)、未知路径
# 404、一切 POST 404、OPTIONS 204 + CORS、CORS 覆盖所有响应（含 404）、服务端不
# 截断/不倒序。只读 + 绑回环 + 无鉴权是本层的全部安全模型。
#
# **地址怎么被消费方发现**（第 2 段）：权威把**动态分配**的监控端口写进
# `<EL 根>/.monitor-port`（裸端口，与 `.mcp-port` 同址同根）；dsh 插件的 host 半经
# **同源只读路由** `/re0-mecha/monitor-url` 把它转成 `{base}` 交给面板，面板
# **每轮轮询现读**（端口漂移自愈，不在模块加载时固化）。库内零端口字面量。

#: 监控端口的发现文件（裸端口文本；与 `.mcp-port` 同在项目根）。
MONITOR_PORT_FILE = ".monitor-port"

def _el_schema_rows(bridge):
    """EL 的配置 schema 行（`/config` 建树的输入）：键全名 + family/unit/type/desc。

    走 `V2Bridge.params(include_internal=True)`（= 引擎 `schema()` 的三族合成；
    engine 族键是**裸名**如 `R_CRIT` ⇒ 树里落 prefix=""）。**无引擎的纯内存桥**
    （判据手搓的那种）拿不到 schema ⇒ 如实空表，端点走"无 schema"分支
    （has_schema=false + 快照全进 orphans）——那是可观测契约的一条分支，不是降级。
    """
    try:
        return bridge.params(include_internal=True, page_size=100000)["params"]
    except Exception:                                      # noqa: BLE001
        return []


class _BridgeMonitorSource:
    """`V2Bridge` → `mecha.cockpit.MonitorSource`（五项全是薄转调）。"""

    def __init__(self, bridge):
        self._b = bridge

    def events(self):
        """全史事件（升序）。框架按 seq/key/value/actor/reason/call_id 取字段。"""
        return self._b.history.events()

    def snapshot(self):
        """当前权威快照（`V2Bridge.effective()` = gate 快照的只读投影）。"""
        return self._b.effective()

    def mode(self):
        """写权模式（EL 给字符串；框架投影时取 `.value` 兜底）。"""
        return self._b.mode

    def schema_rows(self):
        return _el_schema_rows(self._b)

    def status_extra(self):
        """`/status` 的附加键：EL 不给（保持与迁移前逐字节一致）。"""
        return {}


def start_monitor_endpoint(bridge, host="127.0.0.1", port=0, port_file=None):
    """起只读监控端点（薄壳：实现 = `mecha.cockpit`）。返回 `MonitorEndpoint`。

    绑回环、无鉴权——读的是"谁改了什么"的操作流水，非域数据覆写面；写与切模式
    不经这里（控制权归看门人）。dsh 右侧栏面板按 seq 增量轮询 /history 与 /config。

    **端口默认 0（OS 分配）**：库内零端口字面量；发现靠 `port_file`（传了才写）。
    清理规则由框架的 `mecha.portfile` 负责——**启动前无条件清**陈旧文件、
    **收尾只在"内容仍是自己写的端口"时才清**（Windows 强杀不跑 `finally`，两条都要）。

    调用方取返回值的 `.port` / `.url`，收尾调 `.stop()`。
    """
    from mecha.cockpit import MonitorEndpoint
    endpoint = MonitorEndpoint(_BridgeMonitorSource(bridge), host=host, port=port,
                               port_file=port_file)
    endpoint.start()
    return endpoint


# ---------------- 装配：给 app_entry 用 ----------------

def resolve_data_root(explicit, el_root):
    """数据根的**唯一解析点**（三来源 + 固定顺序 + 分歧出声）。

    顺序（⭐ 定案）：

    1. **显式参数**（调用方直接给）；
    2. **``MECHA_SOFTWARE_ROOT``**：**框架/启动者声明的**数据根（三层架构 §5.1）
       —— 它是"这一份软件的数据根"的**单一事实** ⇒ **优先**；
    3. ``RE0_V2_DATA_ROOT``：EL **内部**的管线变量（判据/测试/手工起用它指临时目录）
       ⇒ **回落**位置；
    4. ``el_root``。

    ⚠ **为什么框架声明优先于 EL 内部变量**：反过来的话，一个**被继承或被判据设的**
    环境变量就能把框架声明的根**悄悄顶掉** —— 那正是"看到两份状态、且无任何报错"
    那个坑（§5.6）。判据要覆盖框架那一侧：**设 `MECHA_SOFTWARE_ROOT` 即可**，
    不必也不该靠"临时目录盖过框架注入"。
    ⚠ 两个来源都设且 resolve 后**不一致** ⇒ **响亮说出来**（仍按上面的顺序执行）：
    静默取一个，会让"数据到底写哪去了"变成必须翻代码才知道的事。
    """
    picked = explicit or None
    if not picked:
        picked = os.environ.get("MECHA_SOFTWARE_ROOT") or None
    if not picked:
        picked = os.environ.get("RE0_V2_DATA_ROOT") or None
    if not picked:
        picked = el_root
    _m = os.environ.get("MECHA_SOFTWARE_ROOT")
    _r = os.environ.get("RE0_V2_DATA_ROOT")
    if not explicit and _m and _r and os.path.abspath(_m) != os.path.abspath(_r):
        print("[v2_bind] ⚠ 数据根两个来源不一致：MECHA_SOFTWARE_ROOT=%s / "
              "RE0_V2_DATA_ROOT=%s ⇒ 采信**框架声明**的（%s）；EL 内部变量这次被忽略。"
              % (_m, _r, _m), flush=True)
    return picked


def make_v2_bridge(el_root, *, data_root=None, data_dir_name=".mecha-v2",
                   start_services=True, mcp_port_file=None, monitor_port_file=None):
    """构造共享 Gate 的 V2Bridge；可选起 MCP 服务器 + 只读监控端点。

    app_entry 调它：本地 GUI 传 `start_services=False`（纯本地，不起任何端点）；
    MCP 服务形态（`run_mcp`）用默认 True（起 MCP + 监控端点，供 dsh 连与轮询）。
    数据根优先级：见 `resolve_data_root`（显式 > `MECHA_SOFTWARE_ROOT` 框架声明 >
    `RE0_V2_DATA_ROOT` EL 内部回落 > el_root；两者不一致时**出声**）。
    `EL_NO_AI_SERVER` 置位 → 不起后台
    服务（判据 in-process 用，避免与活实例抢端口/抢 .mcp-port——2026-09-12
    事故的同款防线）。

    两个发现文件（都默认落在 EL 根，与 dsh 的 cwd 同源）：`.mcp-port`（MCP 端点）、
    `.monitor-port`（只读监控端点）。要写别处（判据/一次性验证）就显式传路径。
    """
    from mecha_v2.bridge import V2Bridge
    from mecha.authority import Mode
    data_root = resolve_data_root(data_root, el_root)
    if os.environ.get("EL_NO_AI_SERVER"):
        start_services = False
    bridge = V2Bridge.for_shared(el_root=el_root, data_root=data_root,
                                 data_dir_name=data_dir_name, mode=Mode.HUMAN)
    bridge._mcp_host = None
    bridge._mcp_server = None
    if start_services:
        pf = mcp_port_file or str(Path(el_root) / ".mcp-port")
        # MCP 端点：框架实现（mecha.providers.mcp，第 3 段薄壳化）；实际端口回读、
        # 起完即已 listen、端口文件由框架按"启动前清 / 收尾按内容清"管理。
        mcp_host, port = bridge.serve_mcp(port_file=pf)
        bridge._mcp_host = mcp_host
        bridge._mcp_server = mcp_host.server
        bridge._mcp_port = port
        # 监控端点：框架实现（mecha.cockpit）；**动态端口**（port=0）+ 端口文件
        # 发现。端点对象留给调用方取 .port / 收尾 .stop()。
        mpf = monitor_port_file or str(Path(el_root) / MONITOR_PORT_FILE)
        endpoint = start_monitor_endpoint(bridge, port_file=mpf)
        bridge._monitor_endpoint = endpoint
        bridge._monitor_port = endpoint.port
        print(f"[v2_bind] MCP 服务器 :{port}/mcp（工具经同一道 Gate）；"
              f"只读监控 :{endpoint.port}（发现文件 {Path(mpf).name}；dsh 面板轮询）")
    return bridge
