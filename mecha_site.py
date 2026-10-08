# -*- coding: utf-8 -*-
"""EL 的**站点声明**：把"这个软件怎么被起来"交给 mecha 的统一启动器。

## 这个文件是什么

框架的 supervisor 按约定发现本文件（模块级 ``SITE``），把声明翻成进程编排
（设计文档：`mecha/docs/v2/统一启动器-设计.md`）。⇒ **本项目不再自己编排进程**
（旧 `launcher.py` 那套是它的前身，步骤 4 整份删除）。

⚠ **它是数据，不是逻辑**：不起线程、不连网络、不读业务状态（本文件 import 期零 I/O）。

## 每一行为什么是现在这样（改之前先读；依据都指向设计文档或实测）

| 声明 | 依据 |
|---|---|
| ``id="EL"`` | 暴露名会**小写化**成 ``el_<tool>``；`Site.validate` **大小写不敏感**查重（§2.2.1） |
| ``argv=(python, "app_entry.py", "--mcp")`` | `--mcp` = MCP 服务形态（原 `--authority` 改名）；`run_mcp()` 硬编码 `PROJ_ROOT` ⇒ **只需 cwd** |
| ``ready.port_file=".mcp-port"`` | 与 `src/v2_bind.py` 的**默认发现文件**、`dsh/src/el-params.ts` 的 `mcpPortFile` **同一个名字**（跨语言单一事实） |
| ``ready.ok_codes=()`` | ⭐ **实测**：EL 的 `GET /mcp` 回 **406**（`application/json`；带 `Accept: text/event-stream` 回 400），该端口上**没有**任何 2xx 路由（`GET /` 回 404）。框架默认只认 2xx ⇒ 不放宽会**永远判不就绪**。含义是"**有 HTTP 响应即证明服务在听**"，不是"错误码无所谓" |
| ``ready`` 仍带 `port_file` | 框架的判据是"**端口文件相对 spawn 前变化过**" **且** 真探活（§3 铁律 2）⇒ 旧 `launcher._wait_mcp_port` 那种"文件存在且非空"的假绿（曾把 11 天前的残留判成就绪）不再可能 |
| ``data_root=EL 根`` | 写租约根（`.mecha-v2/` 的上一层）；**同一根同一时刻只能一个写宿主**（§3 铁律 4） |
| ``env["RE0_V2_DATA_ROOT"]`` | ⭐ **状态根**：`src/v2_bind.py:171` 的优先级是 显式 > 这个变量 > el_root。**不钉死** ⇒ 换腿后可能看到**另一份** journal（"切了但看着是空的"，且无任何报错，§5.6） |
| ``env["EL_NO_AI_SERVER"]=""`` | ⚠ 该变量**真值**会让 `v2_bind.py:172` 把 `start_services` 关掉 ⇒ **服务端根本不起**（ready 会超时，却看不出原因）。置空串 = **显式不设这个键**（Python 里空串是假值） |
| ``env["PYTHONIOENCODING"]="utf-8"`` | 子进程 stdout 被重定向到 `runtime/logs/<id>.log` ⇒ **退回 cp936**，一句 `print("⚠…")` 就能让后台任务崩（§5.5(2)）。⚠ 设计文档判它**归 supervisor**，但框架当前**只 setdefault 了 `PYTHONUNBUFFERED`**（实测：全包 `PYTHONIOENCODING` 零命中）⇒ EL 先显式钉住；框架补上后可删 |
| ``has_gui=True`` | EL **有** GUI 窗口（`app_entry.py --gui`）⇒ 面板据此显示「切到手动」（三层架构 §6：`False` 的项目只有 AI 一条腿） |
| ``initial_mode=Mode.AI`` | ⭐ **由启动者声明**（人是启动者）：**从 mecha 启动 = 永远是 AI 组** ⇒ `AI`。⚠ 与软件侧的关系见下：`--mcp` 今天仍**自己** `switch_to('ai')`；**步骤 2/A4** 要改成**读框架注入的 `MECHA_INITIAL_MODE` 并应用它**（缺省回落本声明的值） |
| ``dsh.patches`` | 入库两层：`cordis.patch.yml`（交付层，加载**构建产物** `./lib/index.mjs`）+ `cordis.project.patch.yml`（webserver 端口隔离层）。⚠ 聚合端点那一层（生成物）由**框架**生成（§5.5(4)） |
| ``dsh.plugin_artifacts`` | `dsh/lib/index.mjs` 是**构建产物**（gitignored）⇒ 不存在就**响亮失败**，别等 dsh 起来报看不懂的错。⚠ 前置：`cd dsh && npm run bundle` |
| 框架注入的两个（**不是**本声明的字段，**别在这里钉**） | `MECHA_SOFTWARE_ROOT`（数据根）/ `MECHA_INITIAL_MODE`（`locked`/`human`/`ai`/`open`）——三层架构 §5.1 定案：由 supervisor 在统一注入点给（与 `PYTHONIOENCODING` 同处 ⇒ 带【空串 = 显式取消】语义）。⭐ **软件侧责任（步骤 2/A4）**：`--mcp` 要读 `MECHA_INITIAL_MODE` 并**应用**到闸门，缺省回落本声明的 `initial_mode`（更保守可回落 `LOCKED`）。本轮**只记录、不消费**（硬约束：不改 `--mcp` 现有行为）。⚠ 与 `RE0_V2_DATA_ROOT` 的关系：前者是**框架声明**的数据根、后者是 EL 内部的管线变量 ⇒ 步骤 2 应让两者**同源**（否则又是「看到两份状态」） |
| ``dsh.env={}`` | EL 的 dsh 插件**不读任何环境变量**（实测：`dsh/src/**` 无 `process.env`；它从 cordis config + `.mcp-port` 的 dirname 推项目根）⇒ **不编一个没人读的变量** |
"""

from __future__ import annotations

import sys
from pathlib import Path

from mecha.authority import Mode
from mecha.site import DSHSpec, ReadySpec, Service, Site

#: 本仓根（本文件在仓根 ⇒ 即自身所在目录）。
_SITE_ROOT = Path(__file__).resolve().parent

#: 解释器：用**当前解释器**（与旧 launcher 的 `PY = sys.executable` 同源；
#: 换别的 python 会换掉 site-packages，那是在另一个环境里跑）。
_PY = sys.executable

#: ⭐ **实测**（2026-10-08，真 `--mcp` 子进程）：`GET /mcp` → **406**；`GET /` → 404；
#: 该端口上没有任何 2xx 路由 ⇒ 用 `()` 表达"**任意 HTTP 响应都算在听**"。
_ANY_HTTP: tuple[int, ...] = ()

#: ⭐ **两腿同源**（§5.6）：凡影响"看到哪份状态"的环境变量都在这里显式钉死。
_ENV = {
    "RE0_V2_DATA_ROOT": str(_SITE_ROOT),   # 状态根（journal/artifacts 的父目录）
    "EL_NO_AI_SERVER": "",                 # 置空 = 不设（真值会让 MCP 服务端不起！）
    "PYTHONIOENCODING": "utf-8",           # 日志重定向下防 cp936 崩（见文件头表格）
}


def service_argv() -> tuple[str, ...]:
    """起 EL 的 MCP 服务形态；`--with-gui` 让它在**同一进程**里开窗（复用同一份桥）。

    ⚠ **为什么用旗标**（三层架构 §7）：「切到手动」只在 `has_gui` 的项目显示，而它
    **只切写权**（面板与 dsh 都不关）⇒ 窗口得**先在软件进程里存在**。要不要开窗是
    **每次启动的意图**（人是启动者），不是"这个软件必然要开" ⇒ 用旗标。
    三个东西分工不同、别挤进一个变量：`has_gui` = **能力**（面板显不显示按钮）、
    `MECHA_INITIAL_MODE` = **写权**（框架 §5.1 注入）、`--with-gui` = **本次是否开窗**。
    ⚠ 不加这个旗标 ⇒ 该软件就是 headless（`--mcp` 单独跑的行为**没变**）。
    `run_mcp()` 自己读 `PROJ_ROOT`，不需别的参数。
    """
    return (_PY, "app_entry.py", "--mcp", "--with-gui")


SITE = Site(
    name="Energy Level",
    services=(
        Service(
            id="EL",
            brief="Rb-87 里德堡原子：静态能级 + 动力学演化 + 多原子阵列仿真",
            argv=service_argv(),
            cwd=_SITE_ROOT,
            ready=ReadySpec(
                # 判据 = 端口文件**相对 spawn 前变化过** 且 **真探活通过**（框架两条一起看）。
                port_file=".mcp-port",
                probe="http://127.0.0.1:{port}/mcp",
                ok_codes=_ANY_HTTP,
                timeout_s=90.0,           # ARC cache + assemble：秒级到十几秒，90s 有余量
            ),
            data_root=_SITE_ROOT,
            #: ⭐ 启动者（人）声明的初始写权：从 mecha 启动**永远是 AI 组** ⇒ AI。
            #: （框架把它经 `MECHA_INITIAL_MODE` 送进软件；软件侧的应用是步骤 2。）
            initial_mode=Mode.AI,
            #: ⭐ EL 有 GUI 窗口（`app_entry.py --gui`）⇒ 面板显示「切到手动」。
            has_gui=True,
            env=_ENV,
        ),
    ),
    dsh=DSHSpec(
        command=("dsh",),
        cwd=_SITE_ROOT,
        patches=(
            _SITE_ROOT / "dsh" / "cordis.patch.yml",
            _SITE_ROOT / "dsh" / "cordis.project.patch.yml",
        ),
        #: 3080 是官方 dsh 的端口，项目实例永不占（框架从 3081 起试、实际端口回读）。
        port_preference=3081,
        port_tries=10,
        plugin_artifacts=(_SITE_ROOT / "dsh" / "lib" / "index.mjs",),
        env={},
        ready=None,
    ),
)


__all__ = ["SITE", "service_argv"]
