# re0-mecha-dsh —— Re0 软件的 dsh 集成插件

> ## ⛔ 2026-10-10：**自愈 MCP 桥已整块删除**（用户裁决 = 方案 A）—— 读本文前先看这条
>
> **工具面只有一条腿**：**框架的聚合端点**（dsh 侧由框架的 overlay 插
> `@deepseek-ai/dsh-mcp-client`，EL 的工具名形如 `mcp__mecha__el_<tool>`，
> **读/切写权与开关软件的元工具也在那条腿上**）。
> ⇒ 本仓以下东西**已删除**：`src/host/`（`el-hub.ts` + 桥）、`src/el-params.ts`、
> `test/config.test.ts`、`test/reconnect.test.ts`，以及依赖 `@modelcontextprotocol/sdk` / `ws`。
> ⭐ **现状**：本包只有**面板**那一件事 —— host 半 = **空载体**（只为"一条 patch entry 带一个包"
> 的加载形状），client 半 = 框架面板插件的**一行 bootstrap**。
> ⚠ **代价（用户已认）**：**不经 mecha 单独跑 dsh 时，不再出现任何 EL 工具**。
> ⚠ 反向判断：`dsh/test/re0-panel.test.ts` §6 与框架的 `checks/panel_one_leg_selfcheck.py`。
> ⚠ **本文中凡描述"接管 MCP 连接生命周期 / 建桥 / 注册工具 / SDK 依赖"的段落都已失效**，
> 原样保留只作历史记录（那段死区分析本身仍成立，只是**不再由本插件解决**）。
>
> ## ⚠ 旧库引用横幅（2026-09-25 加，主代理指令）：本文主体（插件架构/加载/配置）仍有效；
> 其中提到 **`mecha-sdk`** 的地方只是**历史背景**（v1 的 client 半随该旧库一起消失）。
> EL 现运行的框架是 **`mecha` 0.2.0**（本机位置见 `README.md` §4），
> 与 0.1.0 模块集完全不同，故凡引用 `mecha-sdk/...py:行号` 的段落**已失效**，
> 请勿据以定位代码。当前真值源：本仓 `docs/capabilities.md` + mecha 仓 `docs/v2/`。

一个 **out-of-tree** 的 dsh（Cordis）插件，把 Re0 软件（mecha v2）接进 dsh：**接管 MCP
连接生命周期**（治好 dsh stock HTTP 桥"服务重启即永久 `Session not found`"的死区），并把
工具**原生注册**进 dsh。设计依据见
内部记录（未随仓发布，存档在仓外）（v1 期取证，死区分析仍成立）。

> **v1 的 client 半（◈驾驶舱面板）已退役**：它 iframe 的 cockpit-server（:8766）随
> mecha-sdk 一起消失；"人监控 AI"的面改由 **GUI 侧边栏（Monitor 面板）**承担（mecha v2
> 第一交付物）。本插件只剩 host 半 = 自愈 MCP 桥 + 工具注册。

## 它解决什么

dsh 的 MCP 桥只在 `transport.onclose` 时重连；streamable-http 的 POST 失败（软件重启后旧
`Mcp-Session-Id` → 404）**只抛错、不触发 onclose**，客户端永久卡死，只能重启 dsh Host。
软件的 MCP 端点恰是 stateful HTTP，落进这个死区。本插件用**自写的自愈桥**接管：任何一次
调用遇 session 失效/连不上 → 拆旧连接 → 重新 `initialize`（拿**新** session）→ 重试；并有
后台退避重连、健康信号、软件未起时不阻塞加载。

**端口漂移自愈**（实测报告 §0：权威重启换端口 56093→52755 曾致老会话永久失联）：桥的
会话工厂**每次建会话现读 `.mcp-port`**（`resolveHubUrl` 不在加载时固化），所以重连循环
的下一拍就落在新端口上——软件重启换端口**无需重启 dsh**。显式 `hubUrl`（远程设备）仍
固化，行为不变。单测 F 组钉这个组合（旧连接炸→重连→现解→连上新端口）。

## 架构（分层，核心可独立单测）

```
src/index.ts               HOST 入口：apply(ctx,config) → 建桥 + 注册工具 + 状态信号
src/host/mcp-bridge.ts     自愈重连桥（**零 SDK/零 dsh 依赖**，纯逻辑，可注入 fake 单测）
src/host/mcp-session-http.ts  唯一 import @modelcontextprotocol/sdk：每次新建会话（=重连拿新 session）
src/host/register-tools.ts MCP 工具 → ctx.tools.register（照 dsh mcp-client/tools.ts 的两阶段 swap）
src/host/config.ts         MCP 端点解析（hubUrl / .mcp-port / 默认）
types/dsh-shims.d.ts       本地最小类型 shim（@deepseek-ai/* 是宿主提供的 peer，本地装不到）
```

**安全红线**：插件**只 connect、绝不 spawn 权威**（框架起那个进程，T4）。换 harness
只丢本插件，软件的 MCP 端点照用（跨 harness）。

## 加载

**⚠ 首次/新检出仍需装依赖**：本包的 **client 半**要从 `@mecha/dsh-panel`（`file:` 依赖，
junction 指向 mecha 仓）取框架那份面板插件 —— 缺 `dsh/node_modules` 时**构建**就失败。
⚠ **2026-10-10 起 host 半不再 import `@modelcontextprotocol/sdk` / `ws`**（桥已删）
⇒ "dsh 加载插件即崩 `ERR_MODULE_NOT_FOUND`"那个故障模式随桥一起消失。
node_modules 不入 git，故覆盖/检出后必须重装：

```sh
cd dsh && npm install && cd ..
```

（走 AI 模式启动器 `python 能级.py` 会自动检测缺失并 npm install，可略过手动这步。）

```sh
# 开发期（Node **原生剥类型**直接加载 TS 源：无需 tsx、无需构建；但仍需上面装过依赖）
dsh web --patch ./dsh/cordis.source.patch.yml
# 交付期（先构建）
cd dsh && npm run bundle && cd ..
dsh web --patch ./dsh/cordis.patch.yml
```

⚠ **本插件现在只提供"面板 + 载体"**：AI 的工具面**不来自本包**（`mcp` 那类工具）
—— 它由**框架的聚合端点**提供（`python -m mecha` 起 dsh 时经 overlay 插入的
`@deepseek-ai/dsh-mcp-client`）。单独 `dsh web --patch ./dsh/cordis.patch.yml`
只能看到面板，**看不到任何 EL 工具**（方案 A 的既定代价）。

配置（在 patch/profile 里给该 entry 的 `config`）：

| 字段 | 默认 | 含义 |
|---|---|---|
| `hubUrl` | `''` | 显式 MCP 端点；留空则从 `mcpPortFile` 解析 |
| `mcpPortFile` | `.mcp-port` | 软件 GUI 进程写的端口文件（相对 dsh 的 cwd；建议给绝对路径） |
| `serverName` | `re0-mecha` | 工具命名空间 → `mcp__re0-mecha__<tool>` |

## 本地开发

```sh
cd dsh
npm install --legacy-peer-deps      # @deepseek-ai/* 是 peer（宿主提供），故 legacy-peer-deps
npm run typecheck                   # tsc --noEmit（host 胶水 + shim）
npm test                            # node --test：自愈重连桥单测（含 §2.3 致死性复现）
```

> node_modules 别进版本库/同步盘（已 .gitignore；若在 BaiduSync 内，建议把 `dsh/node_modules`
> 排除出同步）。

## 验证边界（诚实交代）

**已在本地确定性验证**（无需 dsh / 无需网络）：
- `npm run typecheck`：host 胶水 + shim 全量 `tsc --noEmit` 零错误（已开
  `erasableSyntaxOnly`，见下方约束）。
- `npm test`：**13/13 绿**——自愈桥 7 项（**头号验收**"软件重启→旧 session 404→自愈
  重连(新 session)→恢复"，含致死性对照；F 组=端口漂移自愈）+ config 6 项
  （buildRequestInit 鉴权头 / resolveHubUrl 含现读端口文件）。
- **Node strip-types 加载 host 入口**：`node -e "import('./src/index.ts')"` → `HOST_LOAD_OK`。
  这**复刻了 dsh 加载 host 半的机制**（Node 原生 type-stripping），本地即可确认 host 能加载。

**⚠ strip-only 约束（踩过）**：dsh 用 **Node 原生 strip-types** 加载 host 的 `.ts`——
**只擦类型、不生成代码**。故 host 侧（`src/index.ts` + `src/host/*`）**禁用**：参数属性
（`constructor(private x)`）、`enum`、`namespace`、JSX。tsconfig 已开 `erasableSyntaxOnly: true`
让 `tsc` 本地拦住这类语法。

**已活体验证**（2026-09-20，dsh 0.1.5-rc.2，真 `dsh web --patch` + 浏览器）：
1. ✅ `dsh web --patch ./dsh/cordis.source.patch.yml` 加载插件（host 日志 `[re0-mecha] 连接 Hub MCP` → `已连上`）。
2. ✅ 起 Hub → dsh 里 AI 能调 `mcp__re0-mecha__*` 工具。
3. ✅ **杀掉 Hub 再重启（不重启 dsh）→ 工具自动恢复**（头号验收；对照报告 §2.3 的永久卡死）。

（第 4 项"◈ 驾驶舱按钮"属 v1 client 半，已退役，不再验。）

## 远程 / 跨设备（需求③：引擎在 A 机、控制在 B 机）

- **远程"控制/写"（仅接缝、默认关）**：T6 铁律——MCP **写通道今天只 localhost**（不让能写
  激光的端点裸奔公网）。插件已留 `hubToken` 接缝（→ `Authorization: Bearer`，
  `buildRequestInit` 单测覆盖）；将来写通道若经 **L3 双闸**（人确认 + 硬件互锁）安全地
  远程暴露，插件带 token 即可连远程，无需改插件逻辑。
- **v1 的"远程看"（cockpit-server --host 0.0.0.0）已随驾驶舱退役**；跨设备监控等 GUI 侧
  有真实需求时按 ADR 流程另议。

一句话：远程**写控制**是 L3 门控的未来项，插件侧接缝已就位。
