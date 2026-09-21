# re0-mecha-dsh —— Re0 机甲 Hub 的 dsh 集成插件

一个 **out-of-tree** 的 dsh（Cordis）插件，把 Re0 机甲 Hub 接进 dsh：**接管 MCP 连接生命周期**
（治好 dsh stock HTTP 桥"服务重启即永久 `Session not found`"的死区），并把工具**原生注册**进
dsh。设计依据见 `docs/DSH-MCP现状-给开发者的参考资料.md` 与《dsh 集成插件·实施方案》。

## 它解决什么

dsh 的 MCP 桥只在 `transport.onclose` 时重连；streamable-http 的 POST 失败（Hub 重启后旧
`Mcp-Session-Id` → 404）**只抛错、不触发 onclose**，客户端永久卡死，只能重启 dsh Host。我们
Hub 恰是 stateful HTTP，落进这个死区。本插件用**自写的自愈桥**接管：任何一次调用遇 session
失效/连不上 → 拆旧连接 → 重新 `initialize`（拿**新** session）→ 重试；并有后台退避重连、
健康信号、Hub 未起时不阻塞加载。

## 架构（分层，核心可独立单测）

```
src/index.ts               HOST 入口：apply(ctx,config) → 建桥 + 注册工具 + 状态信号
src/host/mcp-bridge.ts     自愈重连桥（**零 SDK/零 dsh 依赖**，纯逻辑，可注入 fake 单测）
src/host/mcp-session-http.ts  唯一 import @modelcontextprotocol/sdk：每次新建会话（=重连拿新 session）
src/host/register-tools.ts MCP 工具 → ctx.tools.register（照 dsh mcp-client/tools.ts 的两阶段 swap）
src/host/config.ts         Hub 端点解析（hubUrl / .mcp-port / 默认）
types/dsh-shims.d.ts       本地最小类型 shim（@deepseek-ai/* 是宿主提供的 peer，本地装不到）
```

**安全红线**：插件**只 connect、绝不 spawn Hub**（人启动 launcher = 权威，T4）。换 harness 只丢
本插件，Hub 的独立 MCP server 照用（跨 harness）。

## 加载

**⚠ 首次/新检出必先装依赖**：host 代码 import `@modelcontextprotocol/sdk` + `ws`，缺 `dsh/node_modules`
则 dsh 加载插件即崩（`ERR_MODULE_NOT_FOUND: @modelcontextprotocol/sdk`）。node_modules 不入 git，
故覆盖/检出后必须重装：

```sh
cd dsh && npm install && cd ..
```

（走 AI 模式启动器 `python 能级.py` 会自动检测缺失并 npm install，可略过手动这步。）

```sh
# 开发期（tsx 直接加载源、无需构建；但仍需上面装过依赖）
dsh web --patch ./dsh/cordis.source.patch.yml
# 交付期（先构建）
cd dsh && npm run bundle && cd ..
dsh web --patch ./dsh/cordis.patch.yml
```

配置（在 patch/profile 里给该 entry 的 `config`）：

| 字段 | 默认 | 含义 |
|---|---|---|
| `hubUrl` | `''` | 显式 Hub MCP 端点；留空则从 `mcpPortFile` 解析 |
| `mcpPortFile` | `.mcp-port` | launcher 写的端口文件（相对 dsh 的 cwd；建议给绝对路径） |
| `serverName` | `re0-mecha` | 工具命名空间 → `mcp__re0-mecha__<tool>` |

## 本地开发

```sh
cd dsh
npm install --legacy-peer-deps      # @deepseek-ai/* 是 peer（宿主提供），故 legacy-peer-deps
npm run typecheck                   # tsc --noEmit（core + host 胶水 + shim）
npm test                            # node --test：自愈重连桥单测（含 §2.3 致死性复现）
```

> node_modules 别进版本库/同步盘（已 .gitignore；若在 BaiduSync 内，建议把 `dsh/node_modules`
> 排除出同步）。

## 验证边界（诚实交代）

**已在本地确定性验证**（无需 dsh / 无需网络）：
- `npm run typecheck`：core + host 胶水 + client + shim 全量 `tsc --noEmit` 零错误（已开
  `erasableSyntaxOnly`，见下方约束）。
- `npm test`：**11/11 绿**——自愈桥 6 项（**头号验收**"Hub 重启→旧 session 404→自愈重连(新
  session)→恢复"，含致死性对照）+ config 5 项（buildRequestInit 鉴权头 / resolveHubUrl）。
- **Node strip-types 加载 host 入口**：`node -e "import('./src/index.ts')"` → `HOST_LOAD_OK`。
  这**复刻了 dsh 加载 host 半的机制**（Node 24 原生 type-stripping），本地即可确认 host 能加载。

**⚠ strip-only 约束（踩过）**：dsh 用 **Node 原生 strip-types** 加载 host 的 `.ts`——**只擦类型、
不生成代码**。故 host 侧（`src/index.ts` + `src/host/*`）**禁用**：参数属性
（`constructor(private x)`）、`enum`、`namespace`、JSX。tsconfig 已开 `erasableSyntaxOnly: true`
让 `tsc` 本地拦住这类语法。client 侧（`src/client/*.tsx`）的 JSX 由 dsh web 客户端打包器处理，
不受此限（但 client 半需 `npm run bundle` 产出 `lib/client.js`，见下）。

**已活体验证**（2026-09-20，dsh 0.1.5-rc.2，真 `dsh web --patch` + 浏览器）：
1. ✅ `dsh web --patch ./dsh/cordis.source.patch.yml` 加载插件（host 日志 `[re0-mecha] 连接 Hub MCP` → `已连上`）。
2. ✅ 起 Hub（`python -m src.adapter.serve`）→ dsh 里 AI 能调 `mcp__re0-mecha__*` 工具。
3. ✅ **杀掉 Hub 再重启（不重启 dsh）→ 工具自动恢复**（头号验收；对照报告 §2.3 的永久卡死）。
4. ✅ 会话头部出现「◈ 驾驶舱」按钮 → iframe 显示配置态树/事件/告警（client 半 out-of-tree 加载成功）。

**活体踩过、已修的两个加载格式坑**（out-of-tree 插件专属，务必记住）：
- **host = Node 原生 strip-types（strip-only）**：只擦类型、不生成代码。TS **参数属性**
  （`constructor(private x)`）/`enum`/`namespace` 会 `ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX` 使**整棵插件树**
  加载失败。→ 显式字段+构造器赋值；tsconfig 已开 `erasableSyntaxOnly` 本地拦。
- **client = dsh `__ModuleLoader__` classic-script 合并包**：每个 client.js 必须自注册
  `window.__ModuleLoader__.load({id:<包名>, factory:(require)=>{ var module={exports:{}}; ...; return module.exports }})`。
  普通 ESM（顶层 import/export）在合并包里是 **SyntaxError → 毒化整包**（报错会误点名别的模块如
  `dsh-client-hmr`）。→ tsdown 用 `format:'cjs'` + `outputOptions.banner/intro/footer` 复刻（见
  `tsdown.config.ts`）；`react`/`react/jsx-runtime`/`@deepseek-ai/cordis` 是 dsh 的 PLATFORM_MODULES，
  必须 **external 经注入的 require 共享**（勿内联，双 React 会崩）。

## 远程 / 跨设备（需求③：引擎在 A 机、控制在 B 机）

分两条通道，成熟度不同：

- **远程“看”（已可用）**：cockpit-server 早已支持 `--host 0.0.0.0 --token <t>`（只读 + 鉴权；
  `cockpit_selfcheck` E 段测过：无 token→401 / 有 token→200 / POST→405）。B 机的 dsh 面板把
  `config.cockpitUrl` 指向 `http://A:8766/?token=<t>` 即可远程看 A 的配置态树/事件/告警。
- **远程“控制/写”（仅接缝、默认关）**：T6 铁律——MCP **写通道今天只 localhost**（不让能写
  激光的端点裸奔公网）。插件已留 `hubToken` 接缝（→ `Authorization: Bearer`，`buildRequestInit`
  单测覆盖）；将来 Hub 写通道若经 **L3 双闸**（人确认 + 硬件互锁）安全地远程暴露，插件带
  token 即可连远程 Hub，无需改插件逻辑。
- **Hub headless**：`python -m src.adapter.serve --host <iface>` 可无 GUI 跑在 A 机；GUI/cockpit/dsh
  都是它的客户端（星型拓扑）。跨设备的 **GUI** 传输缝是 Phase B（AppBridge 可插拔传输）的事。

一句话：远程**只读监控今天就能跨设备**；远程**写控制**是 L3 门控的未来项，插件侧接缝已就位。

## 现状 / 待办

**已写 + 本地验证**（tsc 零错、单测 6/6、tsdown 产出 lib/index.mjs + lib/client.js 且语法校验通过）：
- host 半：自愈 MCP 桥 + 工具原生注册 + 断联信号 + `webserver/index-inject` 注入 cockpitUrl。
- client 半：`src/client/*` —— `ctx.slots` 挂"◈ 驾驶舱"头部动作，iframe 嵌 cockpit-server
  （配置态树/事件/告警）。数据层在 Python 侧：`src/cockpit` 的 `/api/config-tree`（判据
  `cockpit_config_tree` 5/19 绿；全量基线 32/0）。
- 构建：`npm run bundle` → `lib/index.mjs`（host，@deepseek-ai/* 与 sdk 外置）+ `lib/client.js`
  （react 外置）。

**待活体确认**（需真 dsh web，开发机跑不了）：
- out-of-tree 的 **client 半加载**：package.json `dsh.client` + `/client` 导出能否被 dsh web
  client loader 挂载（in-tree 范例 inspector/agent-team 走 workspace 装配；out-of-tree --patch
  需实测）。退路＝host 半照常（工具+重连），cockpit 用独立标签页开 `http://127.0.0.1:8766/`。
- `webserver/index-inject` 事件在未 inject `webServer` 时是否仍触达（本插件为保 headless 可用
  只监听不 inject）；若 web 模式收不到 bootstrap，则改为 inject `webServer` 或拆一个 web-only 子插件。
- **远程（需求③，Phase 4）**：cockpit-server 网络可达 + 鉴权；`hubUrl`/`cockpitUrl` 指向远程。
