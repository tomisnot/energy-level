# DSH 的 MCP 现状 —— 给"接下来要做什么"的参考资料

> 起因：re0-mecha 接入 DSH 后，**重启 MCP 服务导致客户端永久失联**（`Session not found`），
> 无法在本会话内恢复。本文把 DSH 的 MCP 实现现状查清（源码级），供决定
> "项目该包成 MCP server 还是 DSH 插件"。
>
> 调查对象：`@deepseek-ai/dsh@0.1.5-rc.2`、`@deepseek-ai/dsh-mcp-client`
> （`lib/index.js` 790 行）、`@modelcontextprotocol/sdk@1.30.0`。
> 日期：2026-09-19。所有结论都给了文件/行号，可直接复核。

---

## 0. 一句话结论

**DSH 的 MCP 桥在 stdio 上是有重连的，在 streamable-http 上等于没有重连**——
因为重连只由 `transport.onclose` 触发，而 HTTP 传输在服务重启后**只抛错、不触发 close**，
且 SDK 不会用 404 去重置 session。⇒ **HTTP 型 MCP server 一旦重启，必须重启 Host 或改配置才能恢复。**

**对我们的建议：优先 stdio。**（理由见 §4）

---

## 1. 客户端配置面（全部字段，无遗漏）

`dsh-mcp-client` 的 Config 是 `z.union([stdio, streamable-http])`（`lib/index.js:743-761`）：

| 字段 | stdio | http | 默认 | 说明 |
|---|:--:|:--:|---|---|
| `serverName` | ✅ | ✅ | 必填 | 工具命名空间：`mcp__<serverName>__<tool>`；`[A-Za-z0-9_-]{1,32}`，同一 scope 内唯一 |
| `command`/`args`/`env`/`cwd` | ✅ | — | — | stdio 专用；`env` 合并在**已擦洗**的环境之上 |
| `url`/`headers` | — | ✅ | — | HTTP 端点与额外请求头 |
| `toolCallTimeoutMs` | ✅ | ✅ | `60000` | 每次 `tools/call` 超时 |
| `failOnStartupError` | ✅ | ✅ | `false` | 首次连接失败是否让插件激活失败 |
| `reconnect.{enabled,initialDelayMs,maxDelayMs,maxAttempts}` | ✅ | ✅ | `true / 500 / 30000 / 10` | 见 §2 |

**没有的字段**（我逐个核过，别指望）：
- ❌ 没有 HTTP 侧 `sessionId`（想手工指定/清空 session 做不到）
- ❌ 没有连接超时/发现超时（继承 SDK 的 60s 默认；README 明说这是开放项）
- ❌ 没有 resources / prompts 桥（**只桥 tools**）
- ❌ 没有"强制重连"或"重载"的工具/接口

---

## 2. 重连机制：只认 `onclose`

### 2.1 实现

```js
// lib/index.js:627-631   —— 唯一的断线入口
generation.onclose = () => {
  closeObserved = true;
  closed.resolve();
  if (attemptSettled) generationDown(generation);   // → scheduleReconnect()
};

// lib/index.js:560-565
function generationDown(generation) {
  if (!isCurrent(generation)) return;
  client = void 0; clientClosed = void 0;
  scheduleReconnect();
}
```

`scheduleReconnect`（`:579-605`）就是 README 描述的策略：`initialDelayMs` 起、**每失败一次翻倍**、
封顶 `maxDelayMs`；连续失败超过 `maxAttempts` → **注销全部工具、停止重连**；
连接**稳定存活超过 `maxDelayMs`** 会**重置**尝试预算。日志会打
`connection lost; reconnecting in <delay>ms (attempt n/N)`。

### 2.2 关键：HTTP 走不到这条路

README《Known Limitations》原文（`README.md:193`）：

> **Reconnect triggers on transport close** — a crashed stdio child fires it;
> **Streamable HTTP failures surface per request through the SDK transport's own
> recovery, so an unreachable HTTP server is retried per call rather than
> respawned by the supervisor.**

源码印证：`createTransport`（`:40-50`）对 http 只 `new StreamableHTTPClientTransport(url, {requestInit:{headers}})`，
**没有任何 session 生命周期管理**。

而 SDK 1.30.0 的 HTTP 传输在收到 404 时（`dist/esm/client/streamableHttp.js:365`）：

```js
throw new StreamableHTTPError(response.status, `Error POSTing to endpoint: ${text}`);
```

→ **只抛错**。**不重置 `this._sessionId`、不调用 `this.onclose`**。
（只有 401/403 有 auth 重试分支；404 无任何分支。）

### 2.3 后果链条（我们踩的正是这个）

```
① server 下发 Mcp-Session-Id（实测响应头确实有）
② server 重启 → 旧 session 失效
③ 客户端仍带旧 id 发请求 → 404 "Session not found"
④ SDK 抛错，但**保留旧 id**、**不触发 onclose**
⑤ DSH supervisor 在等 onclose → **重连永不启动**
⑥ 每次工具调用都 404；工具仍在列表里（不消失），但永远失败
⑦ 只有"改配置重载"或"重启 Host"能恢复
```

**实测证据（re0-mecha，2026-09-19）**：

| 我发的 `Mcp-Session-Id` | 服务端回应 |
|---|---|
| 不带 | **400** Bad Request |
| 任意假 id | **404** Not Found |
| 重启前的旧 id | **404** Not Found |
| `initialize` 的响应头 | **`Mcp-Session-Id: d3694a2f…`**（服务端**确实下发并强制** session） |

⇒ 这个 server **不是 stateless**。若它是真 stateless（完全忽略该头），旧 id 无害，
也就不会卡死——**这就是修复方向的关键分叉**（见 §4）。

---

## 3. 两种传输的行为对比

| | stdio | streamable-http |
|---|---|---|
| 谁拉起进程 | **harness**（`command`/`args`/`cwd`） | 外部（人/服务） |
| 进程崩了 | 子进程退出 → `onclose` → **自动重连** ✅ | 传输不感知 → **不重连** ❌ |
| 服务重启 | harness 重启它，天然新连接 | **旧 session 失效 → 永久 404** ❌ |
| 重连预算是否有效 | **有效** | **形同虚设**（永不触发） |
| 启动失败 | `failOnStartupError` 可让 Host 启动失败 | 同 |
| 环境 | `env` 在**已擦洗**环境上合并（`/KEY|PASSWORD|SECRET|TOKEN/i` 与 `DSH_*` 被丢弃） | 走 `headers`，无擦洗概念 |
| 适合 | 本地可执行、由 harness 管生命周期 | 已常驻的服务、远程/多客户端 |

---

## 4. 对"接下来要做什么"的建议

### 4.1 传输选型

**若目标只是"让 AI 能驱动我们的引擎" → 选 stdio。** 这是本次调查最直接的结论：

- 重连**真的会工作**（进程崩溃→自动拉起→工具自动重同步）
- 生命周期归 harness，不用人记着先起服务
- 不占端口，天然避开"两个项目抢 8765"这类冲突（我们本轮真实踩过）

**若必须 HTTP**（多客户端/远程/服务已常驻），需接受并处理这条约束：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. **服务端做成真 stateless** | 完全忽略 `Mcp-Session-Id`（不校验、不下发） | 放弃服务端 session 语义；需确认 SDK 客户端是否容忍 |
| B. **进程级自愈** | server 侧被 kill/重启后，由外部守护（如 launcher 包一层）拉起**并**通知人重载 DSH 配置 | 仍需人工介入一次 |
| C. **改用 stdio** | 见上 | 需要 harness 能 spawn（本项目 AI 侧本就"不 spawn 物理设备进程"，这是**安全策略冲突**，要权衡） |

> ⚠ 注意：re0-mecha 的协议里明确写了"**写通道永不出口本机；由人运行 launcher；harness 绝不拉起权威**"——
> 这是**刻意的安全设计**（不让 agent 去 spawn 控制激光的进程）。**所以对这个项目，stdio 与安全策略冲突**，
> 不能无脑选 A 方案。这是要你们裁决的点。

### 4.2 包成 DSH 插件 vs 保持 MCP server

| | MCP server（现状） | DSH 插件 |
|---|---|---|
| 接入成本 | 低（写配置即可，本项目已验证） | 需实现 Cordis 插件（`apply(ctx, config)`） |
| harness 可替换性 | **高**（任何 MCP 客户端都能用） | **低**（绑 DSH） |
| 连接生命周期 | 受 §2 约束（HTTP 无重连） | **自己管**（可用 stdio seam / 自建 supervisor） |
| 工具注册 | 经 `ctx.tools`，名字 `mcp__server__tool` | 直接用 `ctx.tools.register()`，命名自由 |
| 能否桥 resources/prompts | ❌ 只桥 tools | 可自行决定 |
| 谁能用 | 任意 harness | 只有 DSH |

**判断**：
- 若你们的**核心诉求是"AI 接入"本身**、且希望**不被单一 harness 绑死** → 保持 MCP server，
  但**解决 §2 的重连缺口**（对我们而言：接受"重启后开新会话"的操作纪律，或把 server 做成 stateless）。
- 若你们想要**更深的集成**（自定义工具命名、resources、把引擎状态当 DSH 一等概念） → 包插件；
  但**注意插件会把你们绑到 Cordis/DSH 的 API 上**，而 MCP 是跨 harness 的标准。

### 4.3 不管选哪条，都值得做的三件事

1. **给"断联"一个显式信号**。现在工具**仍在列表里但调用全失败**——使用者（人或 AI）
   看到的是"工具在、但坏了"，很难归因。建议 server 侧提供一个**廉价的存活探测**
   （我们已有 `/healthz` 的先例），让 AI 能先探活再决定是否报错。
2. **把"重启服务"写进操作纪律**：DSH 会话中途重启 MCP 服务 = 本会话该 server 作废。
   要么服务常驻，要么"重启服务 + 开新会话"成对出现。
3. **配置改动是可恢复路径**（README:92 明说 "Editing the configuration entry reloads the
   server connection in place"）——**这是 HTTP 卡死时唯一不重启 Host 的出路**，
   值得在文档里写清楚（改一个无关紧要的字段再改回来，可能就能触发重载）。

---

## 5. 附录：可直接复核的证据清单

| 结论 | 位置 |
|---|---|
| Config schema（全部字段） | `dsh-mcp-client/lib/index.js:743-761` |
| 重连默认值 | `:471-477`（`RECONNECT_DEFAULTS`） |
| 重连策略实现 | `:579-605`（`scheduleReconnect`） |
| 唯一断线入口 = onclose | `:627-631` |
| 失败尝试预算/注销工具 | `:589-595` |
| HTTP 传输无 session 管理 | `:40-50`（`createTransport`） |
| SDK 对 404 只抛错 | `@modelcontextprotocol/sdk@1.30.0` `dist/esm/client/streamableHttp.js:365` |
| SDK 只在 `close()` 时触发 onclose | 同上 `:281-288` |
| "HTTP 重连归属仍是开放项" | `dsh-mcp-client/README.md:208`（Dev Note） |
| "只桥 tools" | `README.md:191`（Known Limitations） |
| "改配置可就地重载连接" | `README.md:92` |

**未验证但值得跟进**：DSH 是否支持**运行时重载**某个 MCP 配置项（README 说"编辑配置条目会就地重载"，
但没写这个"编辑"是指改文件后由 HMR 拾取，还是必须重启）。这决定 §4.1 方案 B 是否可行。

---

## 6. 补充实测：热重载**没能**救活卡死的客户端（负面结论）

DSH 的 profile 配置里确实有 `patchReload` 策略（`package.json` 的 `dsh.profile.patchReload`），
我们的 web profile **明确是 `"live"`**：

```json
"dsh": { "profile": { "bundles": [...], "patchReload": "live" } }
```

实现上，`"live"` 会走 `watchUserPatches(ctx, {filename: <profile>/cordis.patch.yml, …})`
（`lib/profile-boot-Dk-7KqJc.js:321-334`），用 Cordis HMR 的 `hmr.registerConfig(filename, …)`
在文件变化时 `entry.update({config:{...patches}})`——**看起来**"改 patch 文件即就地重载连接"。

**但实测（2026-09-19，DSH 0.1.5-rc.2）**：

| 我做的改动 | 结果 |
|---|---|
| 给 `cordis.patch.yml` 追加一行注释（mtime 更新确认） | ❌ 仍 `Session not found` |
| 再追加一条**结构性**条目 `- insert: []` | ❌ 仍 `Session not found` |

⇒ **`patchReload: "live"` 在本次实测中没有恢复卡死的 MCP 客户端。**
可能原因（未逐一定性）：① 改动只触发了 patch 层的重组，没有销毁并重建那个 mcp-client 插件实例；
② HMR 的 config 监听未覆盖 `cordis.patch.yml` 这一层；③ 重建了但 session 失效状态被延续。

**实践含义**：目前**唯一可靠的恢复路径是重启 DSH Host**（`dsh web`）。
不要指望"改配置"能救——至少不能当作保证。

> 这一条建议开发者**自己复现一次**再决定要不要把它写进使用纪律：
> 步骤 = 起一个 HTTP MCP server → 让 DSH 连上 → 重启该 server → 改 `cordis.patch.yml` → 看工具是否恢复。
> 若能复现"救不活"，那"重启服务必须配一次 Host 重启"就该写死进文档。
