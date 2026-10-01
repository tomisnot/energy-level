/**
 * 自愈重连桥单测 —— **头号验收**：复现《DSH-MCP现状》报告 §2.3 并证明治好。
 *
 * 用可编程假 Hub（管理 session 集合、可 restart 清空=旧 session 全失效、可 down=宕机）
 * 注入 bridge，确定性地验证：
 *   分类器 · A 首连 · B §2.3 重启自愈(+致死性对照) · C 宕机→可读离线错误→恢复 ·
 *   D 重连后工具集变化再通知 · E Hub 未起时 start() 不阻塞插件加载。
 *
 * 跑：`node --import tsx --test test/reconnect.test.ts`（无需真 MCP 服务/网络）。
 *
 * ⚠ 桥本体是**依赖包** `@mecha/dsh-panel`（junction，磁盘上一份代码；EL 不再持有副本）；
 * 本文件是 EL 自己的**验收套件**（`../src/host/el-hub.ts` 的 `elBridgeOptions` 就是生产路径
 * 用的那个参数填充 ⇒ 本文件跑的是生产接线，不是替身接线）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  MechaMcpBridge, isSessionInvalidError, isConnectionError,
  type McpSession, type ToolDef,
} from '@mecha/dsh-panel/mcp-bridge.ts'
import type { BridgeOptions, SessionFactory } from '@mecha/dsh-panel/mcp-bridge.ts'
import { EL_BRIDGE, PLUGIN } from '../src/el-params.ts'

/**
 * 本文件自己的桥参数构造（**测试本级**）。
 *
 * ⚠ 生产路径不再有这一层：host 半的建桥/参数装配已归资产骨架
 * `@mecha/dsh-panel/mount-host-plugin.ts` ⇒ 原先 `host/el-hub.ts` 的 `elBridgeOptions`
 * 已随骨架删除。本文件是**桥的端到端语义验收**（真实重连 / 可读离线错误），要的是
 * "用 EL 的值装一台真桥"，故在此就地构造——**值仍取自 EL 的参数家**（`EL_BRIDGE`），
 * 于是"EL 的 `offlineHint` 真有消费者"那条断言仍然成立。
 */
function elBridgeOptions(
  sessionFactory: SessionFactory,
  overrides: Partial<BridgeOptions> = {},
): BridgeOptions {
  return {
    sessionFactory,
    logLabel: PLUGIN.name,
    offlineHint: EL_BRIDGE.offlineHint,
    ...overrides,
  }
}

const fastSleep = async (): Promise<void> => {}

function connErr(): Error {
  return Object.assign(new Error('fetch failed'), { cause: { code: 'ECONNREFUSED' } })
}
function sessionErr(): Error {
  return Object.assign(new Error('Session not found'), { code: 404 })
}

/** 可编程假 Hub：session 集合 + restart（清空=旧 session 全失效）+ setDown（宕机）。 */
class FakeHub {
  readonly sessions = new Set<string>()
  readonly created: McpSession[] = []
  down = false
  private gen = 0
  private n = 0
  tools: ToolDef[] = [
    { name: 'set_param', description: 'set a key', inputSchema: { type: 'object' } },
    { name: 'run_compute', description: 'run', inputSchema: { type: 'object' } },
  ]

  restart(): void { this.sessions.clear(); this.gen++ }
  setDown(v: boolean): void { this.down = v }

  readonly sessionFactory = async (): Promise<McpSession> => {
    if (this.down) throw connErr()
    const self = this
    const id = `sess-${this.gen}-${++this.n}`
    this.sessions.add(id)
    const session: McpSession = {
      sessionId: id,
      async listTools() { return { tools: self.tools.map(t => ({ ...t })) } },
      async callTool(name: string, args: unknown) {
        if (self.down) throw connErr()
        if (!self.sessions.has(id)) throw sessionErr()   // 重启后旧 id → 404（§2.3）
        return { ok: true, called: name, args, session: id }
      },
      async close() { self.sessions.delete(id) },
    }
    this.created.push(session)
    return session
  }
}

test('错误分类器：session 失效 / 连接失败 / 普通工具错误 三者可辨', () => {
  assert.equal(isSessionInvalidError(sessionErr()), true)
  assert.equal(isSessionInvalidError({ code: 404 }), true)
  assert.equal(isSessionInvalidError(new Error('Session not found')), true)
  assert.equal(isConnectionError(connErr()), true)
  assert.equal(isConnectionError(new Error('ECONNREFUSED')), true)
  // 我们 Hub 的可教学错误是 result（不 throw）；即便 throw 也不能被误判成 session/网络错而乱重连
  const toolErr = Object.assign(new Error('invalid_value: field.B_gauss 需要 float'), { code: -32000 })
  assert.equal(isSessionInvalidError(toolErr), false)
  assert.equal(isConnectionError(toolErr), false)
})

test('A 首连：发现工具 + 调用成功 + 状态 ready', async () => {
  const hub = new FakeHub()
  const br = new MechaMcpBridge(elBridgeOptions(hub.sessionFactory, { sleep: fastSleep }))
  await br.start()
  assert.equal(br.isReady(), true)
  assert.equal(br.getStatus(), 'ready')
  assert.deepEqual(br.getTools().map(t => t.name), ['set_param', 'run_compute'])
  const res = await br.callTool('set_param', { key: 'field.B_gauss', value: 21.7 }) as { ok: boolean; called: string }
  assert.equal(res.ok, true)
  assert.equal(res.called, 'set_param')
  await br.close()
})

test('B【头号验收】§2.3：Hub 重启→旧 session 404→桥自愈重连(新 session)→调用恢复', async () => {
  const hub = new FakeHub()
  const br = new MechaMcpBridge(elBridgeOptions(hub.sessionFactory, { sleep: fastSleep }))
  await br.start()
  const first = await br.callTool('run_compute', { reason: 'r1' }) as { session: string }

  hub.restart()   // —— Hub 重启：旧 session 全部失效（复现报告 §2.3）——

  // 致死性对照：天真客户端仍持旧 session → 直接 404、永不自愈（证明 bug 真实、fake 忠实）
  const stale = hub.created[0]!
  await assert.rejects(
    () => stale.callTool('run_compute', {}, undefined),
    (e: unknown) => (e as { code?: number }).code === 404,
  )

  // —— 桥自愈：下一次调用遇 404 → 重连(重新 initialize 拿新 session) → 重试成功 ——
  const second = await br.callTool('run_compute', { reason: 'r2' }) as { ok: boolean; session: string }
  assert.equal(second.ok, true)
  assert.notEqual(second.session, first.session, '重连后必须是**新** session（绝不复用失效旧 id）')
  assert.equal(br.getStatus(), 'ready')
  await br.close()
})

test('C Hub 宕机：调用抛**可读离线错误**（非静默 404）；Hub 回来后自动重连成功', async () => {
  const hub = new FakeHub()
  const br = new MechaMcpBridge(elBridgeOptions(hub.sessionFactory, {
    sleep: fastSleep,
    reconnect: { maxAttempts: 3, initialDelayMs: 1, maxDelayMs: 2 },
  }))
  await br.start()
  assert.equal(br.isReady(), true)

  hub.setDown(true)
  // 资产把文案里的"服务离线"与**项目指引**分成两段；EL 的指引经 `elBridgeOptions` 填进来
  // ⇒ 两条断言：① 可读离线错误（非静默 404）；② EL 的 offlineHint **真的有消费者**（R17：
  // 参数换成别的值、这条就红——不是"只要抛错就行"）。
  await assert.rejects(() => br.callTool('run_compute', { reason: 'r' }),
    (e: unknown) => {
      const msg = String((e as Error).message)
      assert.match(msg, /服务离线/)
      assert.match(msg, /launcher\.py/, 'EL 的 offlineHint 必须接进离线错误（参数要有消费者）')
      return true
    })
  assert.equal(br.getStatus(), 'offline')

  hub.setDown(false)   // Hub 回来 → 再调用 → 自动重连成功
  const res = await br.callTool('run_compute', { reason: 'r' }) as { ok: boolean }
  assert.equal(res.ok, true)
  assert.equal(br.getStatus(), 'ready')
  await br.close()
})

test('D 重连后工具集变化 → onTools 再次触发（dsh 据此重注册工具）', async () => {
  const hub = new FakeHub()
  const seen: string[][] = []
  const br = new MechaMcpBridge(elBridgeOptions(hub.sessionFactory, {
    sleep: fastSleep,
    onTools: t => seen.push(t.map(x => x.name)),
  }))
  await br.start()
  assert.deepEqual(seen[0], ['set_param', 'run_compute'])

  hub.tools = [...hub.tools, { name: 'undo', inputSchema: { type: 'object' } }]
  hub.restart()   // 迫使下次调用重连 → connectOnce 重新 listTools
  await br.callTool('set_param', { key: 'x', value: 1 })

  assert.equal(seen.length, 2, 'onTools 应再触发一次')
  assert.deepEqual(seen[1], ['set_param', 'run_compute', 'undo'])
  assert.deepEqual(br.getTools().map(t => t.name), ['set_param', 'run_compute', 'undo'])
  await br.close()
})

test('E Hub 未起时 start() 不抛（不阻塞 dsh 插件加载）', async () => {
  const hub = new FakeHub()
  hub.setDown(true)
  const br = new MechaMcpBridge(elBridgeOptions(hub.sessionFactory, {
    sleep: fastSleep,
    reconnect: { maxAttempts: 2, initialDelayMs: 1, maxDelayMs: 2 },
  }))
  await br.start()               // 不应抛（Hub 不在也让插件加载）
  assert.equal(br.isReady(), false)
  await br.close()
})

test('F 端口漂移（实测报告 §0）：重启换端口 → 旧连接炸 → 重连工厂现读 .mcp-port → 连上新端口', async () => {
  // 生产接线的忠实模拟：sessionFactory 每次建会话经**生产那条解析口**
  // （`elResolveHubUrl`：EL 的端口文件名/默认端点 + 资产的"显式 > 文件 > 默认"规则）现读端口文件；
  // 每个 session 记住自己的端口，端口漂移后旧 session 的 callTool 抛"连不上"（旧进程已死）。
  // 桥的重连循环 + 每次现解的 URL 两者配合，漂移才自愈——缺一即红（对照见下）。
  const { writeFile, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const { elResolveHubUrl } = await import('../src/host/el-hub.ts')
  const pf = join(process.cwd(), '.__reconnect_port_drift__.mcp-port')
  let livePort = '56093'
  const created: string[] = []
  const sessionFactory = async (): Promise<McpSession> => {
    const url = await elResolveHubUrl({ mcpPortFile: pf })   // 生产解析口，每次现读
    const port = url.match(/:(\d+)\//)?.[1] ?? ''
    if (port !== livePort) throw connErr()                 // 建会话时端口已变 → 连不上
    created.push(port)
    const session: McpSession = {
      sessionId: `s-${port}`,
      async listTools() { return { tools: [{ name: 'run_sim' }] } },
      async callTool(name: string) {
        if (port !== livePort) throw connErr()             // 漂移后旧 session 调用炸
        return { ok: true, called: name, port }
      },
      async close() { /* noop */ },
    }
    return session
  }
  try {
    await writeFile(pf, livePort, 'utf8')
    const br = new MechaMcpBridge(elBridgeOptions(sessionFactory, { sleep: fastSleep }))
    await br.start()
    assert.equal(br.isReady(), true)
    const r1 = await br.callTool('run_sim', {}) as { port: string }
    assert.equal(r1.port, '56093')

    // —— 软件重启：新端口写进端口文件，旧端口进程已死 ——
    livePort = '52755'
    await writeFile(pf, livePort, 'utf8')

    // 下一次调用命中旧 session → 抛"连不上" → 桥重连 → 工厂现读端口文件 → 连上 52755。
    const r2 = await br.callTool('run_sim', {}) as { ok: boolean; port: string }
    assert.equal(r2.ok, true)
    assert.equal(r2.port, '52755', '重连后必须落在新端口（工厂每次现解 URL）')
    assert.deepEqual(created, ['56093', '52755'])
    await br.close()
  } finally {
    await rm(pf, { force: true })
  }
})

/**
 * ⭐ **G：首连失败之后再调用**——与 C **是两支**，别合并（2026-10-01 补，来自上游的真回归）。
 *
 * C 的形状是"**先连上**（`isReady()===true`）、再把 Hub 弄宕"：那时 session 还在、
 * `status==='ready'` ⇒ `callTool` 里 `ensureReady()` **直接 return**，错误由
 * `session.callTool` 抛出、本来就落在 `try` 内 ⇒ **C 测的是"连上后掉线"那一支**。
 *
 * G 的形状是"**一开始就连不上**"：`start()` 失败 ⇒ `status='offline'`、无 session ⇒
 * `callTool` 必须自己找路重连；资产曾把 `await ensureReady()` 放在 `try` **之外**
 * ⇒ 这里会抛**裸连接错/404**，而精心写的「服务离线…+`offlineHint`」永远跑不到。
 * ⇒ **C 抓不到这个分支**（它在修复前后都绿），只有本条能抓（R7 的红证见提交说明）。
 */
test('G ⭐ 首连失败之后再调用：仍抛**可读离线错误**（裸 404/裸连接错就是缺陷）', async () => {
  const hub = new FakeHub()
  hub.setDown(true)                      // ← 一开始就连不上（不是"连上后掉线"）
  const br = new MechaMcpBridge(elBridgeOptions(hub.sessionFactory, {
    sleep: fastSleep,
    reconnect: { maxAttempts: 2, initialDelayMs: 1, maxDelayMs: 2 },
  }))
  await br.start()                       // 不抛（Hub 不在也让插件加载）
  assert.equal(br.isReady(), false, '前提：首连确实失败了——否则本条测的不是这个分支')

  await assert.rejects(() => br.callTool('run_sim', {}), (e: unknown) => {
    const msg = String((e as Error).message)
    assert.match(msg, /服务离线/, '首连失败后再调用必须是可读离线错误（裸 404 = 缺陷）')
    assert.match(msg, /launcher\.py/, 'EL 的 offlineHint 在这一支也必须真的有消费者')
    return true
  })
  assert.equal(br.getStatus(), 'offline')
  await br.close()
})
