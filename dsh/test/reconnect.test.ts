/**
 * 自愈重连桥单测 —— **头号验收**：复现《DSH-MCP现状》报告 §2.3 并证明治好。
 *
 * 用可编程假 Hub（管理 session 集合、可 restart 清空=旧 session 全失效、可 down=宕机）
 * 注入 bridge，确定性地验证：
 *   分类器 · A 首连 · B §2.3 重启自愈(+致死性对照) · C 宕机→可读离线错误→恢复 ·
 *   D 重连后工具集变化再通知 · E Hub 未起时 start() 不阻塞插件加载。
 *
 * 跑：`node --import tsx --test test/reconnect.test.ts`（无需真 MCP 服务/网络）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  MechaMcpBridge, isSessionInvalidError, isConnectionError,
  type McpSession, type ToolDef,
} from '../src/host/mcp-bridge.ts'

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
  const br = new MechaMcpBridge({ sessionFactory: hub.sessionFactory, sleep: fastSleep })
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
  const br = new MechaMcpBridge({ sessionFactory: hub.sessionFactory, sleep: fastSleep })
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
  const br = new MechaMcpBridge({
    sessionFactory: hub.sessionFactory, sleep: fastSleep,
    reconnect: { maxAttempts: 3, initialDelayMs: 1, maxDelayMs: 2 },
  })
  await br.start()
  assert.equal(br.isReady(), true)

  hub.setDown(true)
  await assert.rejects(() => br.callTool('run_compute', { reason: 'r' }), /Hub 离线/)
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
  const br = new MechaMcpBridge({
    sessionFactory: hub.sessionFactory, sleep: fastSleep,
    onTools: t => seen.push(t.map(x => x.name)),
  })
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
  const br = new MechaMcpBridge({
    sessionFactory: hub.sessionFactory, sleep: fastSleep,
    reconnect: { maxAttempts: 2, initialDelayMs: 1, maxDelayMs: 2 },
  })
  await br.start()               // 不应抛（Hub 不在也让插件加载）
  assert.equal(br.isReady(), false)
  await br.close()
})
