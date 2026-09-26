/**
 * **re0-mecha 项目自有**的面板判据（共享资产之外的那一半）。
 *
 * ⚠ 共享单测（`src/dsh-panel/*.test.ts`）**逐字复制、不许改**；项目要加的"自家语料"
 * 按资产 `dsh-panel/README.md` 的约定**放这里**：项目字面量（路由、端口文件名、
 * 历史默认端口）只许出现在本文件与参数块里。
 *
 * 钉四组：
 *  1. **参数块就是 EL 的事实**（路由 / 端口文件名 / 标题）；
 *  2. ⭐ **单一来源**：路由只写在参数块里，host/client 两半都**引用**它（资产 README：
 *     「两半同源同包 ⇒ 单一来源在包内自动满足」）⇒ 按**内容**读源码断言没在别处再抄；
 *  3. **反面语料**：历史默认端口 `8767` 不许出现在面板源码里（它早已改成动态发现）；
 *  4. ⭐ **段 B 三条**（取数走共享半边后必须成立的性质）：
 *     ① 缺端口文件 ⇒ 路由 `503`，且失败体**没有 `base` 字段**；
 *     ② **面板不空白**：每个非 `ok` 档都有**非空可读文案**（R8 对偶：`ok` 档必须**真喂
 *        数据**，否则"两边都空"也会过）；
 *     ③ **地址拿不到时一次都不取数**：绝不拿空 base 去拼相对路径（否则"软件没在 AI
 *        模式跑"会被显示成"连上了但面板空白"）。
 *
 * 跑：`npm test`（同仓其余单测一起）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { PANEL_CONFIG } from '../src/dsh-panel/panel-config.ts'
import { makeMonitorUrlHandler } from '../src/dsh-panel/monitor-url.ts'
import type { PanelState } from '../src/dsh-panel/monitor-client.ts'
import {
  CONFIG_PATH,
  HISTORY_PATH,
  loadPanel,
  okIsNonDegenerate,
  selfCheckLine,
  stateBadge,
  stateText,
} from '../src/client/panelData.ts'

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = join(HERE, '..', 'src')
const read = (rel: string): string => readFileSync(join(SRC, rel), 'utf8')

/**
 * ⚠ **不要** import `client/index.ts`（也不 import 组件）：它 import `MonitorTabBody.tsx`
 * → `react`，而 react 是 **dsh 宿主提供的 peer**、本仓 devDeps 里没有 ⇒ 一 import 就
 * `ERR_MODULE_NOT_FOUND`。所以"单一来源"用**按内容读源码**来断言（判据本来也该能脱离
 * 运行时而成立）；行为判据则落在**无 react 的数据层** `client/panelData.ts` 上。
 */
function literalOutsideParams(literal: string, files: string[]): string[] {
  return files.filter((f) => read(f).includes(literal))
}

/** 最小 res 替身：只记状态码/头/正文（正文如实解码成字符串——见共享单测的同一注解）。 */
function fakeRes() {
  const res: any = {
    statusCode: 0,
    headers: {} as Record<string, string>,
    body: '',
    setHeader(k: string, v: string) { res.headers[k] = v },
    end(b: string | Uint8Array) {
      res.body = typeof b === 'string' ? b : Buffer.from(b).toString('utf8')
    },
  }
  return res
}

/** 最小 fetch 替身：按 URL 路由到给定形状；**记录每个被请求的 URL**（判据③ 的观测量）。 */
function spyFetch(routes: Record<string, { status: number; body: unknown }>) {
  const seen: string[] = []
  const doFetch = async (url: any): Promise<any> => {
    const key = String(url)
    seen.push(key)
    const r = routes[key]
    if (!r) return { ok: false, status: 404, json: async () => ({}) }
    return { ok: r.status >= 200 && r.status < 300, status: r.status, json: async () => r.body }
  }
  return { doFetch: doFetch as unknown as typeof fetch, seen }
}

// ---------------------------------------------------------------- 1 参数块
test('参数块是本仓的事实：路由 / 端口文件 / 标题', () => {
  assert.equal(PANEL_CONFIG.ROUTE_PATH, '/re0-mecha/monitor-url')
  assert.equal(PANEL_CONFIG.PORT_FILE, '.monitor-port')
  assert.ok(PANEL_CONFIG.TITLE.trim().length > 0, '标题不许空')
  assert.ok(PANEL_CONFIG.PORT_FILE.trim().length > 0,
    'PORT_FILE 留空 ⇒ 路由一律 503（共享资产的刻意设计）——本仓必须填')
})

// ---------------------------------------------------------------- 2 单一来源
test('⭐ 单一来源：两半都引用参数块（路由字面量不在别处再抄一份）', () => {
  assert.match(read('index.ts'), /path:\s*PANEL_CONFIG\.ROUTE_PATH/,
    'host 半注册的路由必须来自参数块')
  assert.match(read('index.ts'), /portFile:\s*PANEL_CONFIG\.PORT_FILE/,
    'host 半读的端口文件名必须来自参数块')
  assert.match(read('client/panelData.ts'), /fetchMonitorBase\(doFetch,\s*PANEL_CONFIG\.ROUTE_PATH\)/,
    '面板取址必须用参数块的路由')
  const offenders = literalOutsideParams(PANEL_CONFIG.ROUTE_PATH,
    ['index.ts', 'client/index.ts', 'client/MonitorTabBody.tsx', 'client/buttons.tsx',
      'client/panelData.ts'])
  assert.deepEqual(offenders, [], `路由字面量被抄进了：${offenders.join('、')}`)
})

// ---------------------------------------------------------------- 3 反面语料
test('反面语料：历史默认端口 8767 不在面板源码里（端口早已动态发现）', () => {
  const offenders = literalOutsideParams('8767',
    ['index.ts', 'client/index.ts', 'client/MonitorTabBody.tsx', 'client/buttons.tsx',
      'client/panelData.ts', 'dsh-panel/panel-config.ts'])
  assert.deepEqual(offenders, [], `面板源码里不该再出现 8767：${offenders.join('、')}`)
})

// ---------------------------------------------------------------- 4① 503 无 base
test('① 缺端口文件 ⇒ 路由 503，且失败体**没有 base 字段**（不回半截地址）', () => {
  const dir = mkdtempSync(join(tmpdir(), 're0-panel-'))
  try {
    const res = fakeRes()
    makeMonitorUrlHandler({ root: dir, portFile: PANEL_CONFIG.PORT_FILE })({} as any, res)
    assert.equal(res.statusCode, 503)
    const body = JSON.parse(res.body)
    assert.equal(body.base, undefined, '失败体不许带 base（否则面板会拿它去 fetch）')
    assert.match(String(body.error), /监控端点未知/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

// ---------------------------------------------------------------- 4② 不空白
test('② 面板不空白：每个非 ok 档都有非空可读文案；ok 档必须**真喂数据**（R8 对偶）', async () => {
  const branches: PanelState[] = [
    { kind: 'address', detail: '读不到监控端口（…不存在）' },
    { kind: 'offline', detail: '/history 返回 503', failed: [HISTORY_PATH] },
    { kind: 'route', detail: '/extra 返回 500', failed: ['/extra'] },
    { kind: 'empty', detail: '连上了但没拿到数据（离线=false 事件行=0 配置族=0）' },
  ]
  for (const st of branches) {
    assert.ok(stateText(st).trim().length > 0, `档 ${st.kind} 的文案不许空（空白=缺陷）`)
    assert.ok(stateBadge(st).text.trim().length > 0, `档 ${st.kind} 的徽标不许空`)
  }
  assert.equal(stateText({ kind: 'ok', stats: { offline: false, events: 1, configFamilies: 1 },
    payloads: {} }), '', 'ok 档不画文案（它画数据）')

  // R8 对偶：替身**真喂数据**——否则"offline vs offline""空 vs 空"同样会过，绿得没信息。
  const { doFetch } = spyFetch({
    [PANEL_CONFIG.ROUTE_PATH]: { status: 200, body: { base: 'http://127.0.0.1:54321' } },
    ['http://127.0.0.1:54321' + HISTORY_PATH]: {
      status: 200, body: { mode: 'human', events: [{ seq: 1, key: 'field.B_gauss' }] } },
    ['http://127.0.0.1:54321' + CONFIG_PATH]: {
      status: 200, body: { has_schema: true, groups: { field: [] }, orphans: [] } },
  })
  const ok = await loadPanel(doFetch)
  assert.equal(ok.kind, 'ok')
  assert.ok(okIsNonDegenerate(ok), '真喂了数据 ⇒ 必须非退化')
  assert.equal(selfCheckLine(ok), '离线=false 事件行=1 配置族=1', '自证行必须真打出来')
})

// ---------------------------------------------------------------- 4③ 不指相对
test('③ 地址拿不到 ⇒ 一次都不取数（绝不拿空 base 拼相对路径）', async () => {
  const { doFetch, seen } = spyFetch({
    [PANEL_CONFIG.ROUTE_PATH]: { status: 503, body: { error: '监控端点未知：…' } },
  })
  const st = await loadPanel(doFetch)
  assert.equal(st.kind, 'address')
  assert.ok(stateText(st).length > 0)
  // 只问了**地址路由**（同源相对路径，那是它的本职）；**没有任何取数**发生。
  assert.deepEqual(seen, [PANEL_CONFIG.ROUTE_PATH],
    `地址拿不到时不该再取数，实际请求了：${seen.join('、')}`)

  // 对照：地址拿得到时，两条**取数都必须是绝对 URL**（相对 = 打到 dsh 自己身上）
  const okSpy = spyFetch({
    [PANEL_CONFIG.ROUTE_PATH]: { status: 200, body: { base: 'http://127.0.0.1:54321' } },
    ['http://127.0.0.1:54321' + HISTORY_PATH]: { status: 200, body: { mode: 'ai', events: [{}] } },
    ['http://127.0.0.1:54321' + CONFIG_PATH]: { status: 200, body: { groups: { a: [] } } },
  })
  await loadPanel(okSpy.doFetch)
  const dataCalls = okSpy.seen.slice(1)
  assert.equal(dataCalls.length, 2)
  for (const u of dataCalls) assert.match(u, /^https?:\/\//, `取数必须绝对 URL：${u}`)
})
