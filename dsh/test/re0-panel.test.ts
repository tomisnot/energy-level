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
  configOf,
  eventsOf,
  loadPanel,
  okIsNonDegenerate,
  selfCheckLine,
  stateBadge,
  stateText,
} from '../src/dsh-panel/panel-data.ts'
import { recPageHtml } from '../src/dsh-panel/panel-view.ts'

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

// ---------------------------------------------------------------- 0 浏览器安全（结构）
/**
 * 沿**相对 import** 走一遍某入口的导入闭包（技法照资产 `monitor-client.test.ts` 那条，
 * 但**根换成 EL 自己的 client 半入口**）：EL 的文件也可能哪天误 import node-only 的
 * `monitor-url.ts`——那条链会把 `node:fs` 拖进浏览器 bundle。
 */
function closureOf(entry: string): { seen: string[]; nodeHits: string[] } {
  const strip = (src: string) => src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
  const seen = new Set<string>()
  const nodeHits: string[] = []
  const queue = [entry]
  while (queue.length) {
    const rel = queue.shift() as string
    if (seen.has(rel)) continue
    seen.add(rel)
    const isPanel = rel.startsWith('dsh-panel/')
    const abs = join(SRC, isPanel ? 'dsh-panel' : 'client',
      isPanel ? rel.slice('dsh-panel/'.length) : rel)
    for (const line of strip(readFileSync(abs, 'utf8')).split('\n')) {
      if (/^\s*import\s+type\b/.test(line)) continue   // 类型导入会被剥掉，不是运行时依赖
      const m = line.match(/from\s+'([^']+)'/)
      if (!m) continue
      const spec = m[1] ?? ''
      if (spec.startsWith('node:')) { nodeHits.push(`${rel} → ${spec}`); continue }
      if (!spec.startsWith('.')) continue              // 裸包名（react 等）由宿主提供
      const parts = isPanel ? ['dsh-panel', ...rel.slice('dsh-panel/'.length).split('/')]
        : ['client', ...rel.split('/')]
      parts.pop()                                      // 去掉文件名 → 当前目录
      for (const seg of spec.split('/')) {
        if (seg === '.') continue
        else if (seg === '..') parts.pop()
        else parts.push(seg)
      }
      queue.push(parts.join('/'))
    }
  }
  return { seen: [...seen], nodeHits }
}

test('⭐ EL client 半的 import 闭包里没有 node:*（浏览器安全靠结构，不靠树摇）', () => {
  const { seen, nodeHits } = closureOf('dsh-panel/MonitorTabBody.tsx')
  assert.deepEqual(nodeHits, [],
    `client 半闭包里出现 node 内建：${nodeHits.join('、')}`
    + ' ⇒ 浏览器 bundle 会失败（node-only 的实现必须留在 dsh-panel/monitor-url.ts）')
  // R8 自证：闭包必须**真的走过了几个文件**，否则"没命中"只是"什么都没读到"
  assert.ok(seen.length >= 3, `闭包只走了 ${seen.length} 个文件 ⇒ 检查可能没生效：${seen}`)
  assert.ok(seen.includes('dsh-panel/routes.ts'), `闭包里应当有 routes.ts：${seen}`)
  assert.ok(!seen.includes('dsh-panel/monitor-url.ts'),
    `client 半不许碰到 node-only 的 monitor-url.ts：${seen}`)
  // 对偶（不许空洞）：node 半**必须**仍然依赖 node——否则上一条"没东西可抓"
  assert.match(read('dsh-panel/monitor-url.ts'), /from 'node:fs'/,
    'node 半应当 import node:fs（否则本检查没有可抓之物）')
})

// ---------------------------------------------------------------- 2 单一来源
test('⭐ 单一来源：两半都引用参数块（路由字面量不在别处再抄一份）', () => {
  assert.match(read('index.ts'), /path:\s*PANEL_CONFIG\.ROUTE_PATH/,
    'host 半注册的路由必须来自参数块')
  assert.match(read('index.ts'), /portFile:\s*PANEL_CONFIG\.PORT_FILE/,
    'host 半读的端口文件名必须来自参数块')
  assert.match(read('dsh-panel/panel-data.ts'), /fetchMonitorBase\(doFetch,\s*PANEL_CONFIG\.ROUTE_PATH\)/,
    '面板取址必须用参数块的路由')
  const offenders = literalOutsideParams(PANEL_CONFIG.ROUTE_PATH,
    ['index.ts', 'client/index.ts', 'client/buttons.tsx', 'dsh-panel/MonitorTabBody.tsx'])
  assert.deepEqual(offenders, [], `路由字面量被抄进了：${offenders.join('、')}`)
})

// ---------------------------------------------------------------- 3 反面语料
test('反面语料：历史默认端口 8767 不在面板源码里（端口早已动态发现）', () => {
  const offenders = literalOutsideParams('8767',
    ['index.ts', 'client/index.ts', 'client/buttons.tsx', 'dsh-panel/MonitorTabBody.tsx',
      'dsh-panel/panel-config.ts'])
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
  const { state: ok } = await loadPanel(doFetch, [])
  assert.equal(ok.kind, 'ok')
  assert.ok(okIsNonDegenerate(ok), '真喂了数据 ⇒ 必须非退化')
  assert.equal(selfCheckLine(ok), '离线=false 事件行=1 配置族=1', '自证行必须真打出来')
})

// ---------------------------------------------------------------- 4④ 硬验收：飞行记录仪
/**
 * ⭐ **硬验收**（用户可见 bug 的封条）：有事件时「飞行记录仪」必须画**事件行**，
 * 而不是空态文案"暂无写事件"。
 *
 * 由来：旧的自写面板拿**请求路径** `HISTORY_PATH`（`/history?since_seq=0`）去索引
 * `payloads`，而 `payloads` 按**基础路由键** `/history` 存 ⇒ **页面永远"暂无写事件"**，
 * 而自证行仍报"事件行=N"（自证真、渲染假——最难查的一类）。
 * 资产把键收进 `eventsOf()` 一处；本判据守**渲染结果**这一层。
 *
 * 夹具是**真端点实测**（2026-09-26：起真监控端点 → 两条真写 → 抓
 * `/history?since_seq=0` 与 `/config` 原文），不是手搓形状。
 */
const REAL_HISTORY = {
  ok: true, mode: 'ai',
  events: [
    { seq: 1, kind: 'set', actor: 'mcp', target: 'field.B_gauss', value: 21.7, after: 21.7,
      reason: '把磁场拉到 21.7 G（验收夹具）' },
    { seq: 2, kind: 'set', actor: 'mcp', target: 'dynamic.t_max', value: 200.0, after: 200.0,
      reason: '拉长演化时长到 200ns（验收夹具）' },
  ],
}
const REAL_CONFIG = { ok: true, has_schema: true, n_keys: 85, n_snapshot_keys: 7,
  orphans: [], groups: { config: [], engine: [], solve: [] } }

test('⭐ 验收：有事件时「飞行记录仪」画事件行，不是"暂无写事件"（真端点夹具）', async () => {
  const { doFetch, seen } = spyFetch({
    [PANEL_CONFIG.ROUTE_PATH]: { status: 200, body: { base: 'http://127.0.0.1:54321' } },
    ['http://127.0.0.1:54321' + HISTORY_PATH]: { status: 200, body: REAL_HISTORY },
    ['http://127.0.0.1:54321' + CONFIG_PATH]: { status: 200, body: REAL_CONFIG },
  })
  const { state: st } = await loadPanel(doFetch, [])
  assert.equal(st.kind, 'ok')
  assert.equal(eventsOf(st).length, 2, '键对了才拿得到两条事件')
  assert.ok(configOf(st), '配置态也拿得到')
  const html = recPageHtml(eventsOf(st), () => false)
  assert.ok(html.includes('field.B_gauss') && html.includes('dynamic.t_max'),
    `事件行必须真出现在 HTML 里：${html.slice(0, 120)}`)
  assert.ok(html.includes('把磁场拉到 21.7 G（验收夹具）'), 'reason 也要画出来')
  assert.ok(!html.includes('暂无写事件'), '有事件时**不许**出现空态文案（这就是那个 bug）')
  assert.ok(seen.length === 3, `一拍应当恰好三次请求（地址+两条基础路由）：${seen.join('、')}`)
})

// ---------------------------------------------------------------- 4③ 不指相对
test('③ 地址拿不到 ⇒ 一次都不取数（绝不拿空 base 拼相对路径）', async () => {
  const { doFetch, seen } = spyFetch({
    [PANEL_CONFIG.ROUTE_PATH]: { status: 503, body: { error: '监控端点未知：…' } },
  })
  const { state: st } = await loadPanel(doFetch, [])
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
  // ⚠ 这行**不许**写成以 `(` 开头的表达式语句：上一行 `})` 没分号 ⇒ ASI 会把它解析成
  // `spyFetch({...})(await loadPanel(okSpy...))`，于是在 `okSpy` 自己的初始化里读它（TDZ）。
  const { state: okState } = await loadPanel(okSpy.doFetch, [])
  assert.equal(okState.kind, 'ok')
  const dataCalls = okSpy.seen.slice(1)
  assert.equal(dataCalls.length, 2)
  for (const u of dataCalls) assert.match(u, /^https?:\/\//, `取数必须绝对 URL：${u}`)
})
