/**
 * **re0-mecha 项目自有**的面板判据（资产包之外的那一半）。
 *
 * 2026-10-01 **依赖化之后**：资产是包 `@mecha/dsh-panel`（`file:` 依赖 ⇒ node_modules 里是
 * **junction**，磁盘上只有一份代码）。⇒ 本文件守的东西也变了：
 *
 * | 不再守（承诺已换人 / 已结构性消灭） | 现在由谁守 |
 * |---|---|
 * | "副本与参考实现逐字一致"（指纹表） | **结构**：不存在副本了（路径是 junction ⇒ 改它等于改 mecha）。原 `checks/dsh_panel_asset_selfcheck.py` 已随副本一起删 |
 * | 503 无 `base` / 非 ok 档文案 / 地址拿不到不取数 / 记录仪渲染 | 资产自带单测（mecha 的门禁跑它们） |
 * | `panelConfig()` 未注入即抛、字段留空即抛 | `@mecha/dsh-panel/panel-data.test.ts:122-129` |
 *
 * 本文件只守**EL 自己的取值与接线**：
 *  1. **EL 注入的那三个值**（路由 / 端口文件名 / 标题）与注入往返；
 *  2. ⭐ **单一来源**：路由字面量只许出现在 `el-params.ts`，两半都经 `configurePanel` 注入；
 *  3. **反面语料**：历史默认端口 `8767` 不许出现在 EL 的面板源码里；
 *  4. ⭐ **EL 入口 + 资产包 的 import 闭包里没有 `node:*`**——闭包检查**跟着 junction 走**
 *     （`@mecha/dsh-panel/…` 解析到 node_modules 里那份），所以它同时证明：
 *     资产包真的被解析到、且它的 client 半没把 node-only 模块拖进浏览器 bundle。
 *     ⚠ 这条是"结构性无法消灭的那一类"，不是脚手架（资产 README 原话）。
 *
 * 跑：`npm test`（同仓其余单测一起）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, relative } from 'node:path'
import { configurePanel, panelConfig } from '@mecha/dsh-panel/panel-config.ts'
import { CONFIG_PATH, HISTORY_PATH, eventsOf, loadPanel } from '@mecha/dsh-panel/panel-data.ts'
import { recPageHtml } from '@mecha/dsh-panel/panel-view.ts'
import { EL_PANEL } from '../src/el-params.ts'

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = join(HERE, '..', 'src')
/** 资产包的落点（`file:` 依赖 ⇒ 这是指向 mecha 仓的 **junction**）。 */
const PANEL_DIR = join(HERE, '..', 'node_modules', '@mecha', 'dsh-panel')
const PKG = '@mecha/dsh-panel/'

const read = (rel: string): string => readFileSync(join(SRC, rel), 'utf8')

function literalOutsideParams(literal: string, files: string[]): string[] {
  return files.filter((f) => read(f).includes(literal))
}

// ---------------------------------------------------------------- 1 EL 注入的值
test('EL 注入的面板参数：值就是 EL 的，且注入能被读回（往返）', () => {
  assert.equal(EL_PANEL.routePath, '/re0-mecha/monitor-url')
  assert.equal(EL_PANEL.portFile, '.monitor-port')
  assert.ok(EL_PANEL.title.trim().length > 0, '标题不许空')
  const dispose = configurePanel(EL_PANEL)
  try {
    assert.deepEqual(panelConfig(), EL_PANEL, '注入后必须原样读回（注入点真的在链上）')
  } finally {
    dispose()
  }
})

// ---------------------------------------------------------------- 2 单一来源
test('⭐ 单一来源：路由字面量只在 el-params.ts，两半都经 configurePanel 注入', () => {
  for (const f of ['index.ts', 'client/index.ts']) {
    assert.match(read(f), /configurePanel\(EL_PANEL\)/, `${f} 必须注入 EL 的面板参数`)
  }
  assert.match(read('el-params.ts'), /'\/re0-mecha\/monitor-url'/,
    '路由字面量必须住在 el-params.ts（EL 的参数家）')
  const offenders = literalOutsideParams(EL_PANEL.routePath,
    ['index.ts', 'client/index.ts', 'client/buttons.tsx'])
  assert.deepEqual(offenders, [], `路由字面量被抄进了：${offenders.join('、')}`)
})

// ---------------------------------------------------------------- 3 反面语料
test('反面语料：历史默认端口 8767 不在 EL 的面板源码里（端口早已动态发现）', () => {
  const offenders = literalOutsideParams('8767',
    ['index.ts', 'client/index.ts', 'client/buttons.tsx', 'el-params.ts'])
  assert.deepEqual(offenders, [], `面板源码里不该再出现 8767：${offenders.join('、')}`)
})

// ---------------------------------------------------------------- 4 浏览器安全（结构）
/**
 * 沿**相对 import 与 `@mecha/dsh-panel/…`** 走一遍导入闭包。
 *
 * ⚠ 闭包**跟着 junction 走进资产包**（`node_modules/@mecha/dsh-panel/…`）：这样"资产包真的
 * 被解析到"与"它的 client 半里没有 `node:*`"由**同一条**检查一起证明。
 * 其它裸包名（`react` 等）由宿主提供，不追。
 */
function closureOf(entryAbs: string): { seen: string[]; nodeHits: string[] } {
  const strip = (src: string) => src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
  const label = (p: string): string => (p.startsWith(PANEL_DIR)
    ? 'panel/' + relative(PANEL_DIR, p).replace(/\\/g, '/')
    : 'el/' + relative(SRC, p).replace(/\\/g, '/'))
  const seen = new Set<string>()
  const nodeHits: string[] = []
  const queue = [entryAbs]
  while (queue.length) {
    const file = queue.shift() as string
    if (seen.has(file)) continue
    seen.add(file)
    for (const line of strip(readFileSync(file, 'utf8')).split('\n')) {
      if (/^\s*import\s+type\b/.test(line)) continue   // 类型导入会被剥掉，不是运行时依赖
      const m = line.match(/from\s+'([^']+)'/)
      if (!m) continue
      const spec = m[1] ?? ''
      if (spec.startsWith('node:')) { nodeHits.push(`${label(file)} → ${spec}`); continue }
      if (spec.startsWith(PKG)) { queue.push(join(PANEL_DIR, spec.slice(PKG.length))); continue }
      if (spec.startsWith('.')) { queue.push(join(dirname(file), spec)); continue }
      // 其它裸包名（react / react-jsx-runtime 等）：宿主提供，不追
    }
  }
  return { seen: [...seen].map(label), nodeHits }
}

test('⭐ EL 入口 + 资产包的 import 闭包里没有 node:*（跟着 junction 走）', () => {
  const { seen, nodeHits } = closureOf(join(SRC, 'client', 'index.ts'))
  assert.deepEqual(nodeHits, [],
    `闭包里出现 node 内建：${nodeHits.join('、')}`
    + ' ⇒ 浏览器 bundle 会失败（node-only 的实现必须留在资产的 monitor-url.ts）')
  // R8 自证：闭包必须**真的走过了若干文件**，且**真的进了资产包**
  assert.ok(seen.length >= 6, `闭包只走了 ${seen.length} 个文件 ⇒ 检查可能没生效：${seen}`)
  assert.ok(seen.includes('panel/routes.ts'),
    `闭包必须走进资产包（junction 没被跟随？）：${seen}`)
  assert.ok(seen.includes('el/el-params.ts'), `闭包里应当有 EL 的参数家：${seen}`)
  assert.ok(!seen.includes('panel/monitor-url.ts'),
    `client 半不许碰到 node-only 的 monitor-url.ts：${seen}`)
  // 对偶（不许空洞）：资产的 node 半**必须**仍然依赖 node——否则上一条"没东西可抓"
  assert.match(readFileSync(join(PANEL_DIR, 'monitor-url.ts'), 'utf8'), /from 'node:fs'/,
    'node 半应当 import node:fs（否则本检查没有可抓之物）')
})

// ---------------------------------------------------------------- 5 迁移验收项
/**
 * ⭐ 资产 README 给 EL 的**迁移验收项**：「飞行记录仪」页在**有事件时**必须显示事件行
 * （那是当初从 EL 搬进资产时抓到的**用户可见 bug**：组件用请求路径去索引按基础路由存的
 * payloads ⇒ 永远"暂无写事件"）。
 *
 * 这里不启动 dsh（用户明令：我们本身就在 dsh 里），而是**喂真形状的 `/history` + `/config`
 * 回执**走一遍资产的数据层与视图层——顺带证明 **EL 注入的参数真的被消费**（第一跳就是
 * EL 的 `routePath`，不是资产的中性默认）。
 */
test('⭐ 迁移验收：注入的参数被数据层消费，且「飞行记录仪」有事件时画事件行', async () => {
  const dispose = configurePanel(EL_PANEL)
  try {
    const seen: string[] = []
    const base = 'http://127.0.0.1:54321'
    const ok = (body: unknown) => ({ ok: true, status: 200, json: async () => body })
    const doFetch = async (url: unknown) => {
      const u = String(url)
      seen.push(u)
      if (u === EL_PANEL.routePath) return ok({ base })
      if (u === base + HISTORY_PATH) {
        return ok({ mode: 'ai', events: [{ seq: 1, kind: 'set', actor: 'ai',
          target: 'field.B_gauss', value: 21.7, after: 21.7,
          reason: '把磁场拉到 21.7 G（迁移验收夹具）' }] })
      }
      if (u === base + CONFIG_PATH) {
        return ok({ has_schema: true, groups: { config: [] }, orphans: [],
          n_keys: 1, n_snapshot_keys: 1 })
      }
      return { ok: false, status: 404, json: async () => ({}) }
    }
    const { state } = await loadPanel(doFetch as never, [])
    assert.equal(state.kind, 'ok', `面板没进 ok 档：${JSON.stringify(state).slice(0, 140)}`)
    assert.equal(seen[0], EL_PANEL.routePath,
      '第一跳必须是**EL 注入的路由**（注入真的被消费）')
    const evs = eventsOf(state)
    assert.equal(evs.length, 1, '拿得到事件（键对了）')
    const html = recPageHtml(evs, () => false)
    assert.ok(html.includes('field.B_gauss') && html.includes('把磁场拉到 21.7 G'),
      `有事件时必须画事件行与 reason：${html.slice(0, 140)}`)
    assert.ok(!html.includes('暂无写事件'), '有事件时**不许**出现空态文案（那就是当初那个 bug）')
  } finally {
    dispose()
  }
})
