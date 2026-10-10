/**
 * **re0-mecha 项目自有**的面板判据（资产包之外的那一半）。
 *
 * 2026-10-01 **依赖化**（资产是包 `@mecha/dsh-panel`，junction 指向 mecha ⇒ 磁盘只有一份），
 * 2026-10-09 **监控面上提**（参数与那条"地址路由"都归框架）⇒ 本文件守的东西又变窄了：
 *
 * | 不再守（承诺已换人 / 已结构性消灭） | 现在由谁守 |
 * |---|---|
 * | "副本与参考实现逐字一致"（指纹表） | **结构**：不存在副本了（junction） |
 * | 503 无 `base` / 非 ok 档文案 / 地址拿不到不取数 / 记录仪渲染 | 资产自带单测（mecha 门禁跑它们） |
 * | `panelConfig()` 未注入即抛、字段留空即抛 | `@mecha/dsh-panel/panel/panel-data.test.ts` |
 * | ⭐ **"EL_PANEL 与声明两处同值"** | ⭐ **结构**：**只剩一个家**（`mecha_site.py` 的 `Service.monitor`）⇒ 没有第二个可漂的值；"声明 → 那份参数 → 每个 tab 各读各的"由 mecha 的 `phase13_panel_decl_selfcheck`（夹具↔三家真声明对拍）与 `panel/auto.test.ts`（②③）守 |
 * | ⭐ **"地址路由读 `.monitor-port` 现取端口"** | ⭐ **框架的 host 插件**：`mecha/dsh-panel/host-plugin/mecha-panel-host.test.ts`（含**与资产实现的等价判据**） |
 *
 * 本文件只守**EL 自己的接线**（五条，都会红）：
 *  1. ⭐ **那一行 bootstrap**：client 半必须接上**框架的** client 插件
 *     （少了它 = 右栏一个 tab 都没有，而那不会报错）；
 *  2. ⭐ **不许再自己注册面板/地址路由**（`panelConfig` / `EL_PANEL` **不许回来**）：
 *     ⚠ 同一条 `routePath` 注册两次是接线错误（谁生效取决于顺序）；
 *  3. **反面语料**：历史默认端口 `8767` 不许出现在 EL 的面板源码里；
 *  4. ⭐ **EL 入口 + 资产包 的 import 闭包里没有 `node:*`**——闭包**跟着 junction 走**，
 *     所以它同时证明：资产包真的被解析到、且它的 client 半没把 node-only 模块拖进浏览器 bundle。
 *  5. ⭐ **2026-10-10（方案 A）：host 半必须是空载体** —— 那条"项目自己接 MCP"的
 *     自愈桥整块删除了；`src/host/` 不存在、`el-params.ts` 不存在、`mountHostPlugin`
 *     与 `@mecha/dsh-panel/host/` **不许**出现在本仓源码里（工具面只有一条腿）。
 *
 * 跑：`npm test`（同仓其余单测一起）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join, relative } from 'node:path'

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = join(HERE, '..', 'src')
const REPO = join(HERE, '..', '..')
/** 资产包的落点（`file:` 依赖 ⇒ 这是指向 mecha 仓的 **junction**）。 */
const PANEL_DIR = join(HERE, '..', 'node_modules', '@mecha', 'dsh-panel')
const PKG = '@mecha/dsh-panel/'

const read = (rel: string): string => readFileSync(join(SRC, rel), 'utf8')

/** 去掉注释与字符串外的说明文字：本文件几处"不许出现 X"的断言看的是**代码**，
 *  ⚠ 而注释里解释"X 已删/不许回来"是**该留的**（否则下一个人查不到为什么不能走老路）
 *  ⇒ 必须先剥注释，否则那几处说明会把自己的判据弄红（本轮实测踩到）。 */
const stripComments = (src: string): string => src
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/(^|[^:])\/\/.*$/gm, '$1')

/** 本仓 dsh 的源码文件（**含 `.tsx`**：死文件曾经就是这么漏掉的）。 */
function sourceFiles(): string[] {
  const out: string[] = []
  const walk = (dir: string): void => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name)
      if (e.isDirectory()) walk(p)
      else if (/\.tsx?$/.test(e.name)) out.push(relative(SRC, p).replace(/\\/g, '/'))
    }
  }
  walk(SRC)
  return out
}

/** 源码里（**剥掉注释后**）含某标识的文件。 */
function codeHits(needle: string): string[] {
  return sourceFiles().filter((f) => stripComments(read(f)).includes(needle))
}

// ---------------------------------------------------------------- 1 一行 bootstrap
test('⭐ client 半只有那一行 bootstrap（框架的代码，不是本项目的领域代码）', () => {
  assert.match(read('client/index.ts'),
    /export \{ apply, name, inject \} from '@mecha\/dsh-panel\/panel\/auto\.ts'/,
    'client 半必须有那一行 bootstrap（少了它 = 右栏一个 tab 都没有，且不报错）')
})

// ---------------------------------------------------------------- 2 不许再自己注册
test('⭐ 面板参数与地址路由**都归框架** ⇒ 本仓不许再有 panelConfig / EL_PANEL / 自己的 tab 注册', () => {
  // ⚠ 判据**改过**（2026-10-09）：原先这一条钉的是"host 半必须把 `EL_PANEL` 交给骨架"。
  //    上提之后**反过来**：再交一次就会与框架那条地址路由**撞同一条路径**
  //    （谁生效取决于注册顺序 —— 那是静默的错，正是要防的形态）。
  assert.doesNotMatch(stripComments(read('index.ts')), /panelConfig\s*:/,
    'host 半**不许**再传 panelConfig（框架的 host 插件按声明注册那条地址路由）')
  const offenders = codeHits('EL_PANEL')
  assert.deepEqual(offenders, [], `EL_PANEL 又回来了（参数的第二处真值）：${offenders.join('、')}`)
  // 也不许自己注册右栏 tab / 头部按钮（阶段① 起归框架：按钮**唯一一个**）。
  // ⚠ 只禁"**自己注册**"；`inject` 里声明 `sidebarRightTabs` 是**必须的**（cordis 的 ctx 是代理：
  //   框架那份插件要经本仓的 ctx 用这些服务，不声明就抛）—— 把"声明服务"当"自己注册"来禁
  //   会造出一条**假判据**（GP 侧实测踩到）。
  assert.doesNotMatch(stripComments(read('client/index.ts')),
    /sidebarRightTabs\s*\.\s*register|slots\s*\.\s*register/,
    '本仓不许再自己注册右栏 tab / 槽位（监控面板归框架）')
})

// ---------------------------------------------------------------- 3 反面语料
test('反面语料：历史默认端口 8767 不在 EL 的面板源码里（端口早已动态发现）', () => {
  const offenders = codeHits('8767')
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
    ? 'pkg/' + relative(PANEL_DIR, p).replace(/\\/g, '/')
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
    + ' ⇒ 浏览器 bundle 会失败（node-only 的实现必须留在资产的 host/ 面）')
  // R8 自证：闭包必须**真的走过了若干文件**，且**真的进了资产包**（不写死资产内部目录名）
  assert.ok(seen.length >= 6, `闭包只走了 ${seen.length} 个文件 ⇒ 检查可能没生效：${seen}`)
  assert.ok(seen.some(s => s.startsWith('pkg/') && s.endsWith('/routes.ts')),
    `闭包必须走进资产包（junction 没被跟随？）：${seen}`)
  assert.ok(seen.some(s => s.startsWith('pkg/') && s.endsWith('/panels.ts')),
    `闭包必须走进框架那份 client 插件（面板参数靠它按声明挂 tab）：${seen}`)
  assert.ok(seen.some(s => s.startsWith('pkg/') && s.endsWith('/auto.ts')),
    `闭包必须走进框架的入口（那一行 bootstrap 指的就是它）：${seen}`)
  assert.ok(!seen.some(s => s.includes('monitor-url')),
    `client 半不许碰到 node-only 的 monitor-url.ts：${seen}`)
  // 对偶（不许空洞）：资产的 node 半**必须**仍然依赖 node——否则上一条"没东西可抓"
  assert.match(readFileSync(join(PANEL_DIR, 'host', 'monitor-url.ts'), 'utf8'), /from 'node:fs'/,
    'node 半应当 import node:fs（否则本检查没有可抓之物）')
})

// ---------------------------------------------------------------- 5 声明还在（唯一家）
test('⭐ 那个唯一家还在：`mecha_site.py` 里有 monitor=MonitorSpec(...)', () => {
  // ⚠ 本判据**只**断言"声明还在"（EL 侧的那一半）；**值**由 mecha 的
  //   `phase13_panel_decl_selfcheck` 拿**真声明**对拍（夹具↔真声明），不在这里抄一遍
  //   （抄一遍就是第二处真值 —— 那正是本次上提要消灭的东西）。
  const decl = readFileSync(join(REPO, 'mecha_site.py'), 'utf8')
  assert.match(decl, /monitor=MonitorSpec\(/, '声明里没有 monitor=MonitorSpec(...)')
  assert.match(decl, /route_path="/, '声明里 monitor 缺 route_path（框架按它注册地址路由）')
  assert.match(decl, /port_file="/, '声明里 monitor 缺 port_file（框架按它现取端口）')
})

// ---------------------------------------------------------------- 6 ⭐ 空载体（方案 A）
test('⭐ host 半是**空载体**：桥那套不许回来（工具面只有一条腿 = 框架的聚合端点）', () => {
  // ⚠ 这是 **2026-10-10 用户裁决（方案 A）** 的机器守卫。为什么非要钉：
  //   桥是"每个项目自己接 MCP"时代的产物，而聚合端点是**后加的** ⇒ 两条腿并存时
  //   ① 只开一家也会加载别家的桥（`merge_dsh` 的 patches 是全站点并集）；② 工具名重复；
  //   ③ ⭐ **能力不等**：写权/开关元工具只在聚合端点那条腿上。
  //   ⇒ 删掉桥之后，三家**接法相同**；⚠ 代价（用户已认）：不经 mecha 单独跑 dsh 时无工具。
  const idx = stripComments(read('index.ts'))
  for (const needle of ['mountHostPlugin', 'hostPluginInject', 'dsh-panel/host']) {
    assert.ok(!idx.includes(needle), `host 半不许再出现 ${needle}（桥已按方案 A 删除）`)
  }
  assert.match(idx, /export const inject: string\[\] = \[\]/,
    'host 半必须**不依赖任何服务**（`inject = []`）——它只是载体')
  assert.match(idx, /export async function apply\(\)/,
    'host 半的 apply **不许**再接受 ctx/config（没有桥要装了）')

  // 结构性：桥的家（`src/host/`）与桥的参数家（`el-params.ts`）都**不该存在**。
  const src = sourceFiles()
  assert.ok(!src.some(f => f.startsWith('host/')),
    `src/host/ 又出现了（桥的家）：${src.join('、')}`)
  assert.ok(!src.includes('el-params.ts'),
    `el-params.ts 又出现了（桥的参数家）：${src.join('、')}`)
  // R8 自证：本判据确实读到了源码（不是"文件都不存在所以全绿"）
  assert.ok(src.includes('index.ts') && src.includes('client/index.ts'),
    `判据没读到本仓源码：${src.join('、')}`)
})
