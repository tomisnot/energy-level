/**
 * EL 的端点解析与远程鉴权头单测（`src/host/el-hub.ts` + 共享资产的 `buildRequestInit`）。
 *
 * ⚠ **规则归资产包**（`@mecha/dsh-panel/config.ts`：显式 URL > 端口文件 > 项目默认，
 * 缺省**不猜端口**）；**EL 的值**（默认端点、端口文件名）住 `el-params.ts`，由
 * `el-hub.ts` 填进去。⇒ 本文件测的是**EL 那条生产接线**，不是资产内部规则
 * （资产自己的规则由它的共享单测守）。
 *
 * 跑：`node --import tsx --test test/config.test.ts`
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildRequestInit } from '@mecha/dsh-panel/config.ts'
import { elHubUrlConfig, elResolveHubUrl } from '../src/host/el-hub.ts'
import { EL_BRIDGE } from '../src/el-params.ts'

test('EL 填给资产的参数：端口文件名与默认端点由 EL 给（资产不猜）', () => {
  const c = elHubUrlConfig({})
  assert.equal(c.mcpPortFile, '.mcp-port')
  assert.equal(c.defaultUrl, 'http://127.0.0.1:8765/mcp')
  // 部署显式给的值优先；空/空白回落到 EL 默认（两个方向都验，防"参数没接上"）
  assert.equal(elHubUrlConfig({ mcpPortFile: '/tmp/x.port' }).mcpPortFile, '/tmp/x.port')
  assert.equal(elHubUrlConfig({ mcpPortFile: '   ' }).mcpPortFile, EL_BRIDGE.mcpPortFile)
})

test('buildRequestInit：hubToken → Authorization Bearer 头', () => {
  assert.deepEqual(
    buildRequestInit({ hubToken: 'secret123' }),
    { headers: { Authorization: 'Bearer secret123' } },
  )
})

test('buildRequestInit：无 token / 空 token → undefined（本地默认不带头）', () => {
  assert.equal(buildRequestInit({}), undefined)
  assert.equal(buildRequestInit({ hubToken: '' }), undefined)
})

test('buildRequestInit：额外头与 token 合并', () => {
  const ri = buildRequestInit({ hubToken: 't', hubHeaders: { 'X-Trace': '1' } }) as RequestInit
  assert.deepEqual(ri.headers, { 'X-Trace': '1', Authorization: 'Bearer t' })
})

test('elResolveHubUrl：显式 hubUrl 优先（远程设备地址原样用）', async () => {
  assert.equal(
    await elResolveHubUrl({ hubUrl: 'http://deviceA:9000/mcp' }),
    'http://deviceA:9000/mcp',
  )
})

test('elResolveHubUrl：端口文件不存在 → 回落 EL 的默认端点 8765', async () => {
  const url = await elResolveHubUrl({ mcpPortFile: '.__nonexistent_port_file__' })
  assert.equal(url, EL_BRIDGE.defaultUrl)
})

test('elResolveHubUrl：每次调用现读端口文件（权威重启换端口 → 解析跟着变）', async () => {
  const { writeFile, rm } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const pf = join(process.cwd(), '.__port_drift_test__.mcp-port')
  try {
    await writeFile(pf, '56093', 'utf8')
    assert.equal(await elResolveHubUrl({ mcpPortFile: pf }), 'http://127.0.0.1:56093/mcp')
    await writeFile(pf, '52755', 'utf8')   // —— 软件重启，端口漂移 ——
    assert.equal(await elResolveHubUrl({ mcpPortFile: pf }), 'http://127.0.0.1:52755/mcp')
  } finally {
    await rm(pf, { force: true })
  }
})
