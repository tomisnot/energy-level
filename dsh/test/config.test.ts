/**
 * config 纯逻辑单测：buildRequestInit（远程鉴权头，需求③接缝）+ resolveHubUrl（端点解析）。
 * 跑：node --import tsx --test test/config.test.ts
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildRequestInit, resolveHubUrl } from '../src/host/config.ts'

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

test('resolveHubUrl：显式 hubUrl 优先（远程设备地址原样用）', async () => {
  assert.equal(
    await resolveHubUrl({ hubUrl: 'http://deviceA:9000/mcp' }),
    'http://deviceA:9000/mcp',
  )
})

test('resolveHubUrl：无 hubUrl 且端口文件不存在 → 回落默认 8765', async () => {
  const url = await resolveHubUrl({ mcpPortFile: '.__nonexistent_port_file__' })
  assert.equal(url, 'http://127.0.0.1:8765/mcp')
})
