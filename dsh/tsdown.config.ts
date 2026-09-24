import { defineConfig } from 'tsdown'

/**
 * 单产物（对应 package.json 的 "." 导出）：
 *  · lib/index.js —— HOST 半（node 平台）：把 @modelcontextprotocol/sdk + ws **打进** bundle
 *    （自包含，运行时不依赖 dsh 解析我们的 node_modules）；只 @deepseek-ai/* peer 外置（宿主提供）。
 * dts 关（类型走 src + types/shim；--patch 源加载本就直读 src）。
 *
 * v1 的 client 半（驾驶舱面板，lib/client.js）已随库替换退役——监控面在 GUI 侧边栏。
 */
const dshExternal = [/^@deepseek-ai\//]

export default defineConfig([
  {
    entry: { index: 'src/index.ts' },
    outDir: 'lib',
    format: 'esm',
    platform: 'node',
    target: 'node20',
    external: dshExternal,
    dts: false,
    sourcemap: true,
    clean: true,
  },
])
