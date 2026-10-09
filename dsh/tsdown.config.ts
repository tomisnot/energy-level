import { defineConfig } from 'tsdown'

/**
 * 两个产物（对应 package.json 的 "." 与 "./client" 导出）：
 *  · lib/index.js   —— HOST 半（node 平台）：把 @modelcontextprotocol/sdk + ws + **资产包**
 *    **打进** bundle（自包含，运行时不依赖 dsh 解析我们的 node_modules）；只 @deepseek-ai/*
 *    peer 外置（宿主提供）。
 *  · lib/client.js  —— CLIENT 半（browser 平台）：react 外置（dsh web client 提供，避免双 React）。
 * dts 关（类型走 src + types/shim；--patch 源加载本就直读 src）。
 *
 * ⭐ **`alwaysBundle` 是依赖化之后的生命线**（2026-10-01）：`@mecha/dsh-panel` 是 `file:`
 * 依赖 ⇒ 默认会被 tsdown 当"依赖"**外置**（bundle 里只剩一句 `import '@mecha/…'`）⇒
 * 打包产物**缺代码**，只在运行时才炸。实测：不加这一项 bundle 从 36 kB 掉到 10 kB。
 * ⇒ 必须显式 `deps.alwaysBundle` 把它拉回来（两个半都要：host 用桥、client 用面板）。
 *
 * client 半 = dsh 界面：◈监控 按钮（开右栏 tab）+ 监控面板
 * （**地址不硬编码**：面板经资产包 `monitor-client.ts` 的 `fetchMonitorBase` 打 host 半的
 * 同源只读路由（路由由 `configurePanel(EL_PANEL)` 注入）取 base，再轮询其 `/history` + `/config`）。
 * ⚠「切换至 GUI」按钮**已删**（三层架构：切写权走控制端点、不动进程）。
 */
const dshExternal = [/^@deepseek-ai\//]

/** 资产包必须**打进** bundle（它是 `file:` 依赖 ⇒ 默认会被外置）。 */
const alwaysBundle = [/^@mecha\//]

export default defineConfig([
  {
    entry: { index: 'src/index.ts' },
    outDir: 'lib',
    format: 'esm',
    platform: 'node',
    target: 'node20',
    deps: { neverBundle: dshExternal, alwaysBundle },
    dts: false,
    sourcemap: true,
    clean: true,
  },
  {
    entry: { client: 'src/client/index.ts' },
    outDir: 'lib',
    // dsh 的 web client 把所有插件的 client.js 合成一个 **classic-script 合并包**，每个模块
    // 必须自注册到 `window.__ModuleLoader__.load({id, factory:(require)=>{...}})`（见 dsh
    // packages/client/tsdown.client.ts 的 clientConfig）。普通 ESM（顶层 import/export）在合并包里
    // 是 SyntaxError → 整包崩。故这里精确复刻该格式：cjs + banner/intro/footer 包裹。
    format: 'cjs',
    platform: 'browser',
    target: 'es2022',
    // react / react/jsx-runtime 是 dsh 的 PLATFORM_MODULES（模块表提供）→ 外置，经注入的
    // require 解析，共享 dsh 的 React 实例（避免双 React）。@deepseek-ai/* 在 client 侧全是
    // 类型导入（已擦除），一并外置以防万一。资产包**外置不了**（见文件头）⇒ alwaysBundle。
    deps: {
      neverBundle: ['react', 'react/jsx-runtime', ...dshExternal],
      alwaysBundle,
    },
    dts: false,
    sourcemap: true,
    clean: false,   // 别清掉上一步的 host 产物
    outputOptions: {
      entryFileNames: 'client.js',
      banner: 'window.__ModuleLoader__.load({ id: "re0-mecha-dsh", factory: (require) => {',
      intro: 'var module = { exports: {} }; var exports = module.exports;',
      footer: 'return module.exports; } });',
    },
  },
])
