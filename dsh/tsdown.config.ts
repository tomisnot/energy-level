import { defineConfig } from 'tsdown'

/**
 * 两个产物（对应 package.json 的 "." 与 "./client" 导出）：
 *  · lib/index.mjs  —— HOST 半（node 平台）：⭐ **2026-10-10 起它只是空载体**
 *    （方案 A：工具面只有一条腿 = 框架的聚合端点；自愈桥连同 `@modelcontextprotocol/sdk` /
 *    `ws` 一起删了）⇒ 本产物**没有任何依赖要打进去**，只 @deepseek-ai/* peer 外置。
 *  · lib/client.js  —— CLIENT 半（browser 平台）：react 外置（dsh web client 提供，避免双 React）；
 *    ⭐ **资产包必须打进来**（`alwaysBundle`，见下）。
 * dts 关（类型走 src + types/shim；--patch 源加载本就直读 src）。
 *
 * ⭐ **`alwaysBundle` 是依赖化之后的生命线**（2026-10-01）：`@mecha/dsh-panel` 是 `file:`
 * 依赖 ⇒ 默认会被 tsdown 当"依赖"**外置**（bundle 里只剩一句 `import '@mecha/…'`）⇒
 * 打包产物**缺代码**，只在运行时才炸。实测：不加这一项 bundle 从 36 kB 掉到 10 kB。
 * ⇒ 必须显式 `deps.alwaysBundle` 把它拉回来（**现在只有 client 半需要它**：host 半不再
 *    import 资产；留着这一项是因为两半共用同一份 `deps` 常量，且它对 client 半是必需的）。
 *
 * client 半 = dsh 界面：框架那份面板插件（`panel/auto.ts`，按声明挂 N 个 tab）
 * （**地址不硬编码**：面板经资产包 `monitor-client.ts` 的 `fetchMonitorBase` 打**框架 host 插件**
 * 注册的同源只读路由取 base，再轮询其 `/history` + `/config`）。
 * ⚠「切换至 GUI」按钮**已删**（三层架构：切写权走控制端点、不动进程）。
 * ⚠ 2026-10-09：监控**地址路由**与**面板参数**都上提进框架（本仓不再有 `EL_PANEL`）。
 * ⚠ 2026-10-10：**自愈 MCP 桥整块删除**（方案 A）⇒ host 半退成空载体。
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
