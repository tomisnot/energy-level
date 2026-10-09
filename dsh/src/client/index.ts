/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * ## ⭐ 2026-10 阶段②：本文件 = **一行 bootstrap，别的都不写**
 *
 * ```ts
 * export { apply, name, inject } from '@mecha/dsh-panel/panel/auto.ts'   // ← 框架的代码
 * ```
 *
 * 那一行接上**框架自己的 client 插件**：它读框架起 dsh 时按**声明**
 * （`energy-level/mecha_site.py` 的 `Service.monitor`）生成的那份参数，
 * **按服务挂 N 个 tab**（三家各挂自己那一份），头部按钮**恒为 1 个**
 * （`mountClientPanel` 的**跨 bundle 全局登记**）。
 *
 * ⚠ 为什么**不是**裸 `import`：cordis 只认**模块自己的** `name`/`inject`/`apply` ——
 * 裸 import 只执行模块体、`apply` **不会被调**；且资产 `sideEffects:false` 还会把它摇掉。
 * ⚠ 为什么用**显式子路径**：三家的"浏览器安全闭包"判据按**真实文件**解析 import
 * （与三家既有的 `…/panel/MonitorTabBody.tsx` 写法一致）。
 *
 * ## 本仓这一半**没有**自定义 UI ⇒ 一行就够
 *
 * EL 没有附加页、也没有自己的会话头部按钮（⭐ 原先那个「切换至 GUI」按钮随三层架构
 * 整条链**已删**：切写权现在走控制端点、不动进程、不留意图文件）⇒ 这一行之后没有别的事。
 *
 * ## 已删掉的东西（别往回加）
 *
 * - 会话头部「◈ 监控」按钮：阶段① 起归框架（全生态**唯一一个**，语义 = 开右栏）；
 * - `sidebarRightTabs` / `sidebar.right.pane.tab` 的手写注册：同上，且
 *   `title: () => panelConfig().title` 是**渲染时求值**的 thunk ⇒ 三家装进同一个 dsh 时
 *   三个 tab 的标题与取数路由**全落到"最后注册那一份"**（阶段① 修掉的真缺陷）。
 *
 * ## 面板参数的家在哪
 *
 * ⭐ **在声明**（`mecha_site.py` 的 `monitor=MonitorSpec(...)`）—— 框架按它生成那份参数、
 * 由**框架自己的 patch** 带上的 host 插件经同源只读路由交给浏览器。
 * ⭐ **2026-10-09 起连地址路由也归框架**：本仓**不再有**面板参数常量、也**不再注册**那条路由
 * （`mountHostPlugin` 不传 `panelConfig`）⇒ 参数只剩**一个家**，没有"两处必须同值"这回事。
 * ⚠ 判据 `test/re0-panel.test.ts` 钉着"不许再加回来"。
 *
 * ⚠ client 半经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改本目录源码后须
 * `npm run bundle` 并重启 dsh 才生效。
 */
export { apply, name, inject } from '@mecha/dsh-panel/panel/auto.ts'
