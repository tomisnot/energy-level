/**
 * re0-mecha dsh 插件 —— **CLIENT 入口**（浏览器侧 Cordis 插件）。
 *
 * dsh 界面两件事（⚠ 原有「切换至 GUI」按钮**已按三层架构删除**）：
 *  1. 右栏 tab body（`MonitorTabBody`）→ 轮询权威**只读监控端点**的 `/history`（写事件，
 *     含 `mode`）+ `/config`（配置态树），显示"当前写权模式 + AI/人最近改了什么"。
 *  2. ⭐ 会话头部那个「◈监控」按钮**不再由本仓注册**——2026-10 阶段① 起**归框架**
 *     （`mountClientPanel` 注册**唯一一个**，语义 = **开右栏**）。⚠ 原先三家各注册一个
 *     ⇒ 3 个项目 = 3 个按钮；而用户要的是"唤醒**所有**监控面板"。
 *  ⚠ **曾经有第三件：会话头部「切换至 GUI」按钮**（经 `/re0-gui` command 让 host 写
 *     `.mode-request`、看门人接手做进程交接）。**三层架构（2026-10-08）删了整条链**：
 *     切写权现在走**控制端点**（`.control-port`）、**不动进程**、不留意图文件
 *     ⇒ 按钮与 command 都不该存在（留一个"按了没人接"的按钮正是本工程要禁的形态）。
 *
 * **地址不再硬编码**（2026-09-25 第 2 段）：监控端点是**动态端口**，面板经共享资产的
 * `fetchMonitorBase` 去 host 半的**同源只读路由**取 `{base}`（host 读权威写的端口文件）。
 * 每一轮轮询都现取 ⇒ 权威重启换端口后下一拍即落在新端口，不需要重启 dsh、
 * 更不会"面板空白但没人报错"（空数据由 `empty` 档显式表达，不伪装成成功）。
 *
 * **路由与端口文件名不再是本仓常量**（2026-10-01 依赖化）：两半都把自己那份
 * `EL_PANEL` 交出去（host 半经骨架、client 半经 `mountClientPanel`）——资产
 * `@mecha/dsh-panel` 经 junction 只有一份代码，项目**不能再改资产文件**；
 * 未交出去时资产会抛（刻意的响亮失败）。
 *
 * ⚠ **为什么标题/取数不再落到别人那份**：原先写的是
 * `title: () => panelConfig().title` —— 那是**渲染时求值**的 thunk ⇒ 三家装进同一个 dsh 时
 * 三个 tab 全读"最后注册那一份"。现在参数**跟着这一次注册走**（helper 内部闭包捕获 +
 * 正文以显式 prop 收到自己那份）。
 *
 * ⚠ client 半经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改本目录源码后须
 * `npm run bundle` 并重启 dsh 才生效。
 */
import type { Context } from '@deepseek-ai/cordis'
import { mountClientPanel } from '@mecha/dsh-panel/panel/mount-client-plugin.ts'
import { MonitorTabBody } from '@mecha/dsh-panel/panel/MonitorTabBody.tsx'
import { EL_PANEL, PLUGIN } from '../el-params.ts'

/** Cordis 插件名（与 host 半一致）。**取自参数家**（`PLUGIN.name` 是唯一来源）。 */
export const name = PLUGIN.name

/** slots（tab body）+ layout（开右栏）+ 右栏 tab 服务。
 *  ⚠ 原交权按钮还另需 `remote` / `remote.commands`；那条链已删 ⇒ 不再声明（R17）。 */
export const inject = ['slots', 'layout', 'sidebarRight', 'sidebarRightTabs']

/** ⭐ §7.3.3 定案：逐服务一个 `kind` / 一个 `id`（契约拒绝重复 id 与同 tier 同 kind 二次注册）。 */
const TAB_KIND = 'mecha:EL'
const TAB_ID = 'mecha-panel:EL'

export async function apply(ctx: Context): Promise<void> {
  await ctx.effect(() => {
    // ⚠ 这里**曾经**注册「切换至 GUI」按钮（发 /re0-gui command，host 写 .mode-request）。
    // 三层架构删了整条链（切写权走控制端点、不动进程）⇒ 不留按钮、也不留 command。

    // ⭐ 一次调用管全部：参数**跟着这次注册走**；头部按钮由框架注册**唯一一个**。
    // 面板**自己取数**（原生形态）：取址/取数/归约/渲染全在资产里 ⇒ 不传 `bodyProps`
    // （本仓没有附加页；原先那个 `resolveMonitorBase` 缝已被兜住，留着就是死缝）。
    const dispose = mountClientPanel(ctx as never, {
      panelConfig: EL_PANEL,
      kind: TAB_KIND,
      id: TAB_ID,
      body: MonitorTabBody,
      titlePrefix: '◈ 监控 · ',
    })
    return () => { dispose() }
  }, 're0-mecha: client monitor UI')
}
