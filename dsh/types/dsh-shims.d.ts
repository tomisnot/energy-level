/**
 * 本地**最小类型 shim** —— 声明本插件用到的 dsh / Cordis API 表面。
 *
 * 为什么用 shim 而非真包：这些 `@deepseek-ai/*` 是 dsh **宿主运行时提供**的 peer
 * （cordis/schemastery 是 vendored、未发布 npm），out-of-tree 插件本地装不到、也不该把某台
 * 机器上 dsh 检出路径的绝对地址写进交付 tsconfig。shim 只声明"我们实际用到的那一小片"，
 * 形状取自 dsh 0.1.5-rc.2 真源码（cordis-tutorial 07、mcp-client/src/tools.ts、slots.zh.md、
 * experimental/inspector、experimental/client-ui-agent-team），使本地 `tsc`/IDE 可校验。
 *
 * **运行时由 dsh 提供真实现**；shim 若与真 API 漂移，由 `dsh web --patch` 活体冒烟兜底
 * （见 README「验证」）。保持最小：只声明用到的成员，用到新的再补，避免与真类型无谓分歧。
 */

declare module '@deepseek-ai/dsh-tools' {
  /** 一次工具执行的不可变身份 + 操作字段（我们只用 signal）。 */
  export interface ToolExecution {
    readonly signal: AbortSignal
    readonly token: unknown
    readonly name: string
    readonly callId: string
    readonly arguments: unknown
    readonly agent?: unknown
  }
  export interface ToolExecutionResult {
    content: unknown
    isError?: boolean
    value?: unknown
    meta?: unknown
  }
  export interface ToolOutputDef {
    schema: Record<string, unknown>
    render(args: unknown, value: unknown): Array<Record<string, unknown>>
    presentationMeta?(args: unknown, value: unknown): unknown
  }
  /** ctx.tools.register 直接吃的原始定义（MCP 来源工具即以此形态注册）。 */
  export interface ToolDefinition {
    name: string
    description?: string
    parameters: Record<string, unknown>
    output?: ToolOutputDef
    execute(args: unknown, exec: ToolExecution): Promise<unknown>
    presentCall?(args: unknown): unknown
    presentResult?(args: unknown, result: ToolExecutionResult): unknown
    finalizeContent?(exec: ToolExecution, result: ToolExecutionResult): unknown
  }
  export interface ToolRuntime {
    /** 注册一个工具，返回其注销 disposer（随插件 fiber 释放）。 */
    register(def: ToolDefinition): () => void
  }
}

declare module '@deepseek-ai/cordis' {
  import type { ToolRuntime } from '@deepseek-ai/dsh-tools'
  import type { SlotsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
  export interface Logger {
    info(...args: unknown[]): void
    warn(...args: unknown[]): void
    error(...args: unknown[]): void
  }
  /** Cordis 上下文（服务容器）。只声明本插件用到的成员。 */
  export interface Context {
    readonly tools: ToolRuntime
    readonly slots: SlotsRuntime
    readonly logger: Logger
    /** 注册可逆副作用：setup 返回 disposer；await 之。 */
    effect(setup: () => unknown | Promise<unknown>, label?: string): Promise<void>
    /** 监听类型化事件，返回取消监听的 disposer。 */
    on(event: string, listener: (...args: any[]) => unknown): () => void
    /** 在 ctx 上提供一个服务，返回 disposer。 */
    provide(key: string, value: unknown): () => void
  }
  export class Service {
    constructor(ctx: Context, key: string)
  }
}

declare module '@deepseek-ai/schemastery' {
  export interface ZSchema<T = unknown> {
    default(value: T): ZSchema<T>
    optional(): ZSchema<T | undefined>
    min(n: number): ZSchema<T>
    max(n: number): ZSchema<T>
  }
  export interface ZBuilder {
    object<S extends Record<string, ZSchema<any>>>(
      shape: S,
    ): ZSchema<{ [K in keyof S]: S[K] extends ZSchema<infer T> ? T : never }>
    string(): ZSchema<string>
    boolean(): ZSchema<boolean>
    number(): ZSchema<number>
    natural(): ZSchema<number>
    array<T>(item: ZSchema<T>): ZSchema<T[]>
    const<T extends string>(value: T): ZSchema<T>
  }
  const z: ZBuilder
  export default z
}

declare module '@deepseek-ai/dsh-host-webserver' {
  /** host 向 web client 的 index.html 注入的全局（client 插件读 globalThis[name]）。 */
  export interface IndexInjection {
    kind: string
    name: string
    value: unknown
  }
}

declare module '@deepseek-ai/dsh-client-ui-slots' {
  /** 组件在某 slot 收到的 props（框架 hook + 注入值）。本插件不用 props，故宽松。 */
  export type PropsRuntime<K extends string = string> = Record<string, any>
  export interface SlotRegisterOptions {
    name: string
    id: string
    order?: number
    locale?: string
    inject?: (...args: any[]) => unknown
  }
  export interface SlotsRuntime {
    /** 向另一个包拥有的 slot 贡献内容；cb 在每段声明生命周期内运行。返回 disposer。 */
    inject(key: string, cb: () => unknown): () => void
    /** 注册一个组件到 slot；返回 disposer。 */
    register(opts: SlotRegisterOptions, component: unknown): () => void
  }
}
