# 机甲协议（Mecha Protocol）· **DRAFT v0**

> **状态：DRAFT，未冻结。** 按纲领 §11 时序警告与开发计划 D-D：**别在 N=1 冻结协议**。
> 本 draft 从**仿真实例①（Re0）**蒸馏；待**硬件实例②**落地、出现第二个真实实现后再冻 v1。
> **定位**：定义"语义层合规系统长什么样"，让**通用驾驶舱**接任何合规项目（复用的是
> 驾驶舱与纪律机器，不是引擎）。本文只描述 Re0 已实现的那一套，不臆造未落地的部分。

---

## 0. 一句话

> 一个**合规项目** = 语义内核（typed registry + 唯一写 choke point）+ append-only
> 事件总线（唯一真值）+ 两条通道（写=MCP/localhost，读=订阅/可远程鉴权）+ 若干
> **只读投影**（驾驶舱）。协议规定这四件的**形状与不变量**，不规定实现语言/框架。

---

## 1. 实体描述符（Entity Descriptor）

每个"旋钮/参数"是一个带元数据的实体（Re0 落点：`src/core/param.py` 的 `Param`，
由 `src/mecha/registry.py` **收集**成表——收集而非登记，故无第二真源）。

| 字段 | 类型 | 含义 | 必需 |
|---|---|---|---|
| `name` | str | 全局唯一键（如 `field.B_gauss` / `solve.MAX_STEP` / `_M_START`） | ✓ |
| `family` | enum | `config`（登记册快照键）/ `solve`（运行级旋钮）/ `engine`（引擎常量） | ✓ |
| `home` | str | 定义它的模块（engine 族；收集器 AST 定位） | engine 必需 |
| `type` | str | `float`/`int`/`str`/`bool`/`list`/`dict`/`identity_key`/… | ✓ |
| `unit` | str\|null | 单位（`Gauss`/`V/cm`/`ns`/`MHz`/`rad`；null=无量纲） | ✓ |
| `lo`/`hi` | num\|null | 值域（null=不限）；`exclusive_lo` 标记严格大于 | 可选 |
| `enum` | list\|null | 离散合法值（如 `decay_index∈{0,1,2}`） | 可选 |
| `required` | bool | 是否必填（缺失 → 报错，不静默兜底） | 可选 |
| `log` | bool | 是否对数刻度（GUI 用） | 可选 |
| `tier` | enum | `run`/`internal`/`physics`/`check`——**`internal` 默认不对 AI/GUI 露面** | ✓ |
| `desc` | str | 一句人话说明 | ✓ |
| `default` | any | 默认值（唯一值来源） | ✓ |

**查询口**（渐进披露，省 context）：`list_params(prefix, family, tier,
include_internal, page)` → 分页表；`describe(name)` → 单条完整。**tier='internal'
默认隐藏**（D-B）。

---

## 2. 事件形状（Event Shape）——唯一真值日志

总线是 **append-only**（M4：AI 不能改写自己的历史）。每条事件：

```jsonc
{
  "seq": 12,                    // 单调递增；重启后从盘续（不重置）
  "ts": 1737000000.123,         // Unix 秒（时间戳形态全项目一致，I4）
  "actor": "ai" | "human" | "engine" | "sensor",   // 谁干的（M1 归因）
  "kind": "set" | "undo" | "run" | "error" | ...,  // 动作类别
  "target": "field.B_gauss",    // 作用对象
  "before": <any>,              // 变更前（大载荷 → 指针，见 §3）
  "after": <any>,               // 变更后
  "reason": "Zeeman 调谐",       // 为什么（M1；写入必填，无理由不进总线）
  "provenance": "session.set",  // 由哪条操作引起（可复现链）
  "reversibility": "undo" | "rerun" | "irreversible"  // 可逆性（分级自治依据）
}
```

**大载荷走指针（I3）**：数组/波形/结果**永不内联**，只给
`{"$pointer": true, "path": ..., "sha1": <40hex>, "shape": [...], "dtype": ...}`。
sha1 取自原始字节 ⇒ 内容被动过可检出。

**截断必告知（I4）**：任一字段超预算 → 落原文到 payloads，字段里只留
`{"$truncated": true, "head": <前512字符>, "kept": N, "total": M, "where": <path>,
"sha1": ...}`——**明确记下截了多少、原文在哪**，不能事后才知道。

---

## 3. 读回语义（Readback，M2 声称 vs 实测）

写回执必须带**由 choke point（而非发起方自己）产出的 readback**：

```jsonc
// set_param 回执
{ "ok": true, "key": "field.B_gauss",
  "resolved": 21.7,     // 归一化后**声称写入**的值
  "readback": 21.7,     // choke point 写完后**实测读回**的值（发起方伪造不了）
  "before": null, "seq": 12 }
```

监控器把 `resolved`（声称）与 `readback`（实测）**并排显示**——糊弄的空间
（"声称的"与"实际发生的"之间的缝）被显式化。

---

## 4. 自治级别（Autonomy Levels，§5）

**统一判据**：*这一步错了能不能撤销、撤销多贵* → 决定交给 AI 的自主度。

| 级 | 性质 | AI 权限 | Re0 落点 |
|---|---|---|---|
| **L1 只读** | 读状态/查诊断 | 自由 | `list_params`/`get_state`/`get_activity`/`estimate_cost` |
| **L2 可逆致动** | 软件限值内写入、可 undo | 允许，自动校验，全程记录 | `set_param`（经 `session.set`）/`undo` |
| **L3 不可逆/安全攸关** | 出光/加压/删档… | **只能提议**，人确认 + 硬件互锁 | 仿真档：`run_compute` 默认放行（可逆、可重跑）；硬件档：Phase 5 预留"提议→人确认→互锁"闸 |

**契约自由度锁死**（P0.4）：L2 写入必过 registry 校验（类型/单位/值域/枚举/必填），
非法 → 报错且**状态不变**（自愈回退，不静默）。**表达自由度拉满**：如何组合已验证
的操作不受限（CodeAct，Phase 4）。

---

## 5. 契约一致（Contract Consistency，I4）

- **同一把体积闸**：所有回程/事件字段走同一收缩逻辑（`gate_value`），截断口径一致。
- **单位/时间戳形态一致**：`unit` 是描述符字段；`ts` 统一 Unix 秒。
- **人机同一条代码路径（I5）**：AI 的 `run_compute` 与"人点运行"走**同一段**
  `runspec.build_run → evolve_run`（唯一分派点），杜绝会漂移的双路径。
- **契约外字段永远 None（I6）**：读某键的代码必须能指出写它的那一行；配置 schema
  与 `runspec` 实读键 **AST 交叉校验**，不覆盖即判据红。

---

## 6. 拓扑与通道（§10）

```
   [harness]  MCP客户端 ─(写:意图/回执，门控，localhost)─▶ ┌── 机甲本体 ──────────┐
                                                          │ 写: MCP → choke point │
   [驾驶舱]   只读订阅 ─(读:diff-since-seq，可远程鉴权)──▶ │   ↓                   │
                                                          │ registry + 事件总线   │← 唯一真值
                                                          │   ↑（所有 actor）     │
                                                          │ 人GUI/传感器/引擎     │
                                                          └──────────────────────┘
```

- **T1 两条通道一个真值源**：写经 choke point 落总线，读从同一总线出。
- **T2 读总线不读 MCP 镜像**：读通道承载**所有 actor**（AI/人/引擎/传感器）。
- **T5 黑匣子**：驾驶舱是**独立进程**，读**磁盘 journal**——项目死了仍显示死前全貌。
- **T6 读写端点分家**：写端点（MCP）**永不出本机**；只读订阅端点可（鉴权后）远程。

**读接口原语（R1）**：`diff-since-seq`（头号）、分页 live state、liveness/staleness、
attach/orientation packet、in-flight job status、read-only+idempotent。

---

## 7. 合规判据（Compliance，纲领 §8 清单）

一个项目自称"语义层合规"，须能逐条回答（Re0 的判据即其可执行形式）：

1. 每个 AI 面在 registry 里、可读/写/存档？ → `mecha_kernel_selfcheck` A/B
2. 经唯一 choke point、会进总线？ → `mecha_kernel_selfcheck` D/H（I1 无侧门 AST 守卫）
3. 有语义入口（不猜索引）、回执带可自检量？ → `mecha_mcp_selfcheck` C（resolved+readback）
4. 有读回面、读的键谁写（I6）？ → `mecha_kernel_selfcheck` C（schema ⊇ runspec 实读键）
5. 大载荷走指针+sha1、回程同一把闸（I3/I4）？ → `mecha_kernel_selfcheck` G + `mecha_mcp` H
6. 可逆性分级、L3 强制"提议+确认+互锁"？ → `mecha_backend_selfcheck`（Phase 5）
7. 失败报错还是静默降级（D5）？ → `mecha_kernel_selfcheck` E + `cockpit` C（降级告警）
8. 有自检 + 致死性证明、不永久 SKIP（D4）？ → 每个判据都带致死性子条件
9. 人与 AI 同一条代码路径（I5）？ → `ops.run` 复用 `runspec` 唯一分派

---

## 8. 冻结前的已知空白（N=1 的证据）

- **只有仿真实例**：可逆性=reversible、ground truth 在数学里、无硬实时、无部分可观测。
  硬件实例②会逼出：状态对账（模型 vs 传感器）、dry-run 数字孪生、L3 互锁的真实形状。
- **协议未提炼成语言无关的 schema 文件**（当前是 Python 约定 + 本 doc）。第二个实例
  落地时再决定是否出 JSON Schema / IDL。
- **鉴权只做了 token**：远程只读的最小闸；生产级（TLS/mTLS/OAuth）留待需要时。

> **冻结条件**：硬件实例②落地、出现"仿真与硬件对同一份 protocol 的不同 enforcement"
> 的真实分叉后，把本 draft 蒸馏为 v1（纲领 §11）。
