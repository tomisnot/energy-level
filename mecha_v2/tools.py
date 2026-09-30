"""EL 的 AI 工具面：把能力包装成 dsh 原生形态的工具（Phase 4 判据的落地）。

命名守 Phase 4 守卫（动词_宾语、无视图名、无前缀）；每个 description 只含
任务概念（L4），不含框架/实现词汇；output.schema 显式声明 required 键。

工具经 Gate 写、经 Surface 跑——**没有一条路径绕过写入门**（洞1/洞2 的
Phase 5 侧证：AI 拿到的就是这些工具，物理演化只有 run_sim→Surface.run 一条）。

## `default` 声明是**必填项**（R5 三分法，2026-09-25 第 3 段）

框架的 null 垫片**只读声明里的 `default` 键**，**不读**闭包签名——写工具体常被
`def wrapped(**kwargs)` 包一层，签名侧恒为空集（EL 的 `hub.py` 注释记过 P1-8 同款）。
三分规则：

- 声明有 `default` 且**非 None** ⇒ 客户端送显式 `null` 时**丢键**，让闭包真默认生效；
- 声明有 `default` 且**为 None** ⇒ 保留 `null`（那是"清空"语义的唯一表达）；
- 声明**没有** `default` 键 ⇒ 视同真默认 `None` ⇒ 同样保留 `null`。

⇒ 凡闭包真默认**非 None** 的参数，**必须**在声明里写出该默认值（共 **10 个**：
`set_config.reason`、`set_drive.{pol,pulse,detuning_mhz,mode,atom,name,reason}`、
`read_waves.atom`、`reset_config.reason`）；真默认就是 `None` 的参数（8 个）
**一律不声明**——声明成 `null` 会落进第二格，语义不同。
判据 `checks/mcp_toolface_selfcheck.py` 守这条（**完备 + 唯一**），并断言
"显式 null ⇒ 到达工具体的是真默认"。

## Known Limitations and Deferred Work

- 工具函数是纯闭包工厂（build_tools 返回注册进宿主 ctx.tools 的形状）；真
  dsh defineTool 适配在部署接线时做，这里形状已对齐（ToolRegistry.register 可直挂）。
- run_sim 同步返回摘要（不含大数组 pops）；要完整演化数据走 artifact 指针，
  等 Jobs/Spill 的 dsh Provider 接线后再开（避免现在塞一个没人消费的大回执）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from mecha.authority import Authority, Mode
from mecha.errors import MechaError
from mecha.gate import Channel, Gate
from mecha.history import History
from mecha.providers.codeact import Action, CodeAct
from mecha.providers.jobs import JobRegistry, JobState
from mecha.surface import EntryKind, ExecutionContext, Surface
from mecha.tools import CURRENT_CALL_ID, ToolRegistry, define_tool

from .engine import ElEngine

# —— 任务概念词汇（description 只准用这些，禁框架内部词）——
# 每个工具的 execute 闭包捕获 (gate, surface, engine, channel)。

#: run_actions 声明的只读查询名（**单一来源**：description 由它生成、Surface
#: QUERY 注册由它驱动、判据按它逐个验真——审查②a：声明与能力面不许两套账）。
#: CodeAct 内置 op（estimate/summary/run/schema）不在此列，见 providers/codeact。
#: read_waves 既是顶层工具（实测报告 P1-7：常见查询该一步问到），也保留为
#: 查询项复用同一份 execute（一份实现两处用，不造第二真值）。
DOMAIN_QUERY_OPS = ("read_config", "query_params", "preview_run", "query_levels",
                    "read_waves")

#: set_drive 的 atom 参数 → 写回的配置键（②b，ADR 2026-09-23-EL驱动手感）。
_WAVE_KEY = {"single": "dynamic.waves_single",
             "A": "dynamic.waves_A", "B": "dynamic.waves_B"}


def build_tool_registry(engine: ElEngine, gate: Gate, surface: Surface,
                        authority: Authority, channel: Channel,
                        jobs: JobRegistry) -> ToolRegistry:
    """装配 EL 的工具面。channel 决定 actor（AI 侧恒为 'ai'，改不了）。

    jobs 必填（显式装配，不默认建）——**单一注册表实例**由宿主在装配点建好
    传进来（消 D-21：两处各建 runner 会互相误翻 interrupted）。
    """
    reg = ToolRegistry()
    schema = engine.schema()
    settable = list(schema["settable_keys"])

    def _coerce(rec: dict, value):
        """数值/整型键：把数字字符串收编回数值（使用者 AI 坑 1 的根治）。

        dsh 宿主对**未声明 type 的参数**（set_config 的 value 是任意 JSON 形态）
        会把标量数值统一转成 str ⇒ 引擎严格校验当场拒。这里在过门**之前**按该键
        的 schema 类型把"看起来是数字的字符串"转回 int/float——**只收编数字串**，
        真·字符串键（str 类型）绝不误伤。非数字串原样交给 validate（仍会拒且教学）。
        """
        tname = (rec or {}).get("type")
        if isinstance(value, str) and tname in ("int", "float"):
            s = value.strip()
            try:
                return int(s) if tname == "int" else float(s)
            except ValueError:
                return value          # 非纯数字串：交 validate 如实拒（不猜）
        return value

    def _set(key: str, value, reason: str = "ai tool"):
        # call_id 从请求作用域读（审查 2.7）：hub 的 MCP 层在 execute 前 set，
        # 把这次工具调用与 dsh 行为史钉在一起（总纲 §5.2②互引）。
        rec = dict(schema["config"]).get(key) or dict(schema["solve"]).get(key) or {}
        before = gate.snapshot.get(key)          # 写前旧值（回读对照用）
        ev = gate.set(channel, key, _coerce(rec, value), reason=reason,
                      call_id=CURRENT_CALL_ID.get())
        # 实测报告 P1-4（v1 有、v2 退化的"写一次就自证"）：回执自带回读——
        # resolved=引擎实际落地的值、readback=从快照读回的值、before=写前旧值。
        # AI 不用二次查询即可核对"我声称写的"与"域里真存的"是否一致。
        return {"ok": True, "key": key, "applied": True,
                "before": before, "resolved": ev.after,
                "readback": gate.snapshot.get(key)}

    # ---- set_config ----
    reg.register(define_tool(
        name="set_config",
        description=(
            "设置一个配置键的值。键必须是可写配置键（见 query_params 的清单）。"
            "改引擎内部常量会被拒。设置后立即可被 estimate_run/preview_run/run_sim 读到。"
            "reason 用一句中文写清这次改动的目的（会进操作留痕、供人监控）。"
        ),
        parameters={"key": {"type": "string", "description": "可写配置键，如 dynamic.t_max"},
                    "value": {"description": "新值（任意 JSON 形态），类型须匹配该键（见 query_params）"},
                    "reason": {"type": "string", "default": "ai tool",
                               "description": "一句话中文说明改动目的（进留痕，如'把演化时长拉到 200ns 看完整拉比振荡'）"}},
        output_schema={"type": "object", "required": ["ok", "key"]},
        execute=lambda key, value, reason="ai tool": _set(key, value, reason),
        banned_words=("gate", "choke", "读回面", "session", "channel"),
    ))

    # ---- read_config ----
    reg.register(define_tool(
        name="read_config",
        description="读取当前所有已设置的配置键与其值（快照）。用于确认改动是否生效。",
        parameters={},
        output_schema={"type": "object", "required": ["ok", "config"]},
        execute=lambda: {"ok": True, "config": gate.snapshot},
        banned_words=("gate", "snapshot 内部", "journal"),
    ))

    # ---- query_params ----
    reg.register(define_tool(
        name="query_params",
        description=(
            "列出所有可写配置键的元数据：类型、单位、值域、枚举、说明。"
            "写 set_config 前先查这里，避免类型/枚举错。"
        ),
        parameters={},
        output_schema={"type": "object", "required": ["ok", "keys"]},
        execute=lambda: {
            "ok": True,
            "keys": {k: {kk: vv for kk, vv in v.items()
                         if kk in ("type", "unit", "lo", "hi", "enum", "desc", "default")}
                     for k, v in dict(schema["config"]).items()
                     if k in settable}
            | {k: {kk: vv for kk, vv in v.items()
                   if kk in ("type", "unit", "lo", "hi", "desc")}
               for k, v in dict(schema["solve"]).items()},
        },
        banned_words=("registry", "scan", "AST"),
    ))

    # ---- estimate_run ----
    reg.register(define_tool(
        name="estimate_run",
        description=(
            "预估按当前配置跑一次要花多久、规模多大（不真正演化）。"
            "先看它决定要不要调小基组或时长。"
        ),
        parameters={},
        output_schema={"type": "object", "required": ["ok", "est_sec"]},
        execute=lambda: {"ok": True, **surface.estimate(gate.snapshot)},
        banned_words=("cost.py", "标定", "backend"),
    ))

    # ---- preview_run ----
    reg.register(define_tool(
        name="preview_run",
        description=(
            "预览当前配置会驱动哪些跃迁、基组够不够（不演化）。"
            "若报缺组，先补 groups.list 再 run_sim，否则结果失真。"
        ),
        parameters={},
        output_schema={"type": "object", "required": ["ok", "basis_enough"]},
        execute=lambda: {"ok": True, **engine.preview(gate.snapshot)},
        banned_words=("collect_transitions", "build_states"),
    ))

    # ---- run_sim ----
    reg.register(define_tool(
        name="run_sim",
        description=(
            "按当前配置真正跑一次演化，返回终态布居摘要（top 态 + 平均布居）。"
            "跑前建议先 estimate_run 看耗时、preview_run 看基组是否完整。"
        ),
        parameters={"cancel_after_sec": {"type": "number",
                    "description": "可选：超过此秒数自动请求取消"}},
        output_schema={"type": "object", "required": ["ok", "method"]},
        execute=lambda cancel_after_sec=None: _run_sim(
            surface, gate, cancel_after_sec),
        banned_words=("evolve_run", "progress_arr", "engine_kwargs"),
    ))

    # ---- query_levels ----
    reg.register(define_tool(
        name="query_levels",
        description=(
            "列出当前基组解析出的量子态（label + 身份键 + 能量 MHz）。"
            "设初态 dynamic.psi_single 用的身份键就从这里取，别猜索引。"
        ),
        parameters={},
        output_schema={"type": "object", "required": ["ok", "n_states", "states"]},
        execute=lambda: {"ok": True, **engine.list_levels(gate.snapshot)},
        banned_words=("base_sublevels", "make_levels"),
    ))

    # ---- submit_job / read_job / cancel_job（异步长任务，审查 §1.2①）----
    # 单一注册表实例由宿主传入（消 D-21）；read_job 合一"查单个/列全部"
    # ——枚举是投影不是工具（命名守卫禁 list_/get_ 前缀）。
    reg.register(define_tool(
        name="submit_job",
        description=(
            "把当前配置提交为后台演化任务，立即返回任务号。长任务（两体/多体、"
            "大基组）用它代替 run_sim——不必让调用方干等。用 read_job 轮询进度，"
            "done 后 result 即 run_sim 同款摘要。提交时刻的配置被固定，之后的"
            "改动不影响在飞任务。"
        ),
        parameters={},
        output_schema={"type": "object", "required": ["ok", "job_id"]},
        execute=lambda: _submit_job(jobs, surface, gate),
        banned_words=("evolve_run", "队列", "worker"),
    ))
    reg.register(define_tool(
        name="read_job",
        description=(
            "查后台任务状态：给 job_id 查单个（done 时含 result 摘要，failed "
            "时含 error）；不给 job_id 返回全部任务的状态清单。"
        ),
        parameters={"job_id": {"type": "string",
                    "description": "submit_job 返回的任务号；省略则列全部"}},
        output_schema={"type": "object", "required": ["ok"]},
        execute=lambda job_id=None: _read_job(jobs, job_id),
        banned_words=("registry", "轮询"),
    ))
    reg.register(define_tool(
        name="cancel_job",
        description=(
            "请求取消后台任务（在下一个计算步边界生效；已结束的取消无效果但"
            "不报错，状态如实返回）。任务号不存在会被拒。"
        ),
        parameters={"job_id": {"type": "string", "description": "要取消的任务号"}},
        output_schema={"type": "object", "required": ["ok", "job_id"]},
        execute=lambda job_id: _cancel_job(jobs, job_id),
        banned_words=("线程", "interrupt"),
    ))

    # ---- set_drive（②b：驱动手感，ADR 2026-09-23-EL驱动手感）----
    # 换算在 ElEngine.drive_wave（引擎线程，cache 亲和）；这里只"读现有波 →
    # 追加/替换 → 经 gate.set 写回单键"——门不旁路，与其余写工具同一条路。
    reg.register(define_tool(
        name="set_drive",
        description=(
            "给一对量子态之间加一个驱动场（比手搓波形清单省事）。src/tgt 用 "
            "query_levels 返回的身份键。强度二选一：rabi_mhz（拉比频率，物理"
            "自然语言，会自动换算成光强）或 intensity_mwcm2（直接给光强）。"
            "时长用 pulse='pi'/'pi2'（按拉比频率自动定时长）或 duration_ns 显式"
            "给。偏振 pol 取 π/σ+/σ-。默认追加到现有驱动清单，mode='replace' "
            "则整组替换。两体演化用 atom='A'/'B' 分别设两侧驱动。设完可 run_sim 演化。"
            "reason 用一句中文写清这条驱动的目的（进操作留痕）。"
        ),
        parameters={
            "src": {"description": "源态身份键（query_levels 的 key）"},
            "tgt": {"description": "目标态身份键（query_levels 的 key）"},
            # 可选数值参数**不声明 type**（坑 2 根治）：宿主对声明了 type:number
            # 的可选参数在缺省时 float(null) 崩（到达 execute 之前）。下游 drive_wave
            # 一律 float() 兜底，null/数字/数字串都安全，故不声明 type 更稳。
            "rabi_mhz": {"description": "拉比频率 MHz（与 intensity_mwcm2 二选一）"},
            "intensity_mwcm2": {"description": "光强 mW/cm²（与 rabi_mhz 二选一）"},
            # ↓ 真默认非 None ⇒ **必须**声明 default（框架的 null 垫片只读声明，
            #   不读闭包签名）：否则客户端送显式 null 会以 None 到达工具体。
            "pol": {"type": "string", "default": "π",
                    "description": "偏振：π / σ+ / σ-（默认 π）"},
            "pulse": {"type": "string", "default": "",
                      "description": "脉冲形状 pi / pi2（自动定时长）；省略则用 duration_ns"},
            "duration_ns": {"description": "脉冲时长 ns（pulse 省略时必填）"},
            "detuning_mhz": {"default": 0.0, "description": "相对共振的失谐 MHz（默认 0）"},
            "mode": {"type": "string", "default": "append",
                     "description": "append（默认追加）/ replace（整组替换）"},
            "atom": {"type": "string", "default": "single",
                     "description": "单体 single（默认）；两体 A / B"},
            "name": {"type": "string", "default": "drive",
                     "description": "波形名（默认 drive；多条波时用可辨的名字）"},
            "reason": {"type": "string", "default": "ai set_drive",
                       "description": "一句话中文说明这条驱动的目的（进留痕）"},
        },
        output_schema={"type": "object", "required": ["ok", "key", "wave"]},
        execute=lambda src, tgt, rabi_mhz=None, intensity_mwcm2=None,
        pol="π", pulse="", duration_ns=None, detuning_mhz=0.0,
        mode="append", atom="single", name="drive", reason="ai set_drive": _set_drive(
            engine, gate, channel, src, tgt, pol=pol, rabi_mhz=rabi_mhz,
            intensity_mwcm2=intensity_mwcm2, pulse=pulse,
            duration_ns=duration_ns, detuning_mhz=detuning_mhz, mode=mode,
            atom=atom, name=name, reason=reason),
        banned_words=("gate", "dipole", "dme", "serialize", "wave_list 结构"),
    ))

    # ---- read_waves（实测报告 P1-7：常见查询升顶层，一步问到）----
    # 命名守卫禁 get_ 前缀（架构不让步）⇒ 用 read_ 动词，与 read_config 同族。
    reg.register(define_tool(
        name="read_waves",
        description=(
            "看当前挂了哪些驱动波（逐波摘要：名字/频率 MHz/光强/偏振/包络形状/"
            "时长/起始时刻），不内联大数组。确认驱动配置、排查遗留驱动用它。"
            "两体演化 atom 取 A / B 分别看，单体默认 single。"
        ),
        parameters={"atom": {"type": "string", "default": "single",
                             "description": "single（默认）/ A / B"}},
        output_schema={"type": "object", "required": ["ok", "atom", "waves"]},
        execute=lambda atom="single": _read_waves(gate, atom),
        banned_words=("gate", "wave_list 结构", "snapshot"),
    ))

    # ---- reset_config（实测报告 P1-5：上手前先能"收拾台面"）----
    # 清空 = 对当前快照每个已写键写回 None（validate 已放行 None=清空回默认，
    # 与 GUI"留空=默认"同一语义、同一条门）——不另造第二条落地路径。
    reg.register(define_tool(
        name="reset_config",
        description=(
            "把当前所有已设置的配置键清空回默认（含驱动波形清单）。开始一轮新"
            "实验前用它收拾台面，不必逐键清。清空同样留痕可追溯；回执列出清了"
            "哪些键。reason 用一句中文说明目的。"
        ),
        parameters={"reason": {"type": "string", "default": "ai reset_config",
                               "description": "一句话中文说明清空目的（进留痕）"}},
        output_schema={"type": "object", "required": ["ok", "cleared", "n"]},
        execute=lambda reason="ai reset_config": _reset(
            gate, channel, reason),
        banned_words=("gate", "snapshot 内部", "journal"),
    ))

    # ---- run_actions（CodeAct 编排，审查 §1.2②：机制已建、消费方接上）----
    # 只读查询 op 的真身 = Surface 的 QUERY 注册项（审查②a：description 里
    # "或已注册的只读查询名"必须为真）。注册驱动自 DOMAIN_QUERY_OPS 单一来源，
    # 分派直接复用上面各工具已注册的 execute（一份实现两处用，不造第二真值）。
    for _qn in DOMAIN_QUERY_OPS:
        _tool = reg.get(_qn)
        surface.register(EntryKind.QUERY, _qn, _tool.execute,
                         called_by="framework:CodeAct.run_actions")

    reg.register(define_tool(
        name="run_actions",
        description=(
            "一次提交一串动作按序执行（省往返）。每步是 {op, args}：op 可为 "
            "estimate（探规模）、summary（读引擎概况）、run（跑一次演化）、"
            "schema，或只读查询名（" + " / ".join(DOMAIN_QUERY_OPS) + "）。"
            "estimate/run 也接受顶层工具名 estimate_run/run_sim 作别名。"
            "run/estimate 的配置三种给法：args.spec 整表替换；args.spec_patch "
            "只改列出的键、其余沿用当前配置（扫描首选，顶层 spec_patch 同义）；"
            "都不给=当前配置。pick=只回关注量（键路径列表，如 "
            "[\"final_top_pop.60P_{3/2} m_J=-0.5\"]），投影每步 output 控体积。"
            "任一步失败即停在该步并如实返回失败原因，之前的步不回滚。"
            "每步都留可追溯记录。"
        ),
        parameters={"actions": {"type": "array",
                    "description": "动作列表，每项 {\"op\": 名称, \"args\": 对象}"},
                    "spec": {"type": "object",
                             "description": "run/estimate 缺省 spec 时用的整表配置"},
                    "spec_patch": {"type": "object",
                                   "description": "run/estimate 缺省 spec 时在现有配置上覆写这几键（扫描只写变化量）"},
                    "pick": {"type": "array",
                             "description": "只回这些键路径（点分层；整键优先匹配），投影每步 output"}},
        output_schema={"type": "object", "required": ["ok", "steps", "stopped_at"]},
        execute=lambda actions, spec=None, spec_patch=None, pick=None: _run_actions(
            surface, gate, channel, actions, spec, spec_patch, pick),
        banned_words=("eval", "解释器", "dispatch"),
    ))

    return reg


def _run_actions(surface: Surface, gate: Gate,
                 channel: Channel, actions, spec=None, spec_patch=None,
                 pick=None):
    """把一串结构化动作交给 CodeAct 编排（逐子调用记日志，可归因）。

    **审计落一份进程内临时 History，不混进账**（总纲 §5.2②：账
    与操作者行为史是两份资产）。若把 `codeact.N` 写进域史，Gate 重启折叠快照
    会把这些审计事件当域状态吃进去 ⇒ read_config 出噪声、存档 load 被 validate
    当场拒。操作者行为史归宿主（dsh 按 call_id 记），这里逐步随回执如实返回。

    实测第二轮 §4 的三处顺手感（都在编排入口做适配，CodeAct 内核不动）：
    - **spec_patch**（4-1）：run/estimate 的配置支持"在现有配置上只改几个键"
      （顶层或每步 args 均可，步级优先）——扫描不必每点重抄全量 spec。
    - **op 别名**（4-2）：`estimate_run`/`run_sim` 直接映射到内置 `estimate`/`run`，
      工具名与 op 名不再两套账。
    - **pick**（4-4）：只回关注量（键路径投影每步 output），扫描 50 点不撑爆上下文。
    """
    if not isinstance(actions, (list, tuple)):
        raise MechaError("actions 必须是列表", kind="bad_action",
                         hint='形如 [{"op": "estimate"}, {"op": "run"}]')
    if pick is not None and not isinstance(pick, (list, tuple)):
        raise MechaError("pick 必须是键路径列表", kind="bad_action",
                         hint='如 ["final_top_pop", "method"]；一层键用点分层')
    acts = []
    for a in actions:
        if not isinstance(a, Mapping) or "op" not in a:
            raise MechaError(f"动作项 {a!r} 缺 op", kind="bad_action",
                             hint='每项须是 {"op": 名称, "args": 对象}')
        op = str(a["op"])
        if op in _OP_ALIAS:
            op = _OP_ALIAS[op]
        args = dict(a.get("args") or {})
        if op in ("run", "estimate"):
            if "spec" not in args:
                base = dict(spec) if spec else dict(gate.snapshot)
                step_patch = args.pop("spec_patch", None)
                if step_patch is None and spec is None and spec_patch:
                    step_patch = spec_patch        # 顶层 patch（步没给才生效）
                if step_patch is not None:
                    if not isinstance(step_patch, Mapping):
                        raise MechaError("spec_patch 必须是对象", kind="bad_action",
                                         hint='形如 {"dynamic.t_max": 500}')
                    base.update(step_patch)
                args["spec"] = base
            else:
                args.pop("spec_patch", None)       # 整表优先，patch 不掺和
        acts.append(Action(op=op, args=args))
    actor = channel.actor                      # 恒为 'ai'（通道钉死，改不了）
    audit = History()                          # 临时审计史（不进域史，见 docstring）
    codeact = CodeAct(surface, audit, lambda: actor)
    res = codeact.run(acts, call_id=CURRENT_CALL_ID.get())
    # 实测报告 P1-6：结果要**带名字、可自读**——每步回执 {index, op, ok,
    # detail, output}（output 直接挂进步里，不再靠 index 与位置数组对齐）；
    # 成功步 detail 给短摘要（回执首个标量键值），失败步 detail 是教学原因。
    steps = []
    for s in res.steps:
        out = res.outputs[s.index] if s.index < len(res.outputs) else None
        if pick and s.ok:
            out = _pick(out, pick)
        detail = s.detail or _step_digest(out)
        steps.append({"index": s.index, "op": s.op, "ok": s.ok,
                      "detail": detail, "output": out})
    return {"ok": res.stopped_at == -1,
            "steps": steps,
            "stopped_at": res.stopped_at}


#: op 别名（实测第二轮 4-2）：顶层工具名直接当 op 用，消"每次试一次"。
_OP_ALIAS = {"estimate_run": "estimate", "run_sim": "run"}


def _pick(out, paths):
    """把 output 投影成只含关注键的 dict（实测第二轮 4-4）。

    每个路径**整键优先**（JSON 键可以含点，如 `final_top_pop.60P_{3/2} m_J=-0.5`
    本身就是一个键），匹配不到再按点分层下钻。缺失键**如实列进 `_missing`**
    （绝不静默给 null——写错键名当场看得见）。
    """
    if not isinstance(out, Mapping):
        return out
    picked, missing = {}, []
    for path in paths:
        p = str(path)
        if p in out:
            picked[p] = out[p]
            continue
        node = out
        for part in p.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                missing.append(p)
                break
        else:
            picked[p] = node
    result = dict(picked)
    if missing:
        result["_missing"] = missing
    return result


def _step_digest(out) -> str:
    """成功步的短摘要：回执 dict 的首个标量键值（无则 ok）。"""
    if isinstance(out, Mapping):
        for k, v in out.items():
            if isinstance(v, (str, int, float, bool)) and k != "ok":
                return f"{k}={v}"
        return "ok" if out.get("ok") else ""
    return "ok" if out is not None else ""


def _submit_job(jobs: JobRegistry, surface: Surface, gate: Gate):
    snap = dict(gate.snapshot)     # 提交时刻固定配置（在飞任务不受后续改动影响）

    def _work(ctx: ExecutionContext):
        return surface.run(snap, context=ctx)

    job = jobs.submit(_work)
    return {"ok": True, "job_id": job.id, "state": job.state.value}


def _read_job(jobs: JobRegistry, job_id):
    if job_id:
        job = jobs.get(job_id)     # 未知任务号抛 unknown_job（fail loud）
        out = {"ok": True, **job.status()}
        if job.state is JobState.DONE:
            out["result"] = job.result
        elif job.state is JobState.FAILED:
            out["error"] = job.error
        return out
    return {"ok": True, "jobs": jobs.list()}


def _cancel_job(jobs: JobRegistry, job_id):
    jobs.cancel(job_id)            # 未知任务号抛 unknown_job
    return {"ok": True, "job_id": job_id,
            "state": jobs.get(job_id).state.value}


def _run_sim(surface: Surface, gate: Gate, cancel_after_sec):
    ctx = ExecutionContext()
    timer = None
    if cancel_after_sec:
        import threading
        timer = threading.Timer(float(cancel_after_sec), ctx.request_cancel)
        timer.start()
    try:
        result = surface.run(gate.snapshot, context=ctx)
    finally:
        if timer is not None:
            timer.cancel()
    return result


def _set_drive(engine: ElEngine, gate: Gate, channel: Channel, src, tgt, **kw):
    """set_drive：换算在引擎线程（cache 亲和），写回经 gate（门不旁路）。

    atom 决定写哪个键；append 读现有波再追加、replace 整组替换。换算失败
    （禁戒跃迁 / 身份键不在基组 / 强度时长没给全）抛 InvalidSpec，工具层
    如实回错（L4 可教学）。
    """
    atom = kw.pop("atom", "single")
    mode = kw.pop("mode", "append")
    reason = kw.pop("reason", "ai set_drive")   # 不进 drive_wave（它只吃物理参数）
    key = _WAVE_KEY.get(atom)
    if key is None:
        raise MechaError(f"atom={atom!r} 非法", kind="bad_drive",
                         hint="atom 取 single / A / B")
    wave = engine.drive_wave(gate.snapshot, src=src, tgt=tgt, **kw)
    cur = list(gate.snapshot.get(key) or [])
    waves = ([dict(w) for w in cur] if mode == "append" else []) + [wave]
    gate.set(channel, key, waves, reason=reason,
             call_id=CURRENT_CALL_ID.get())
    out = {"ok": True, "key": key, "wave": wave, "n_waves": len(waves)}
    # 实测第二轮 4-3：t_max 不跟随脉冲时长是**实验设计陷阱**（终态布居读的是
    # "脉冲结束后又自由演化了一段"或"脉冲被掐断"）。回执如实对账：波结束时刻
    # =offset+平台长，与当前 t_max 不等就给一句提示（不拦止——自由演化/截断
    # 都可能是故意设计，提示让人/AI 自己确认）。
    env = wave.get("env")
    if isinstance(env, (list, tuple)) and len(env) >= 5:
        pulse_end = float(wave.get("offset_ns") or 0.0) + float(env[4])
        out["pulse_end_ns"] = round(pulse_end, 6)
        t_max = gate.snapshot.get("dynamic.t_max")
        try:
            tm = float(t_max)
        except (TypeError, ValueError):
            tm = None
        if tm is not None and abs(tm - pulse_end) > 1e-9:
            if tm > pulse_end:
                out["t_max_note"] = (
                    f"当前 dynamic.t_max={tm} ns 晚于本波结束 {out['pulse_end_ns']} ns"
                    "——run_sim 终态布居含脉冲后的自由演化；要读脉冲结束瞬间的"
                    "布居，把 t_max 调到≈波结束")
            else:
                out["t_max_note"] = (
                    f"当前 dynamic.t_max={tm} ns 早于本波结束 {out['pulse_end_ns']} ns"
                    "——演化窗口内脉冲被截断（只跑到平台中途）")
    return out


def _reset(gate: Gate, channel: Channel, reason: str):
    """reset_config：把当前快照里每个已写键写回 None（=清空回默认）。

    走与单键写同一条门（validate 放行 None 是既有语义），逐键留痕可追溯；
    回执列出清了哪些键——"台面收没收拾干净"一眼可核对。
    """
    keys = sorted(gate.snapshot)
    call_id = CURRENT_CALL_ID.get()
    for k in keys:
        gate.set(channel, k, None, reason=reason, call_id=call_id)
    return {"ok": True, "cleared": keys, "n": len(keys)}


def _read_waves(gate: Gate, atom: str = "single"):
    """read_waves：读某原子侧的驱动波摘要（不内联大数组）。

    纯读快照里的 wave_list，逐波回 name/频率/光强/偏振/包络形状/时长。要完整
    演化数据走 run_sim 回执，要原始波形对象是 GUI 的事——查询面只给可概览的摘要。
    """
    key = _WAVE_KEY.get(atom)
    if key is None:
        raise MechaError(f"atom={atom!r} 非法", kind="bad_drive",
                         hint="atom 取 single / A / B")
    out = []
    for w in (gate.snapshot.get(key) or []):
        env = w.get("env")
        shape = {"kind": env[0], "duration_ns": (env[4] if len(env) > 4 else None)} \
            if isinstance(env, (list, tuple)) and env else {"kind": w.get("type")}
        out.append({"name": w.get("name"),
                    "freq_mhz": w.get("freq_mhz"),
                    "intensity_mwcm2": w.get("intensity", w.get("intensity_mwcm2")),
                    "pol": w.get("pol"), "offset_ns": w.get("offset_ns"),
                    "envelope": shape})
    return {"ok": True, "atom": atom, "key": key, "waves": out}
