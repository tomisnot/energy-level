# -*- coding: utf-8 -*-
"""MCP 工具面 **`default` 声明守卫**（R5 三分法：完备 + 唯一 + 行为）。

## 守什么（为什么必须有）

框架的 **null 垫片**只读**声明里的 `default` 键**，不读闭包签名（写工具体常被
`**kwargs` 包一层，签名侧恒为空集）。三分规则：

| 声明 | 显式 `null` 到达时的行为 |
|---|---|
| 有 `default` 且非 `None` | **丢键** ⇒ 工具本体用闭包真默认 |
| 有 `default` 且为 `None` | 保留 `null`（"清空"语义的唯一表达） |
| **没有** `default` 键 | 视同真默认 `None` ⇒ 保留 `null` |

⇒ 凡闭包真默认**非 None** 的参数，声明里**必须**写出该值；漏一个、或值写错，
**模型侧行为变差而不报错**（`pol=null` 静默以 `None` 到达工具体——症状是"说明书说
默认 π 但必须显式传"）。EL 在 2026-09-25 第 3 段补齐了 10 个声明，**本判据是它们唯一
的守卫**（此前没有任何东西会在漏声明时变红）。

## 三组断言

- **P 分区（纯声明面，期望值取自闭包）**：逐工具
  P1 `defaulted`（框架经 `tool_input_schema` 算的）== 声明里「非 None 的 `default`」集；
  P2 **完备**：闭包真默认非 None 的参数必须在其中，且**值逐位相等**；
  P3 **不越界**：声明了非 None `default` 的参数，闭包真默认也必须非 None；
  P4 **唯一表达**：不得出现 `default: null` 声明（与"不声明"同义 ⇒ 只留一种写法）；
  P5 声明参数键集 == 闭包参数键集（防声明与实现漂移）。
- **B 行为（期望值取自闭包，**不取自被检声明**——否则漏声明时 B 组会静默放过）**：
  B1a **宿主边界**：每个真默认非 None 的可选参数，传显式 `None` 后必须**被丢掉**
       （spy 记录里该键**不存在**）；真默认为 `None` 的参数必须**保留为 `None`**。
  B1b **工具本体**：拿 EL 自己的工具端到端看效果——`set_config(..., reason=None)`
       落进 History 的 `reason` 必须是真默认 `"ai tool"`；`reset_config(reason=None)`
       同理；`read_waves(atom=None)` 回执的 `atom` 必须是 `"single"`。
  B2 **另一格**：无 `default` 声明且真默认为 None 的参数传显式 `null` ⇒ 到达本体仍是
       `None`（`set_config(key, value=None)` 的清空语义）。
  观测技法：从 SDK 注册表取**已注册 handler 本体**直调（框架 docstring 认可：进程内
  不许直调 `MCPServer.call_tool`），`toolhost` 换成**记录型宿主**。
- **C 致死性**：把故意做坏的声明（漏声明 / 值不符 / `default: null` / 越界声明）喂给
  同一批探测器，必须逐条报出；干净声明必须零问题。

## ⚠ B 组的两条写法纪律（R9：**期望值必须取自被检对象之外**）

**⛔ 别把这两处"改回"更自然但空洞的写法**——本判据第一版正是那样，且**双向空洞**：

1. **期望值取自闭包签名，不取自被检声明。**
   若从声明推期望（"声明说 `pol` 默认 π，那就断言收到 π"），一旦有人**漏声明**，
   `defaulted` 随之变小 ⇒ B 组**自动跳过该参数** ⇒ 出现 **P 红而 B 绿**的假安全。
   闭包签名是**独立信源**（工具本体的真默认），声明是被检对象——两者分开，漏声明才
   会同时被 P 与 B 抓住（实测：删掉 `set_drive.pol` 的声明 ⇒ P 与 B1a 同时红）。
2. **边界观测量是"键不存在"，不是"值等于默认"。**
   垫片的真实行为是**把 `null` 键丢掉**（`{k: v for k, v in kwargs.items() if not
   (v is None and k in defaulted)}`），随后由 Python 的 `**args` 解包把闭包真默认补上。
   所以在**宿主边界**上，正确现象是"该键**不在** args 里"，而不是"args[键] == 真默认"。
   写成后者会**在真实现上恒被跳过**（键根本不存在）、在坏实现上也不报——同样是空洞。
   真默认"确实生效"这一层，另由 **B1b 在工具本体上端到端看效果**（History 的
   `reason`、回执的 `atom`）来证明，而不是靠边界值推断。

## 非退化自证（R8）

打印并断言：工具数 > 0、参数总数 > 0、"需声明"参数数 > 0、另一格 > 0、
**spy 真收到调用**、**History 真落了事件**。逐字节/零失败在退化输入下同样成立，
所以必须自证"数据真的到了"。

隔离：纯内存装配（`_SchemaOnly` 引擎 stub + `accept_all` 校验器 + AI 模式），
不建 ARC cache、不碰数据根、不起任何端点。

用法：python checks\\mcp_defaults_selfcheck.py     退出码 0 全过 / 1 有问题。
"""
import copy
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from src.core import protocol as P                           # noqa: E402
from mecha.authority import Authority, Mode                  # noqa: E402
from mecha.gate import Channel, Gate, accept_all             # noqa: E402
from mecha.history import History                            # noqa: E402
from mecha.providers.jobs import JobRegistry                 # noqa: E402
from mecha.providers.mcp import build_mcp_server, tool_input_schema  # noqa: E402
from mecha.surface import Surface                            # noqa: E402
from mecha.toolhost import LocalToolHost                     # noqa: E402

from mecha_v2.tools import build_tool_registry               # noqa: E402

C = Checker('mcp_defaults')


# ---------------------------------------------------------------- 装配（纯内存）
class _SchemaOnly:
    """只回 `schema()` 的哑引擎：够 `build_tool_registry` 用，不建 ARC cache。"""

    def schema(self):
        return {"run_output_required": ["ok", "method"],
                "estimate_output_required": ["est_sec"],
                "config": P.config_schema(), "solve": P.solve_schema(),
                "engine": P.engine_params(), "settable_keys": P.settable_keys()}


def _stack():
    """→ (registry, history)。AI 模式 + accept_all：写侧工具真能落史（可观测）。"""
    eng = _SchemaOnly()
    hist = History()
    auth = Authority(mode=Mode.AI)
    gate = Gate(auth, hist, validate=accept_all)
    reg = build_tool_registry(eng, gate, Surface(eng), auth, Channel("ai"),
                              JobRegistry())
    return reg, hist


class _SpyHost:
    """记录型工具宿主：记下 (name, args) 后委托 `LocalToolHost`。

    观测量刻意选在**宿主边界**——null 垫片发生在它之前（handler 内），所以这里看到的
    正是"交给工具本体"的入参；工具本体随后跑失败也不影响已记录的 args。
    """

    def __init__(self, registry):
        self._inner = LocalToolHost(registry)
        self.calls = []

    def call(self, name, args=None):
        self.calls.append((name, dict(args or {})))
        return self._inner.call(name, args)

    def declare(self, schemas):
        return self._inner.declare(schemas)

    def schemas(self):
        return self._inner.schemas()

    def teardown(self):
        return self._inner.teardown()

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _closure_params(tool):
    """闭包签名参数 → 真默认（`inspect.Parameter.empty` = 必填）。"""
    out = {}
    for p in inspect.signature(tool.execute).parameters.values():
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            out[p.name] = p.default
    return out


def _declared(parameters):
    """→ (非 None 的 default 声明, 全部 default 键, 参数名集)。"""
    nonnull, keys = {}, set()
    for name, decl in parameters.items():
        if not isinstance(decl, dict) or "default" not in decl:
            continue
        keys.add(name)
        if decl["default"] is not None:
            nonnull[name] = decl["default"]
    return nonnull, keys, set(parameters)


def _handler_of(server, name):
    """从 SDK 注册表取已注册 handler 本体（判据刻意读私有面，见框架 docstring）。"""
    manager = getattr(server, "_tool_manager", None)
    tool = None if manager is None else manager.get_tool(name)
    if tool is None:
        raise AssertionError('SDK 注册表里取不到 %r（SDK 形状变了？）' % name)
    return tool.fn


# ---------------------------------------------------------------- P 分区（纯函数）
def partition_problems(name, schema, closure):
    """→ 问题列表（纯函数：喂合成坏声明即可证明它抓得住）。"""
    bad = []
    parameters = dict(schema.get("parameters") or {})
    nonnull, decl_keys, decl_params = _declared(parameters)
    try:
        _in_schema, defaulted = tool_input_schema(name, schema, execute=None,
                                                  declared_required=[])
    except Exception as e:                                     # noqa: BLE001
        return ['%s: tool_input_schema 失败 %s: %s' % (name, type(e).__name__, e)]

    if set(defaulted) != set(nonnull):                          # P1
        bad.append('%s: defaulted=%s ≠ 声明非 None default=%s'
                   % (name, sorted(defaulted), sorted(nonnull)))
    for missing in sorted(set(closure) - decl_params):           # P5
        bad.append('%s: 闭包参数 %r 未在声明里' % (name, missing))
    for extra in sorted(decl_params - set(closure)):
        bad.append('%s: 声明参数 %r 闭包签名里没有' % (name, extra))
    for pname, real in closure.items():                          # P2 / P3
        if real is inspect.Parameter.empty or pname not in decl_params:
            continue
        if real is None:
            if pname in nonnull:
                bad.append('%s.%s: 声明 default=%r，闭包真默认是 None（越界声明）'
                           % (name, pname, nonnull[pname]))
        elif pname not in nonnull:
            bad.append('%s.%s: 闭包真默认 %r 未在声明里写 default（漏声明 ⇒ '
                       '显式 null 会以 None 到达工具体）' % (name, pname, real))
        elif nonnull[pname] != real:
            bad.append('%s.%s: 声明 default=%r ≠ 闭包真默认 %r'
                       % (name, pname, nonnull[pname], real))
    for pname in sorted(decl_keys - set(nonnull)):               # P4
        bad.append('%s.%s: 声明了 default=null（与"不声明"同义 ⇒ 只留一种写法）'
                   % (name, pname))
    return bad


# ---------------------------------------------------------------- B 行为
def _placeholder(decl):
    """给必填参数凑占位值（工具体跑不跑得通不影响边界观测）。"""
    return "check-placeholder" if isinstance(decl, dict) and \
        decl.get("type") == "string" else 1


def boundary_problems(registry, schemas):
    """B1a/B2：经真 handler + spy 看**宿主边界**的入参。

    ⚠ R9（别改回空洞版，理由见模块 docstring）：期望值取自 ``closure``（独立信源），
    **不**取自 ``schemas``（被检对象）；观测量是"应丢的键**不存在**/应留的键为 None"，
    **不**是"键的值 == 真默认"——垫片是**丢键**，值由 ``**args`` 解包补上，
    边界上根本看不到"等于默认"这种形态。
    """
    bad = []
    spy = _SpyHost(registry)
    server = build_mcp_server(registry, server_name="el-defaults-check",
                              instructions="判据：default 声明守卫。", toolhost=spy)
    by_name = {s["name"]: s for s in schemas}
    want_drop = want_keep_null = 0
    for name in sorted(by_name):
        params = dict(by_name[name].get("parameters") or {})
        closure = _closure_params(registry.get(name))
        args = {p: (_placeholder(params.get(p) or {})
                    if real is inspect.Parameter.empty else None)
                for p, real in closure.items()}
        _handler_of(server, name)(ctx=None, **args)
    for name, recorded in spy.calls:
        closure = _closure_params(registry.get(name))
        for pname, real in closure.items():
            if real is inspect.Parameter.empty:
                continue
            if real is None:                       # B2：无声明/声明 null ⇒ 保留 null
                want_keep_null += 1
                if pname not in recorded or recorded[pname] is not None:
                    bad.append('B2 %s.%s：真默认 None 的参数传 null 后应仍以 None 到达'
                               '（实得 %s）' % (name, pname,
                                              '缺失' if pname not in recorded
                                              else repr(recorded[pname])))
            else:                                  # B1a：真默认非 None ⇒ 必须被丢掉
                want_drop += 1
                if pname in recorded:
                    bad.append('B1a %s.%s：真默认 %r 的参数传 null 后**没被丢掉**'
                               '（到达本体的是 %r ⇒ 真默认失效）'
                               % (name, pname, real, recorded[pname]))
    if not spy.calls:
        bad.append('B 观测退化：spy 一次调用都没收到')
    return bad, {'calls': len(spy.calls), 'want_drop': want_drop,
                 'want_keep_null': want_keep_null}


def body_problems(registry, history):
    """B1b/B2 的**工具本体**证据：拿 EL 自己的工具端到端看效果。

    - `set_config(..., reason=None)` ⇒ History 事件的 `reason` 必须是真默认 `"ai tool"`
    - `reset_config(reason=None)`   ⇒ 同上（`"ai reset_config"`）
    - `read_waves(atom=None)`       ⇒ 回执 `atom` 必须是真默认 `"single"`
    - `set_config(key, value=None)` ⇒ 清空语义：落史的值必须是 `None`（另一格活着）
    """
    import json

    bad = []
    spy = _SpyHost(registry)
    server = build_mcp_server(registry, server_name="el-defaults-body",
                              instructions="判据：default 声明的本体效果。",
                              toolhost=spy)
    seen = {}

    def _call(name, **kw):
        return json.loads(_handler_of(server, name)(ctx=None, **kw))

    # —— B1b：三个"真默认非 None"的参数，看本体真的收到了默认值 ——
    before = history.last_seq
    _call("set_config", key="field.B_gauss", value=21.7, reason=None)
    evs = [e for e in history.events() if e.seq > before]
    seen['set_config 事件数'] = len(evs)
    if not evs:
        bad.append('B1b set_config：reason=null 后没落史（观测退化）')
    elif evs[-1].reason != "ai tool":
        bad.append('B1b set_config.reason：本体收到 %r ≠ 真默认 "ai tool"'
                   % evs[-1].reason)

    before = history.last_seq
    _call("reset_config", reason=None)
    evs = [e for e in history.events() if e.seq > before]
    seen['reset_config 事件数'] = len(evs)
    if not evs:
        bad.append('B1b reset_config：reason=null 后没落史（观测退化）')
    elif evs[-1].reason != "ai reset_config":
        bad.append('B1b reset_config.reason：本体收到 %r ≠ 真默认 "ai reset_config"'
                   % evs[-1].reason)

    out = _call("read_waves", atom=None)
    seen['read_waves.atom'] = out.get("atom")
    if out.get("atom") != "single":
        bad.append('B1b read_waves.atom：回执 %r ≠ 真默认 "single"' % out.get("atom"))

    # —— B2：另一格（无 default 声明、真默认 None）——显式 null 必须原样到达 ——
    before = history.last_seq
    _call("set_config", key="field.B_gauss", value=None, reason="判据清空")
    evs = [e for e in history.events() if e.seq > before]
    seen['清空事件数'] = len(evs)
    if not evs:
        bad.append('B2 set_config.value=null：没落史（观测退化）')
    elif evs[-1].value is not None:
        bad.append('B2 set_config.value：传 null 应原样到达（清空语义），实得 %r'
                   % evs[-1].value)
    return bad, seen


# ---------------------------------------------------------------- main
def main():
    registry, history = _stack()
    schemas = registry.schemas()                    # 已按键名排序
    print('[mcp_defaults] 工具 %d 个；逐工具核「完备 + 唯一 + 行为」' % len(schemas))

    problems = []
    n_params = n_defaulted = n_none = n_required = 0
    for schema in schemas:
        name = schema["name"]
        closure = _closure_params(registry.get(name))
        n_params += len(closure)
        for real in closure.values():
            if real is inspect.Parameter.empty:
                n_required += 1
            elif real is None:
                n_none += 1
            else:
                n_defaulted += 1
        problems += partition_problems(name, schema, closure)
    C.group('P 分区：defaulted == 声明「非 None default」集（完备 + 唯一 + 值逐位）',
            [('无问题（%s）' % (problems[0][:90] if problems else '共 0 条'),
              not problems)])

    bbad, bstats = boundary_problems(registry, schemas)
    bodybad, bseen = body_problems(registry, history)
    print('[mcp_defaults] 非退化自证：工具=%d 参数=%d 必填=%d 需声明=%d 另一格=%d；'
          '调用=%d 应丢键=%d 应留 null=%d；本体证据=%s'
          % (len(schemas), n_params, n_required, n_defaulted, n_none,
             bstats['calls'], bstats['want_drop'], bstats['want_keep_null'],
             bseen))
    C.group('B 行为：显式 null ⇒ 真默认生效（边界丢键 + 本体生效）；另一格保留 null',
            [('B1a/B2 边界无问题（%s）'
              % (bbad[0][:80] if bbad else '共 0 条'), not bbad),
             ('B1b/B2 本体无问题（%s）'
              % (bodybad[0][:80] if bodybad else '共 0 条'), not bodybad),
             ('非退化自证：工具>0、spy 收到 %d 次调用、应丢键 %d 个、应留 null %d 个'
              % (bstats['calls'], bstats['want_drop'], bstats['want_keep_null']),
              len(schemas) > 0 and bstats['calls'] > 0
              and bstats['want_drop'] > 0 and bstats['want_keep_null'] > 0),
             ('非退化自证：本体证据真出现（%s）' % bseen,
              bseen.get('set_config 事件数', 0) > 0
              and bseen.get('reset_config 事件数', 0) > 0
              and bseen.get('清空事件数', 0) > 0)])

    # ---- C 致死性：故意做坏的声明必须被抓（喂同一批探测器）----
    subs = []
    base = {s["name"]: s for s in schemas}
    pick = next((nm for nm, sc in base.items()
                 if any(v is not None and v is not inspect.Parameter.empty
                        for v in _closure_params(registry.get(nm)).values())), None)
    if pick is None:
        subs.append(('找不到可做坏的样本工具（判据需更新）', False))
    else:
        cl = _closure_params(registry.get(pick))
        target = next(p for p, v in cl.items()
                      if v is not None and v is not inspect.Parameter.empty)
        subs.append(('干净声明 → 零问题（实际 %d）'
                     % len(partition_problems(pick, base[pick], cl)),
                     not partition_problems(pick, base[pick], cl)))
        mut = copy.deepcopy(base[pick])
        mut["parameters"][target].pop("default", None)           # ① 漏声明
        got = partition_problems(pick, mut, cl)
        subs.append(('① 漏声明 %s.%s → 报出（%s）'
                     % (pick, target, got[0][:58] if got else '没报'), bool(got)))
        mut = copy.deepcopy(base[pick])
        mut["parameters"][target]["default"] = "__wrong__"        # ② 值不符
        got = partition_problems(pick, mut, cl)
        subs.append(('② default 值写错 → 报出（%s）'
                     % (got[0][:58] if got else '没报'), bool(got)))
        mut = copy.deepcopy(base[pick])
        mut["parameters"][target]["default"] = None               # ③ 需声明却写 null
        got = partition_problems(pick, mut, cl)
        subs.append(('③ 把需声明的 default 改成 null → 报出（%s）'
                     % (got[0][:58] if got else '没报'), bool(got)))
        none_param = next((p for p, v in cl.items() if v is None), None)
        if none_param is not None:
            mut = copy.deepcopy(base[pick])
            mut["parameters"][none_param]["default"] = "x"        # ④ 越界声明
            got = partition_problems(pick, mut, cl)
            subs.append(('④ 给真默认 None 的参数声明非 None default → 报出（%s）'
                         % (got[0][:58] if got else '没报'), bool(got)))
    C.group('C 致死性：漏声明 / 值写错 / null 声明 / 越界声明 必须被抓（非哑守卫）', subs)

    ok = C.summary()
    for p in problems + bbad + bodybad:
        print('   - %s' % p)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
