# -*- coding: utf-8 -*-
"""文档-代码一致性守卫（v2）—— 让「SKILL 说 12 个工具、代码实际注册 13 个」这类漂移自己变红。

**为什么需要它**：本项目反复付代价的一类错是**文档漂移**——工具面在代码、SKILL、
spec 里各写一遍，改代码时漏改文档，人只能靠肉眼发现。属纲领 I6/D4 同族：
**声明与实现不一致时，没有任何东西会响**。

v2 权威工具面 = `examples/energy_level/tools.build_tool_registry` 注册的
`ToolRegistry.schemas()` 名单（不维护手写 TOOL_NAMES，消 D-25）。本守卫只做一件
有界的事，但把它做死：

  1. 从**真注册表**取工具名（build 只依赖 engine.schema()，不跑演化、不建 cache
     ——用 schema-only 引擎，秒级）。
  2. `skills/energy-level/SKILL.md` 的「## 工具清单」表：每行首个反引号名 ⇒ 必须
     都在注册表里；且注册表每个工具都必须在该表出现（**两边互相钉死**）。
  3. **致死性**：同一探测器喂合成坏文档（表里多一个不存在的工具 / 漏一个已注册
     工具）**必须**报出问题——证明它不是"永远返空"的哑守卫。

**边界（诚实声明）**：只看**工具面**这一个截面；参数键、事件形状、readback 语义的
漂移本守卫不管。表名提取按"行首反引号标识符"，非工具的反引号写法（如 `list_tools`、
`read_config` 之外的方法名）若出现在表里会被当工具名——所以**表里只准列工具**。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = Checker('docs_consistency')

_SKILL = os.path.join(_ROOT, 'skills', 'energy-level', 'SKILL.md')


# ---------------------------------------------------------------- 取权威工具名
def _registered_tool_names():
    """真注册表投影：build_tool_registry(engine, gate, surface, authority, ch, jobs)。

    build 期只调 engine.schema()（工具闭包不执行、QUERY 注册只存表）⇒ 用 schema-only
    引擎即可，无需建 ARC cache（省 ~10s）。名单 = 注册的 MCP 工具（不含 run_actions
    里派生的纯查询项 get_waves——它不是顶层工具）。
    """
    from src.core import protocol as P
    from examples.energy_level.tools import build_tool_registry
    from mecha.authority import Authority, Mode
    from mecha.gate import Channel, Gate, accept_all
    from mecha.history import History
    from mecha.providers.jobs import JobRegistry
    from mecha.surface import Surface

    class _SchemaOnly:
        def schema(self):
            return {"run_output_required": ["ok", "method"],
                    "estimate_output_required": ["est_sec"],
                    "config": P.config_schema(), "solve": P.solve_schema(),
                    "engine": P.engine_params(), "settable_keys": P.settable_keys()}

    eng = _SchemaOnly()
    hist = History()
    auth = Authority(mode=Mode.AI)
    gate = Gate(auth, hist, validate=accept_all)
    reg = build_tool_registry(eng, gate, Surface(eng), auth,
                              Channel("ai"), JobRegistry())
    return {s["name"] for s in reg.schemas()}


# ---------------------------------------------------------------- 纯函数探测器
_TOOL_ROW_RE = re.compile(r'^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|')


def skill_table_names(text):
    """「## 工具清单」小节里每行首个反引号名（= 表列的工具）。"""
    m = re.search(r'^##\s*工具清单.*?$', text, re.M)
    if not m:
        return None
    tail = text[m.end():]
    nxt = re.search(r'^##\s', tail, re.M)
    section = tail[:nxt.start()] if nxt else tail
    return {_TOOL_ROW_RE.match(ln).group(1)
            for ln in section.splitlines() if _TOOL_ROW_RE.match(ln)}


def audit_skill(text, tools):
    """→ 问题列表：表内工具名 ⇄ 注册表互钉。"""
    problems = []
    tbl = skill_table_names(text)
    if tbl is None:
        return ['SKILL 里找不到「## 工具清单」小节（结构变了？守卫需同步）']
    unknown = tbl - tools
    if unknown:
        problems.append('SKILL 表里的工具名不在注册表：%s' % sorted(unknown))
    missing = tools - tbl
    if missing:
        problems.append('注册表里有 %d 个工具 SKILL 表未列：%s'
                        % (len(missing), sorted(missing)))
    return problems


# ---------------------------------------------------------------- A. 权威面可用
_err = None
TOOLS = set()
try:
    TOOLS = _registered_tool_names()
except Exception as e:                                       # noqa: BLE001
    _err = 'build_tool_registry 失败：%s: %s' % (type(e).__name__, e)
C.group('A 前提：从真注册表取到权威工具面（不维护手写名单）', [
    ('注册表工具数 > 0（%d）%s' % (len(TOOLS), _err or ''), bool(TOOLS)),
])

# ---------------------------------------------------------------- B. SKILL ⇄ 代码
if TOOLS and os.path.isfile(_SKILL):
    with open(_SKILL, encoding='utf-8') as f:
        _skill_text = f.read()
    _probs = audit_skill(_skill_text, TOOLS)
    C.group('B SKILL 工具表 ⇄ 注册表互钉（问题：%s）' % (_probs or '无'), [
        ('表内工具名都在注册表里，且注册表每个工具都被表列出', not _probs),
    ])
else:
    C.check('B SKILL ⇄ 代码互钉', False,
            'SKILL 不存在或工具面取不到（skill=%s err=%s）' % (_SKILL, _err))

# ---------------------------------------------------------------- C. 致死性
_FAKE_TOOLS = {'set_config', 'read_config', 'run_sim'}
_FAKE_BAD_EXTRA = '## 工具清单\n\n| 工具 | 作用 |\n|---|---|\n| `set_config` | w |\n| `read_config` | r |\n| `run_sim` | run |\n| `nonexistent_tool` | 幽灵 |\n'
_FAKE_BAD_MISSING = '## 工具清单\n\n| 工具 | 作用 |\n|---|---|\n| `set_config` | w |\n'
_FAKE_GOOD = '## 工具清单（v2 实际面 = 3）\n\n| 工具 | 作用 |\n|---|---|\n| `set_config` | w |\n| `read_config` | r |\n| `run_sim` | run |\n'
C.group('C 致死性：探测器抓得住坏文档、不误报好文档（非哑守卫）', [
    ('表里多一个不存在的工具 → 报错',
     any('不在注册表' in p for p in audit_skill(_FAKE_BAD_EXTRA, _FAKE_TOOLS))),
    ('表里漏一个已注册工具 → 报错',
     any('未列' in p for p in audit_skill(_FAKE_BAD_MISSING, _FAKE_TOOLS))),
    ('干净表（三工具齐全）→ 无问题',
     not audit_skill(_FAKE_GOOD, _FAKE_TOOLS)),
    ('找不到「## 工具清单」小节 → 报错（结构漂移不静默）',
     bool(audit_skill('# 没有该小节\n', _FAKE_TOOLS))),
])

_ok = C.summary()
if __name__ == '__main__':
    sys.exit(0 if _ok else 1)
