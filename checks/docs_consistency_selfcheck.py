# -*- coding: utf-8 -*-
"""文档-代码一致性守卫 —— 让「文档说 16 个工具、代码实际 18 个」这类漂移自己变红。

**为什么需要它**：本项目反复付代价的一类错是**文档漂移**——同一事实在代码、spec、skill、
专家手册里各写一遍，改代码时漏改文档，人只能靠肉眼发现（实测：`server.py` 注册 18 个工具，
而 `SKILL.md` 写"16 个"、`spec-draft.md` 仍指已删除的 `src/mecha/server.py`）。
这属于纲领 **I6**（"契约外字段读出来永远是 None"⇒ 哑告警）与 **D4**（假绿比红更危险）的同一族：
**声明与实现不一致时，没有任何东西会响**。

本守卫只做一件有界的事，但把它做死：
  1. `mecha-sdk/docs/spec-draft.md` 的工具面段落里出现的工具名 ⇒ 必须都在 `server.TOOL_NAMES` 里；
     且 `TOOL_NAMES` 的每个工具都必须在该文档里被提到（**两边互相钉死**）。
  2. 工作区副本 `.dsh/skills/re0-mecha/SKILL.md` 里"## 工具（N 个）"的 **N** 必须 == `len(TOOL_NAMES)`。
  3. 库的 `mecha/SKILL.md`（领域无关版）同理检查其数量声明（若有）。
  4. **致死性**：同一探测器喂合成坏文档（数量写错 / 工具名写错）**必须**报出问题——
     证明它不是"永远返空"的哑守卫。

**边界（诚实声明，避免守卫假装万能）**：
  · 只看**工具面**这一个截面。参数键、事件形状、readback 语义的漂移**本守卫不管**
    （那些需要 P0-5/P0-6 的键契约对拍，见 `docs/规划-下一阶段.md`）。
  · 只做**集合与计数**比对，不理解语义（"这个工具该不该有"不在这里判）。
  · 找不到 `mecha-sdk` 时**明确 SKIP 并说明**（不静默通过）——库已外移，守卫不能假装它还在本仓。
  · ★ **启发式的误报边界**：工具名是按"反引号里出现 `name(...)`"提取的，
    所以**叙述性**的反引号调用写法（如 `` `engine.mcp_tools()` ``、`` `obj.method()` ``）
    会被当成工具名而报红。**文档里举例请写 `name` 或改成不带括号的说法，不要写 `name(...)`**；
    要指某个方法请用别的方式表述。**不要为了文档方便而放宽本守卫**——
    它正是抓"文档说 16、代码实际 18"这类漂移的那一条（放宽一次就永久失去这个能力）。

纯文本 + 正则 + 一次 import；不跑引擎、不写任何文件。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = Checker('docs_consistency')

# ---------------------------------------------------------------- 定位 mecha-sdk
# 库已从本仓外移（2026-09-22），故这里必须**显式找**而不是假定在同仓。
_CANDIDATES = [
    os.environ.get('MECHA_SDK_DIR'),
    os.path.join(os.path.dirname(_ROOT), 'mecha-sdk'),          # 同级仓（当前布局）
]


def _find_sdk():
    for c in _CANDIDATES:
        if c and os.path.isfile(os.path.join(c, 'mecha', 'server.py')):
            return os.path.abspath(c)
    return None


_SDK = _find_sdk()
_SPEC = os.path.join(_SDK, 'docs', 'spec-draft.md') if _SDK else None
_LIB_SKILL = os.path.join(_SDK, 'mecha', 'SKILL.md') if _SDK else None
_WS_SKILL = os.path.join(_ROOT, '.dsh', 'skills', 're0-mecha', 'SKILL.md')

# ---------------------------------------------------------------- 纯函数探测器
_COUNT_RE = re.compile(r'工具（(\d+)\s*个）|工具\s*\((\d+)\s*个\)')


def declared_tool_count(text):
    """文档里声明的工具数量（找不到 → None）。"""
    m = _COUNT_RE.search(text)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def backticked_head_names(text):
    """反引号内容的**首个标识符**集合。

    文档里工具常写成 `` `list_params(prefix,family,page)` `` 或 `` `undo(reason)` ``——
    直接取整个反引号内容会得到一个带参数表的串，导致"工具被提及"漏判（本守卫第一版
    就踩了这个坑：把 6 个其实写了的工具误报成未提及）。
    """
    heads = set()
    for span in re.findall(r'`([^`]+)`', text):
        m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', span)
        if m:
            heads.add(m.group(1))
    return heads


def spec_tool_section(text):
    """spec 的「## 1. MCP 工具面」段落（到下一个二级标题为止）。"""
    m = re.search(r'^##\s*1\.[^\n]*\n(.*?)(?=^##\s|\Z)', text, re.S | re.M)
    return m.group(1) if m else ''


def audit_spec(text, tools):
    """→ 问题列表。tools = 权威工具名集合。

    · 段落里出现的**工具名**必须都在 tools 里（不认识的名字 = 契约与实现脱节）；
    · tools 的每个工具都必须在该段落里被提到（新增工具后忘了写进契约 = 漂移）。
    """
    problems = []
    sec = spec_tool_section(text)
    if not sec:
        return ['spec 里找不到「## 1. MCP 工具面」段落（结构变了？守卫需同步）']
    heads = backticked_head_names(sec)
    known = set(tools) | _ALL_KNOWN_ALIASES
    # 段落里"看起来是工具名"的：在 tools/别名里，或带调用括号（`foo(...)`）。
    called = {re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', s).group(1)
              for s in re.findall(r'`([^`]+)`', sec) if '(' in s}
    unknown = (heads & called) - known
    if unknown:
        problems.append('spec 提到的工具名不在 TOOL_NAMES：%s' % sorted(unknown))
    missing = set(tools) - heads
    if missing:
        problems.append('TOOL_NAMES 里有 %d 个工具 spec 未提及：%s'
                        % (len(missing), sorted(missing)))
    return problems


def audit_skill(text, tools):
    """→ 问题列表（若文档**声明了**数量，则必须等于实际）。

    注意：**不要求**每份 skill 都写数量——领域无关版 `mecha/SKILL.md` 是**故意不写**的
    （"工具/键名由项目适配器提供，用 `list_tools` 发现"，见该文件 §开头）。对它而言
    "不写数量"才是对的设计，硬要求数量反而是错的（本守卫第一版就犯了这个错）。
    """
    problems = []
    n = declared_tool_count(text)
    if n is not None and n != len(tools):
        problems.append('skill 声明 %d 个工具，实际注册 %d 个（差 %d）'
                        % (n, len(tools), len(tools) - n))
    return problems


def declares_tool_count(text):
    return declared_tool_count(text) is not None


# 已知的历史别名（出现在 spec 文字里但已改名的旧名），不算"未知"——
# 显式登记，避免守卫因叙述性旧名误报。
_ALL_KNOWN_ALIASES = {'run_python', 'run_codeact'}

# ---------------------------------------------------------------- 取权威工具名
_sdk_problem = None
TOOLS = set()
if _SDK:
    try:
        from mecha.server import TOOL_NAMES
        TOOLS = set(TOOL_NAMES)
    except Exception as e:                                   # noqa: BLE001
        _sdk_problem = 'import mecha.server 失败：%s: %s' % (type(e).__name__, e)
else:
    _sdk_problem = '找不到 mecha-sdk（已试：%s）' % [c for c in _CANDIDATES if c]


def _read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------- A. 守卫可用性
C.group('A 前提：能取到权威工具面（库已外移，找不到需明说而非静默通过）', [
    ('找到 mecha-sdk 目录并成功导入 mecha.server.TOOL_NAMES（%s）'
     % (_SDK or '未找到'), bool(TOOLS) and not _sdk_problem),
    ('工具数 > 0（%d）' % len(TOOLS), len(TOOLS) > 0),
] + ([('SDK 不可用原因：%s' % _sdk_problem, False)] if _sdk_problem else []))

# ---------------------------------------------------------------- B. spec ⇄ 代码
if TOOLS and _SPEC and os.path.isfile(_SPEC):
    _spec_problems = audit_spec(_read(_SPEC), TOOLS)
    C.group('B spec（合规契约）⇄ 代码工具面互钉（问题：%s）'
            % (_spec_problems or '无'), [
        ('spec 提到的工具名都在 TOOL_NAMES 里，且 TOOL_NAMES 每个工具都被 spec 提到',
         not _spec_problems),
    ])
else:
    C.check('B spec ⇄ 代码工具面互钉', False,
            'spec 不存在或工具面取不到（spec=%s）' % _SPEC)

# ---------------------------------------------------------------- C. skill 数量声明
if TOOLS and os.path.isfile(_WS_SKILL):
    _ws_problems = audit_skill(_read(_WS_SKILL), TOOLS)
    C.group('C 工作区 skill 副本的数量声明 ⇄ 代码（问题：%s）'
            % (_ws_problems or '无'), [
        ('「工具（N 个）」的 N == len(TOOL_NAMES)==%d' % len(TOOLS), not _ws_problems),
    ])
else:
    C.check('C 工作区 skill 数量声明', False, 'skill 或工具面取不到')

if TOOLS and _LIB_SKILL and os.path.isfile(_LIB_SKILL):
    _lib_text = _read(_LIB_SKILL)
    _lib_problems = audit_skill(_lib_text, TOOLS)
    # 领域无关版**故意不写数量**：只在它写了数量时校验，没写视为正确设计。
    C.check('C2 库内 SKILL.md（领域无关版）数量声明一致（未声明=合规）',
            not _lib_problems,
            '; '.join(_lib_problems) or '')
    C.check('C2b 领域无关版确实不写死工具数量（避免与代码漂移）',
            not declares_tool_count(_lib_text),
            'mecha/SKILL.md 写了「工具（N 个）」——领域无关版应由 list_tools 发现，'
            '写死数量会随代码漂移')
else:
    C.check('C2 库内 SKILL.md 存在且可审', False, '未找到 %s' % _LIB_SKILL)

# ---------------------------------------------------------------- D. 致死性
_FAKE_TOOLS = {'list_params', 'set_param', 'undo'}
_FAKE_SPEC_BAD = (
    '# fake\n\n## 1. MCP 工具面（写 / 读）\n\n'
    '| 类 | 工具 |\n|---|---|\n| 读 | `list_params(prefix)` / `nonexistent_tool(x)` |\n\n'
    '## 2. 别的东西\n')
_FAKE_SPEC_GOOD = (
    '# fake\n\n## 1. MCP 工具面（写 / 读）\n\n'
    '| 类 | 工具 |\n|---|---|\n| 读 | `list_params(prefix)` |\n'
    '| 写 | `set_param(key,value,reason)` / `undo(reason)` |\n\n'
    '## 2. 别的东西\n')
_FAKE_SKILL_BAD = '## 工具（16 个）\n'
_FAKE_SKILL_GOOD = '## 工具（3 个）\n'
_FAKE_SKILL_NOCOUNT = '# mecha 语义层 · AI 调用指南（领域无关）\n\n用 `list_tools` 发现。\n'

_lethal_spec = audit_spec(_FAKE_SPEC_BAD, _FAKE_TOOLS)
_good_spec = audit_spec(_FAKE_SPEC_GOOD, _FAKE_TOOLS)
_lethal_skill = audit_skill(_FAKE_SKILL_BAD, _FAKE_TOOLS)
_good_skill = audit_skill(_FAKE_SKILL_GOOD, _FAKE_TOOLS)

C.group('D 致死性：探测器抓得住坏文档、且不误报好文档（非哑守卫）', [
    ('spec 里的未知工具名（带括号写法）→ 报错',
     any('不在 TOOL_NAMES' in p for p in _lethal_spec)),
    ('spec 漏提工具 → 报错', any('未提及' in p for p in _lethal_spec)),
    ('干净 spec（工具名带参数表）→ 无问题（防"带括号就误报"）', not _good_spec),
    ('skill 数量写错（16 vs 实际 3）→ 报错',
     any('声明 16' in p for p in _lethal_skill)),
    ('skill 数量写对（3 vs 实际 3）→ 无问题', not _good_skill),
    ('领域无关版不写数量 → 不报错（防"硬要求数量"的误判）',
     not audit_skill(_FAKE_SKILL_NOCOUNT, _FAKE_TOOLS)),
])

_ok = C.summary()
# 只在**作为脚本跑**时带退出码；被 import（如复用其中的纯探测器做测试）时不打断调用方。
# run_checks.py 以子进程跑本文件，故正常路径不受影响。
if __name__ == '__main__':
    sys.exit(0 if _ok else 1)
