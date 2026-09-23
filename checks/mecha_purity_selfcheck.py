# -*- coding: utf-8 -*-
"""提炼守卫（mecha 纯净性）——「拷走 mecha/+cockpit/ 就等于提炼走框架」的**致死判据**。

点2 重构把领域知识从 mecha 收回 core、立 adapter 胶水层后，`mecha` 必须是**领域无关框架**：
它只经 `runtime` 注入的 `EngineProtocol`（`mecha/protocol.py` 的 ABC）消费领域，自身**零领域
import**。没有这道守卫，任何人往 mecha 塞一行 `from ..core import runspec` 就会悄悄把框架和
领域重新焊死——假绿比红更危险（D4）。本守卫就是那道"提炼边界"的看门狗。

  A. mecha 零领域 import：AST 扫**库的两个包** `mecha/` + `cockpit/` 的全部 .py
     （os.walk，子目录也覆盖），任何 reach 出本包的 import（level≥2 相对 /
     绝对 src.<非本库包> / 裸 arc·core·adapter…）→ 违规。
     ⚠ 旧版只 `os.listdir(mecha/)` 列 **mecha 顶层** .py，而 `cockpit` 是库根的
     **平级包**（不在 mecha/ 内）⇒ cockpit 的 3 个文件根本不在扫描面内——与本文档
     开头"拷走 `mecha/`+`cockpit/` 就等于提炼走框架"的承诺**不符**（本仓
     `journal_files` 犯过同一个错：承诺与实现脱节，见 docs/工程教训-踩过的坑.md）。
     2026-09-22 修：两个包都扫（当天实测 cockpit 零越界，是"覆盖缺口"不是"正在漏"）。
  B. 依赖单向 core↛mecha：AST 扫 `src/core/**/*.py`（**含 manybody/ 子包**，os.walk——
     旧版只列顶层，与 A 的 cockpit 同族缺口），不得 import mecha/adapter（core 是纯物理、
     不认识框架）；adapter/__init__ 是唯一同时踩 core 与 mecha 的胶水层。
  C. 提炼缝就位：mecha/protocol.py 定义 EngineProtocol(ABC 且有抽象方法)、mecha/runtime.py
     提供 set_engine/engine/has_engine——注入式引擎的接缝（mecha 靠它拿领域，而非 import 领域）。
  D. **致死性**：同一探测器扫合成坏源（from ..core import runspec / import arc /
     from src.adapter import X / from .. import core；以及合成 **cockpit** 坏源）
     **必须**检出；扫合成好源（mecha 内部相对 import + stdlib + numpy；cockpit
     内部相对 import + stdlib + starlette）不误报——证明 A 的探测器载重、
     非永远返空的哑守卫。

纯 AST + 轻量 import，不跑引擎（秒级）。只读 <repo> 真实源码，不写任何文件。
"""
import abc
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, 'src')
import mecha as _mecha_pkg            # 独立库 mecha-sdk（editable 安装）；本体已不在本仓
_MECHA = os.path.dirname(os.path.abspath(_mecha_pkg.__file__))
_LIB = os.path.dirname(_MECHA)                   # 库根（mecha/ 与 cockpit/ 的父目录）
_COCKPIT = os.path.join(_LIB, 'cockpit')         # ★ 平级包：提炼边界的另一半
_CORE = os.path.join(_SRC, 'core')
_ADAPTER = os.path.join(_SRC, 'adapter')

C = Checker('mecha_purity')

# mecha 是 src 下的叶子框架包：绝对 import 只准 stdlib / 框架基础设施库（numpy·mcp·
# starlette…），一旦踩到这些"领域/同层业务包"就是越界（框架↛领域）。
_DOMAIN_ABS = {'arc', 'core', 'adapter', 'ui', 'sweep', 'julia'}


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _mecha_domain_hits(tree):
    """mecha 里 reach 出 mecha 的领域 import → [(lineno, 描述)]。

    规则（mecha 只准 import stdlib/框架库 + 自己）：
      · 相对 import level≥2（from ..core / from ..adapter / from .. import core）→ 出了 mecha → 违规
      · 绝对 src.X（X≠mecha）→ 违规；裸 arc/core/adapter/ui/sweep → 违规
      · level==1 相对（mecha 内部）与 stdlib/框架库 → 放行
    """
    hits = []
    for nd in ast.walk(tree):
        if isinstance(nd, ast.ImportFrom):
            if nd.level and nd.level >= 2:
                hits.append((nd.lineno, 'from %s%s（相对越界出 mecha）'
                             % ('.' * nd.level, nd.module or '')))
            elif nd.level == 0 and nd.module:
                parts = nd.module.split('.')
                if parts[0] == 'src':
                    if len(parts) < 2 or parts[1] != 'mecha':
                        hits.append((nd.lineno, 'from %s（绝对跨层）' % nd.module))
                elif parts[0] in _DOMAIN_ABS:
                    hits.append((nd.lineno, 'from %s（领域包）' % nd.module))
        elif isinstance(nd, ast.Import):
            for al in nd.names:
                parts = al.name.split('.')
                if parts[0] == 'src':
                    if len(parts) < 2 or parts[1] != 'mecha':
                        hits.append((nd.lineno, 'import %s（绝对跨层）' % al.name))
                elif parts[0] in _DOMAIN_ABS:
                    hits.append((nd.lineno, 'import %s（领域包）' % al.name))
    return hits


def _framework_hits(tree):
    """core 里 import mecha/adapter → [(lineno, 模块)]（依赖必须单向：core↛框架）。"""
    hits = []
    for nd in ast.walk(tree):
        cands = []
        if isinstance(nd, ast.ImportFrom):
            if nd.module:
                cands.append(nd.module)
            cands += [al.name for al in nd.names]      # from . import mecha
        elif isinstance(nd, ast.Import):
            cands += [al.name for al in nd.names]
        for m in cands:
            toks = m.split('.')
            if 'mecha' in toks or 'adapter' in toks:
                hits.append((nd.lineno, m))
    return hits


def _import_roots(tree):
    """收集一个模块 import 到的所有顶层名（判 adapter 是否同时踩 core 与 mecha）。"""
    roots = set()
    for nd in ast.walk(tree):
        if isinstance(nd, ast.ImportFrom):
            if nd.module:
                roots.add(nd.module.split('.')[0])
            for al in nd.names:
                roots.add(al.name.split('.')[0])
        elif isinstance(nd, ast.Import):
            for al in nd.names:
                roots.add(al.name.split('.')[0])
    return roots


# ---------------- A. 扫真库：mecha/ + cockpit/ 两个包的全部 .py ----------------
# ★ 具名装配点（判据必须测装配）：文件收集与逐文件扫描都走这一个函数——
#   D 组的致死性也调它（不自己造等价物）。改成"只扫一个包"必须让 A 组红。
def lib_py_files(dirs):
    """给定包目录下的全部 .py（os.walk）→ [relpath]（面清点/覆盖断言用）。"""
    out = []
    for pkg in dirs:
        for dp, _dn, fn in os.walk(pkg):
            for f in fn:
                if f.endswith('.py'):
                    out.append(os.path.relpath(os.path.join(dp, f), _LIB)
                               .replace(os.sep, '/'))
    return sorted(out)


def scan_lib_packages(dirs):
    """扫给定包目录（os.walk，子目录也覆盖）→ [(relpath, lineno, 描述)]。

    两个包都传 = 提炼边界的完整承诺（"拷走 mecha/+cockpit/ 即提炼走框架"）；
    只传一个 = 覆盖缺口复辟（旧版 bug：cockpit 平级于 mecha/，listdir 够不着）。
    """
    viol = []
    for rel in lib_py_files(dirs):
        full = os.path.join(_LIB, rel.replace('/', os.sep))
        for ln, desc in _mecha_domain_hits(ast.parse(_read(full))):
            viol.append('%s:%d %s' % (rel, ln, desc))
    return viol


mecha_viol = scan_lib_packages([_MECHA, _COCKPIT])
_all_files = lib_py_files([_MECHA, _COCKPIT])
_n_files = len(_all_files)
_cockpit_files = {f for f in _all_files if f.startswith('cockpit/')}
# 覆盖缺口的**真**致死性：把扫描面收窄成只扫 mecha/ ⇒ cockpit 文件集体消失
_one_pkg_files = lib_py_files([_MECHA])

# ---------------- B. 扫真实 core/**/*.py（core↛mecha；os.walk 含子包） ----------------
# 同族覆盖缺口（与 A 的 cockpit 一对）：旧版 os.listdir(_CORE) 只列 **顶层** .py，
# 而 core 有 `manybody/` 子包（runspec_mb.py 等 8 个文件）⇒ 子包越界看不见。
# 2026-09-22 改 os.walk（实测全库零越界，同样"是缺口不是正在漏"）。
core_viol = []
for dp, _dn, fn in os.walk(_CORE):
    for f in sorted(fn):
        if f.endswith('.py'):
            rel = os.path.relpath(os.path.join(dp, f), _CORE).replace(os.sep, '/')
            for ln, mod in _framework_hits(ast.parse(_read(os.path.join(dp, f)))):
                core_viol.append('%s:%d %s' % (rel, ln, mod))

# adapter 是唯一胶水层：__init__ 同时 import core 与 mecha
_ad_roots = _import_roots(ast.parse(_read(os.path.join(_ADAPTER, '__init__.py'))))
ad_glue = ('core' in _ad_roots) and ('mecha' in _ad_roots)

# ---------------- C. 提炼缝就位 ----------------
from mecha import protocol as _MP                      # noqa: E402
from mecha import runtime as _MRT                      # noqa: E402
_EProto = getattr(_MP, 'EngineProtocol', None)
seam_proto = (isinstance(_EProto, type) and issubclass(_EProto, abc.ABC)
              and len(getattr(_EProto, '__abstractmethods__', frozenset())) > 0)
seam_runtime = all(hasattr(_MRT, n) for n in ('set_engine', 'engine', 'has_engine'))

# ---------------- D. 致死性：探测器必须能辨别好坏（含 cockpit 面） ----------------
_ROGUE = [
    'from ..core import runspec\n',
    'import arc\n',
    'from src.adapter import EngineAdapter\n',
    'from .. import core\n',
    'from ..core.static_atom import build_level_rows\n',
]
_GOOD = ('from .session import Session\nfrom . import runtime\n'
         'import os, json, numpy as np\nfrom mcp.server.mcpserver import MCPServer\n')
# ★ cockpit 面（P1-9h 覆盖缺口修复的致死性）：**同一个**逐文件扫描函数喂合成源。
# 坏源 = 用户点名的例子（from ..core import runspec 出现在 cockpit 源里）；
# 好源 = 真实 cockpit 的 import 形态（from . import backend + stdlib + starlette）。
_COCKPIT_ROGUE = 'from ..core import runspec\nimport json, os\n'
_COCKPIT_GOOD = ('from . import backend\nimport json, os, argparse\n'
                 'from starlette.applications import Starlette\n'
                 'from starlette.routing import Route\n')
lethal = [len(_mecha_domain_hits(ast.parse(s))) for s in _ROGUE]
good_hits = len(_mecha_domain_hits(ast.parse(_GOOD)))
lethal_cockpit = len(_mecha_domain_hits(ast.parse(_COCKPIT_ROGUE)))
good_cockpit_hits = len(_mecha_domain_hits(ast.parse(_COCKPIT_GOOD)))

C.group('A 库零领域 import（拷走 mecha/+cockpit/ 即提炼走框架；扫 %d 个 .py；'
        '违规：%s）' % (_n_files, mecha_viol or '无'), [
    ('mecha/ + cockpit/ 两个包**全部** .py 无任何 reach 出本包的 import'
     '（level≥2 / src.<非本库包> / 裸领域包）', not mecha_viol),
    ('**覆盖面断言**：cockpit 的 3 个文件（%s）**在扫描面内**——只扫 mecha/ 顶层'
     '的旧实现让 cockpit 整个包游离在边界外' % sorted(_cockpit_files),
     _cockpit_files == {'cockpit/__init__.py', 'cockpit/backend.py',
                        'cockpit/server.py'}),
    ('**致死性（覆盖缺口）**：扫描面收窄成只扫 mecha/ ⇒ cockpit 文件从面内消失'
     '（上面那条断言随之红——证明它不是空话）',
     not any(f.startswith('cockpit/') for f in _one_pkg_files)
     and any(f.startswith('cockpit/') for f in _all_files)),
])
C.group('B 依赖单向 core↛mecha + adapter 是唯一胶水层（core 含 manybody/ 子包全扫；'
        '违规：%s）' % (core_viol or '无'), [
    ('src/core/**/*.py（含子包）无 import mecha/adapter（core 是纯物理、不认识框架）',
     not core_viol),
    ('adapter/__init__ 同时 import core 与 mecha（唯一同时踩两侧的胶水层）', ad_glue),
])
C.group('C 提炼缝就位：EngineProtocol(ABC) + runtime 注入点', [
    ('mecha.protocol.EngineProtocol 是带抽象方法的 ABC（框架只认接口）', seam_proto),
    ('mecha.runtime 提供 set_engine/engine/has_engine（注入式引擎接缝）', seam_runtime),
])
C.group('D 致死性：探测器检出坏 import 且不误报好 import（非哑守卫；含 cockpit 面）', [
    ('from ..core import runspec 被检出', lethal[0] >= 1),
    ('import arc 被检出', lethal[1] >= 1),
    ('from src.adapter import X 被检出', lethal[2] >= 1),
    ('from .. import core 被检出', lethal[3] >= 1),
    ('from ..core.static_atom import X 被检出', lethal[4] >= 1),
    ('mecha 内部相对 import + stdlib + numpy + mcp 不误报', good_hits == 0),
    ('★ **cockpit 合成坏源**（from ..core import runspec）必须检出',
     lethal_cockpit >= 1),
    ('★ **cockpit 合成好源**（from . import backend + stdlib + starlette）'
     '零误报', good_cockpit_hits == 0),
])

sys.exit(0 if C.summary() else 1)
