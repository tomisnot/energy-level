# -*- coding: utf-8 -*-
"""README ⇄ 代码一致性守卫（mecha-sdk）——让"README 说 17 方法、实际 13"这类漂移自己变红。

**为什么需要**：`mecha-sdk/README.md` 曾一次漂三处（实测）：
  · extras 漏 `numpy`（`pyproject.toml` 里有）；
  · 写「EngineProtocol = 17 方法」（ABC 实际 **13 必实现 + 4 可选钩子**，且钩子默认值在 `hooks`）；
  · 模块清单缺 `hooks`/`gc`/`runtime`/`subscribe`/`app_bridge`。
**声明与实现不一致时没有任何东西会响**（I6 哑告警 / D4 假绿同一族）——本守卫补上这一环。

三项对拍（**纯文本 + 正则 + 一次 import**）：
  ① README 模块清单 == 实际 `mecha/*.py` 文件集（允许**显式白名单**注明"内部模块不提"）；
  ② README 声称的方法数 == ABC 实际抽象方法数（并核对可选钩子数 == `hooks.__all__`）；
  ③ README 的 extras 清单 == `pyproject.toml` 的 extras。

**边界（诚实声明）**：
  · 只看这三项截面（模块集/方法数/extras）；README 的**散文措辞**不在此判。
  · 模块名按"反引号里出现的 `xxx`"提取 ⇒ **README 里举例请写模块名本身**，
    不要写路径或调用写法（否则会被当模块名）。
  · **不许误报**：好 README（名称齐全、数字对）必须 0 问题；坏 README 必须报出。

致死性：喂合成坏 README（少一个模块 / 数字写错 / extras 少一个）⇒ 必须报出。
"""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402

C = Checker('mecha_readme_consistency')

# ---- 定位库 ----
try:
    import mecha as _mp
    _MECHA_DIR = os.path.dirname(os.path.abspath(_mp.__file__))     # <lib>/mecha
    _LIB = os.path.dirname(_MECHA_DIR)                              # <lib>
except Exception:                                                   # noqa: BLE001
    C.skip('README ⇄ 代码一致性', 'mecha-sdk 不在（库已外移）')
    sys.exit(0 if C.summary() else 1)

_README = os.path.join(_LIB, 'README.md')
_PYPROJECT = os.path.join(_LIB, 'pyproject.toml')

# 允许"README 不单独列"的内部模块（必须写明原因，否则视为漂移）
_NOT_LISTED_OK = set()          # 目前为空：README 列全了


def _modules_on_disk():
    return {f[:-3] for f in os.listdir(_MECHA_DIR)
            if f.endswith('.py') and f != '__init__.py'}


def audit_readme(text, mods, pyproject):
    """→ 问题列表（纯函数，便于喂合成坏输入做致死性）。"""
    problems = []
    # ① 模块清单
    mentioned = set(re.findall(r'`([a-z_][a-z0-9_]*)`', text))
    missing = sorted(mods - mentioned - _NOT_LISTED_OK)
    if missing:
        problems.append('README 模块清单缺：%s' % missing)
    # ② 方法数：README 里的"N 个必实现"与"ABC 抽象方法数"
    m = re.search(r'(\d+)\s*个必实现', text)
    if m:
        n_readme = int(m.group(1))
        abc = _abc_counts()
        if n_readme != abc['n_abstract']:
            problems.append('README 说 %d 个必实现，ABC 实际 %d'
                            % (n_readme, abc['n_abstract']))
    else:
        problems.append('README 未声明"必实现"方法数（无法对拍）')
    m2 = re.search(r'(\d+)\s*个可选钩子', text)
    if m2:
        n_hooks_readme = int(m2.group(1))
        hooks_n = len(_hook_names())
        if n_hooks_readme != hooks_n:
            problems.append('README 说 %d 个可选钩子，hooks 实际 %d'
                            % (n_hooks_readme, hooks_n))
    # ③ extras
    ex_readme = set(re.findall(r'extras[：:]\s*([^\n。]+)', text))
    if ex_readme:
        got = set()
        for chunk in ex_readme:
            got |= set(re.findall(r'`([a-z_][a-z0-9_]*)`', chunk))
        want = _extras(pyproject)
        if got != want:
            problems.append('README extras %s ≠ pyproject %s'
                            % (sorted(got), sorted(want)))
    return problems


def _abc_counts():
    src = open(os.path.join(_MECHA_DIR, 'protocol.py'), encoding='utf-8').read()
    tree = ast.parse(src)
    n_abs = 0
    for nd in ast.walk(tree):
        if isinstance(nd, ast.FunctionDef):
            if any(isinstance(d, ast.Name) and d.id == 'abstractmethod'
                   for d in nd.decorator_list):
                n_abs += 1
    return {'n_abstract': n_abs}


def _hook_names():
    src = open(os.path.join(_MECHA_DIR, 'hooks.py'), encoding='utf-8').read()
    m = re.search(r'__all__\s*=\s*\[([^\]]+)\]', src)
    if not m:
        return []
    return [x.strip().strip("'\"") for x in m.group(1).split(',') if x.strip()]


def _extras(pyproject):
    src = open(pyproject, encoding='utf-8').read()
    m = re.search(r'\[project\.optional-dependencies\](.*?)(?=\n\[|\Z)', src, re.S)
    if not m:
        return set()
    return set(re.findall(r'^([a-z_][a-z0-9_]*)\s*=', m.group(1), re.M))


_mods = _modules_on_disk()
_abc = _abc_counts()
_hooks = _hook_names()
_extras_want = _extras(_PYPROJECT)
_text = open(_README, encoding='utf-8').read()
_problems = audit_readme(_text, _mods, _PYPROJECT)

C.group('① README 模块清单 == 实际 `mecha/*.py`（%d 个模块）' % len(_mods), [
    ('库内模块 %d 个：%s' % (len(_mods), sorted(_mods)), len(_mods) > 10),
    ('README 全列到（缺：%s）'
     % (sorted(_mods - set(re.findall(r'`([a-z_][a-z0-9_]*)`', _text))
               - _NOT_LISTED_OK) or '无'),
     not (_mods - set(re.findall(r'`([a-z_][a-z0-9_]*)`', _text))
          - _NOT_LISTED_OK)),
    ('新机制 `hooks`/`gc` 被提到（新读者要知道"可选钩子有默认值""治理由库提供"）',
     '`hooks`' in _text and '`gc`' in _text),
])
C.group('② 方法数：README 声明 == ABC 实际（抽象 %d / 钩子 %d）'
        % (_abc['n_abstract'], len(_hooks)), [
    ('ABC 抽象方法 %d 个' % _abc['n_abstract'], _abc['n_abstract'] == 13),
    ('hooks 导出 %d 个' % len(_hooks), len(_hooks) == 4),
    ('README 声明的"必实现"数 == 13',
     bool(re.search(r'13\s*个必实现', _text))),
    ('README 声明的"可选钩子"数 == 4',
     bool(re.search(r'4\s*个可选钩子', _text))),
    ('README 不再声称"17 方法"（旧漂移）', '17 方法' not in _text),
])
C.group('③ README extras == pyproject extras（%s）' % sorted(_extras_want), [
    ('pyproject extras = %s' % sorted(_extras_want),
     _extras_want == {'mcp', 'cockpit', 'numpy'}),
    ('README 的 extras 行含 numpy（旧漂移：漏了它）',
     bool(re.search(r'extras[：:][^\n]*numpy', _text))),
])
C.group('④ 综合：README 与代码一致（问题：%s）' % (_problems or '无'), [
    ('零问题', not _problems),
])

# ---------------- 致死性 + 不许误报 ----------------
_bad_mod = _text.replace('`gc`', '`GONE`')
_bad_num = _text.replace('13 个必实现', '17 个必实现')
_bad_extras = _text.replace('extras：`mcp`、`cockpit`、`numpy`',
                            'extras：`mcp`、`cockpit`')
C.group('⑤ **致死性**：坏 README 必被检出；好 README **不误报**', [
    ('少一个模块 → 报出',
     bool(audit_readme(_bad_mod, _mods, _PYPROJECT))),
    ('方法数写错 → 报出',
     any('必实现' in p for p in audit_readme(_bad_num, _mods, _PYPROJECT))),
    ('extras 少一个 → 报出',
     any('extras' in p for p in audit_readme(_bad_extras, _mods, _PYPROJECT))),
    ('**不许误报**：当前真 README → 0 问题', not _problems),
])

sys.exit(0 if C.summary() else 1)
