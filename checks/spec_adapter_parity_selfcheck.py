# -*- coding: utf-8 -*-
"""spec ⇄ 参照实现 **双向核对**（P0-6 那条线的延伸：契约守卫要双向）。

主代理给的验收法：spec 读完后，新领域应能**只看 spec + hooks.py** 写出适配器。
故用 `examples/adapters.py` 的 `ToyEngine` 反向核对：
  ① **spec → 代码**：spec 列的"必须实现"方法，ToyEngine 是否都有？
  ② **代码 → spec**：ToyEngine 用到的方法/键，spec 是否都写了？
  ③ 可选钩子：spec 列的 4 个钩子名必须与 `mecha/hooks.py` 的导出**完全一致**
     （hooks.py 是默认值的唯一权威定义点）。
对不上的地方 = spec 不可用的地方。

**致死性**：喂一份"漏了某项"的合成 spec 文本 ⇒ 必须报出问题（不是永远返空的哑守卫）。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402

C = Checker('spec_adapter_parity')

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import mecha as _mecha_pkg
    # `mecha.__file__` = <lib>/mecha/__init__.py ⇒ 取**再上一层**才是库根
    _LIB = os.path.dirname(os.path.dirname(os.path.abspath(_mecha_pkg.__file__)))
except Exception:                                             # noqa: BLE001
    _LIB = None
_SPEC = os.path.join(_LIB, 'docs', 'spec-draft.md') if _LIB else None
_EX = os.path.join(_LIB, 'examples', 'adapters.py') if _LIB else None
_HOOKS = os.path.join(_LIB, 'mecha', 'hooks.py') if _LIB else None

if not (_SPEC and os.path.exists(_SPEC) and _EX and os.path.exists(_EX)):
    C.skip('spec ⇄ ToyEngine 双向核对', 'mecha-sdk 不在（库已外移）')
    sys.exit(0 if C.summary() else 1)

_spec = open(_SPEC, encoding='utf-8').read()
_ex = open(_EX, encoding='utf-8').read()
_hooks = open(_HOOKS, encoding='utf-8').read()


def _methods(src, cls):
    """取某个类的 `def name(` 集合（只到下一个顶层 class 为止）。"""
    m = re.search(r'^class %s\b.*?(?=^class |\Z)' % cls, src, re.S | re.M)
    body = m.group(0) if m else ''
    return set(re.findall(r'^\s+def (\w+)\(', body, re.M))


# ---- ① spec 列的必实现方法（§0.1 表格第一列的反引号方法名） ----
_spec_required = set(re.findall(r'^\| `(\w+)\(', _spec, re.M))
# 只取 §0.1 段内的（§0.2/0.4 是可选钩子，另测）
_sec01 = _spec.split('### 0.1')[1].split('### 0.2')[0]
_required = set(re.findall(r'^\| `(\w+)\(', _sec01, re.M))

_toy = _methods(_ex, 'ToyEngine')
_schema_only = _methods(_ex, 'SchemaOnlyAdapter')
_missing_in_toy = sorted(_required - _toy)
_missing_in_schema_only = sorted(_required - _schema_only)

# ---- ② ToyEngine 用到的引擎方法，spec 是否都提到 ----
# ToyEngine 里所有 def 名（排除下划线私有与 _rec 之类）
_toy_public = {n for n in _toy if not n.startswith('_')}
_not_in_spec = sorted(n for n in _toy_public if ('`%s(' % n) not in _spec
                      and ('`%s`' % n) not in _spec)

# ---- ③ 可选钩子名：spec ⇄ hooks.py 导出必须完全一致 ----
_hook_names = set(re.findall(r"__all__\s*=\s*\[([^\]]+)\]", _hooks)[0]
                  .replace("'", '').replace('"', '').split(','))
_hook_names = {h.strip() for h in _hook_names if h.strip()}
_spec_hooks = set(re.findall(r'^\| `(\w+)\(', _spec.split('### 0.2')[1]
                             .split('### 0.3')[0], re.M))
_hooks_only_in_spec = sorted(_spec_hooks - _hook_names)
_hooks_only_in_code = sorted(_hook_names - _spec_hooks)

# ---- ④ 必给键：spec 必须写明（I6 的键契约） ----
_key_contracts = ['ok', 'est_sec', 'full_pops']

C.group('① spec → 代码：spec 列的"必须实现"方法，参照适配器是否都有', [
    ('spec §0.1 列了 %d 个必须实现的方法' % len(_required), len(_required) == 13),
    ('ToyEngine 全都实现（缺：%s）' % (_missing_in_toy or '无'),
     not _missing_in_toy),
    ('**无引擎的 SchemaOnlyAdapter 也全都实现**（缺：%s）'
     % (_missing_in_schema_only or '无'), not _missing_in_schema_only),
])
C.group('② 代码 → spec：ToyEngine 用到的方法，spec 是否都写了', [
    ('ToyEngine 公开方法 %d 个' % len(_toy_public), len(_toy_public) > 0),
    ('都在 spec 里出现（未提及：%s）' % (_not_in_spec or '无'), not _not_in_spec),
])
C.group('③ 可选钩子名：spec ⇄ hooks.py **完全一致**（hooks 是默认值唯一权威定义点）', [
    ('spec §0.2 列了 %d 个钩子' % len(_spec_hooks), len(_spec_hooks) == 4),
    ('hooks.py 导出 %d 个' % len(_hook_names), len(_hook_names) == 4),
    ('spec 有而代码无：%s' % (_hooks_only_in_spec or '无'), not _hooks_only_in_spec),
    ('代码有而 spec 无：%s' % (_hooks_only_in_code or '无'), not _hooks_only_in_code),
])
C.group('④ 键契约已写进 spec（I6：读了就必须声明）', [
    ('`result_summary` 必须含 `ok` 已写明',
     'result_summary' in _spec and '`ok`' in _spec),
    ('`estimate` 应含 `est_sec` 且缺键不静默当 0 已写明',
     '`est_sec`' in _spec and 'estimate_note' in _spec),
    ('`full_pops`（语义随规模变）已写明',
     'full_pops' in _spec and '联合分布' in _spec),
    ('"空 body = 未接线可见信号" 已写明',
     '未接线' in _spec and 'body' in _spec),
])
C.group('⑤ 结构：draft 纪律 + 新增两节在位', [
    ('仍标 draft（N=1 不冻 v1）',
     'draft' in _spec.lower() and '不冻 v1' in _spec),
    ('已注明实例②结论被证伪（验收不成立）',
     'MECHA-INDICTMENT' in _spec or '证伪' in _spec),
    ('§0 引擎接入面在位', '## 0. 引擎接入面' in _spec),
    ('§8 数据治理契约在位', '## 8. 数据治理契约' in _spec),
    ('§8 写明"三源可达性"与"只归档不删"',
     '三源可达性' in _spec and '只归档' in _spec),
])

# ---------------- 致死性：合成"漏项"的 spec 必须被检出 ----------------
# ⚠ 锚点要带**反引号**（§0.1 表里是 `| \`evolve_run(` ），否则替换不命中 ⇒ 假致死性
_ANCHOR = '| `evolve_run('
assert _ANCHOR in _spec, '致死性锚点未命中（spec 表格格式变了？）'
_mut_spec = _spec.replace(_ANCHOR, '| `EVOLVE_RUN_TYPO(')
_mut_required = set(re.findall(r'^\| `(\w+)\(', _mut_spec.split('### 0.1')[1]
                              .split('### 0.2')[0], re.M))
_mut_missing = sorted(_mut_required - _toy)
C.group('⑥ **致死性**：spec 写错一项 ⇒ 双向核对必须报出问题（非哑守卫）', [
    ('变异是**改名**：`evolve_run` 被替换掉、`EVOLVE_RUN_TYPO` 出现',
     'evolve_run' not in _mut_required and 'EVOLVE_RUN_TYPO' in _mut_required),
    ('变异后 ToyEngine 对不上（检出：%s）' % (_mut_missing or '无'),
     bool(_mut_missing)),
    ('未变异时对得上（对照：不误报）', not _missing_in_toy),
])

sys.exit(0 if C.summary() else 1)
