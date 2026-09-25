# -*- coding: utf-8 -*-
"""清洗守卫（切换判据核心·致死）：主干**代码**里 grep-0 旧框架符号。

《回归主仓库》要求：Re0 覆盖旧仓前，主干（src/ + 根 .py + checks/，排除 _archive）不得
再**依赖**旧框架。判据用 tokenize 只扫**代码 token（NAME）**、跳过 COMMENT/STRING——
解释性注释（"本层取代旧 pm_setup"）是有益文档、不算脏；真正的 `import pm_setup` /
`pm_setup.god_split(...)` / `self.engine_overrides` 才是哑弹，必须 0 命中。

禁止的旧框架标识符（作为代码 NAME 出现即 FAIL）：
  pm_setup / engine_overrides / god_split / god_write / god_read / sim_state /
  _SOLVE_SPECS / register_panel_params / new_registry / _P（core.params 别名）/ archive
（`core.params` 的引用总以 `import params as _P` 出现 → 由 `_P` 兜住。）

本守卫自身 + checks/_archive 排除。命中即 FAIL 并逐条打印（file:line:符号）。
"""
import io
import os
import sys
import tokenize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checks.checklib import Checker                        # noqa: E402

C = Checker('cleanliness_guard')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBID_IDS = {
    'pm_setup', 'engine_overrides', 'god_split', 'god_write', 'god_read',
    'sim_state', '_SOLVE_SPECS', 'register_panel_params', 'new_registry',
    '_P', 'archive',
}
SELF = os.path.abspath(__file__)


def _py_files():
    out = []
    for base in (os.path.join(ROOT, 'src'), os.path.join(ROOT, 'checks')):
        for dp, _dns, fns in os.walk(base):
            if '_archive' in dp or '__pycache__' in dp:
                continue
            for fn in fns:
                if fn.endswith('.py'):
                    out.append(os.path.join(dp, fn))
    for fn in ('app_entry.py',):
        p = os.path.join(ROOT, fn)
        if os.path.exists(p):
            out.append(p)
    return out


hits = []
_SKIP = (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
         tokenize.COMMENT, tokenize.ENCODING)
for path in _py_files():
    if os.path.abspath(path) == SELF:
        continue
    rel = os.path.relpath(path, ROOT)
    try:
        with open(path, 'rb') as f:
            toks = list(tokenize.tokenize(f.readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        hits.append('%s  [tokenize 失败: %s]' % (rel, e))
        continue
    prev = None
    for tok in toks:
        if tok.type == tokenize.NAME and tok.string in FORBID_IDS:
            # `_P` 只当**模块别名**（core.params）才算脏；`self._P` 等属性访问跳过
            if not (tok.string == '_P' and prev is not None
                    and prev.type == tokenize.OP and prev.string == '.'):
                hits.append('%s:%d  %s' % (rel, tok.start[0], tok.string))
        if tok.type not in _SKIP:
            prev = tok

for h in hits[:60]:
    print('  HIT ' + h)
if len(hits) > 60:
    print('  ...（共 %d 处，仅示前 60）' % len(hits))

C.group('主干代码 grep-0 旧框架符号（tokenize 只扫 NAME，跳过注释/字符串）', [
    ('零命中（实际 %d 处；见上 HIT 明细）' % len(hits), len(hits) == 0),
])

sys.exit(0 if C.summary() else 1)
