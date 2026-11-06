# -*- coding: utf-8 -*-
"""MCP 工具面与 SKILL 文档的一致性：工具名单不许两套账。

守三件事：

1. `mecha_v2/tools.py` 里 `define_tool(name=...)` 注册的名字，与
   `skills/energy-level/SKILL.md` 的「工具清单」表**集合相等**（另加宿主注入的
   `read_write_authority`）；
2. `.dsh/skills/energy-level/SKILL.md` 是权威文件的镜像是同一份内容（symlink 或逐字节）；
3. `re0-mecha` 只是别名指针，指向 `energy-level`，不维护第二份工具表。

这是文档与注册表的机械一致性检查，不评价措辞。
"""
import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.checklib import Checker

C = Checker('skill_mcp_contract')

TOOLS_PY = ROOT / 'mecha_v2' / 'tools.py'
SKILL = ROOT / 'skills' / 'energy-level' / 'SKILL.md'
MIRROR = ROOT / '.dsh' / 'skills' / 'energy-level' / 'SKILL.md'
ALIAS = ROOT / '.dsh' / 'skills' / 're0-mecha' / 'SKILL.md'


def registered_tools():
    tree = ast.parse(TOOLS_PY.read_text(encoding='utf-8'))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == 'define_tool'):
            continue
        for kw in node.keywords:
            if kw.arg == 'name' and isinstance(kw.value, ast.Constant):
                names.add(str(kw.value.value))
    return names


def skill_tools():
    text = SKILL.read_text(encoding='utf-8')
    start = text.index('## 2. 工具清单')
    end = text.index('## 3.', start)
    names = set()
    for line in text[start:end].splitlines():
        if not line.startswith('| `'):
            continue
        name = line.split('`', 2)[1]
        names.add(name)
    return names


def main():
    reg = registered_tools()
    doc = skill_tools()
    expected = reg | {'read_write_authority'}
    C.group('工具清单 = 注册表 + 宿主元工具', [
        ('注册数 >= 15（实际 %d）' % len(reg), len(reg) >= 15),
        ('文档缺：%s' % sorted(expected - doc), not (expected - doc)),
        ('文档多：%s' % sorted(doc - expected), not (doc - expected)),
        ('build_calib 在文档里', 'build_calib' in doc),
        ('read_write_authority 在文档里', 'read_write_authority' in doc),
    ])

    same = (MIRROR.exists()
            and (MIRROR.resolve() == SKILL.resolve()
                 or MIRROR.read_bytes() == SKILL.read_bytes()))
    C.check('dsh 镜像是同一份 SKILL', same)
    C.group('re0-mecha 只是别名指针', [
        ('文件存在', ALIAS.exists()),
        ('指向 energy-level', ALIAS.exists()
         and '../energy-level/SKILL.md' in ALIAS.read_text(encoding='utf-8')),
    ])
    return 0 if C.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
