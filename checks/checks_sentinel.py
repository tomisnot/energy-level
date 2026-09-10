# -*- coding: utf-8 -*-
"""checks 存活哨兵（t78）：把"脚本静默腐烂"变成一次可见的失败。

背景：2026-09-10 清理时发现 5 个 check 脚本早已因 `src/core` 重构后的旧导入
路径（`src.core.evolve`、`src.pulse` 等）而**跑不起来**，却没有任何机制提醒。
另有一次归档把 `mw_bench_firstprinciples.py`（其实是 `ui_selftest` 的隐式
同目录依赖）误判为独立脚本，回归当场挂掉——说明依赖关系无人看守。

本哨兵做四件事（纯静态，秒级）：
  0. **兼容导出不遮蔽子包**：`src.<子包>` 属性必须就是该子包模块（t78 教训：
     `src/__init__.py` 的 `from .core import core` 曾把 `src.core` 覆盖成
     `static_atom`，使 `import src.core.x as y` 报错指向 static_atom）；
  1. 语法：每个脚本必须可 compile；
  2. 导入可解析：脚本里出现的 `src.*` 绝对导入必须有对应模块（含 src.core
     的兼容 shim）；
  3. **同目录隐式依赖**：脚本若裸导入 checks/ 里的同级模块名（`import mw_bench_
     firstprinciples` 这种），登记出来——归档/改名这些模块前必须查这份清单。

退出码 = 不可解析导入的脚本数（0 = 通过）。用法：python checks\\checks_sentinel.py
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _mod_exists(dotted):
    """dotted（如 src.core.hard）是否为可导入模块/包。"""
    rel = dotted.replace('.', os.sep)
    if os.path.exists(os.path.join(_ROOT, rel + '.py')):
        return True
    if os.path.isdir(os.path.join(_ROOT, rel)):
        return True
    # src.core 的兼容 shim（旧名 → 新模块，见 src/core/__init__.py）
    parts = dotted.split('.')
    if len(parts) == 3 and parts[:2] == ['src', 'core']:
        try:
            import src.core as _c
            if hasattr(_c, parts[2]):
                return True
        except Exception:                                       # noqa: BLE001
            pass
    return False


def _scan(path):
    """→ (语法错, 不可解析导入列表, 同目录隐式依赖列表)。"""
    src = open(path, encoding='utf-8').read()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return f'{e.__class__.__name__}: {e.msg}（line {e.lineno}）', [], []
    bad, sib = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # 相对导入（包内）→ 跳过
                continue
            names = [node.module] if node.module else []
        else:
            continue
        for n in names:
            if not n:
                continue
            if n.split('.')[0] == 'src' and not _mod_exists(n):
                bad.append(n)
            elif '.' not in n and os.path.isfile(
                    os.path.join(_HERE, n + '.py')):
                sib.append(n)                    # checks/ 内的同级模块
    return None, sorted(set(bad)), sorted(set(sib))


def _shadowed_subpackages():
    """兼容导出不得遮蔽同名子包（t78 教训）。

    `src/__init__.py` 曾用 `from .core import core` 把 `src.core` 这个**属性**
    覆盖成 `static_atom`，导致 `import src.core.<x> as <别名>` 报错指向
    static_atom（CPython 对带 as 的点号导入走父模块属性遍历）。
    不变量：`src.<子包>` 属性必须就是 `sys.modules['src.<子包>']`。
    """
    import src
    bad = []
    for name in ('core', 'ui'):
        mod = sys.modules.get(f'src.{name}')
        if mod is None:
            continue
        if getattr(src, name, None) is not mod:
            bad.append(f'{name} → {getattr(getattr(src, name, None), "__name__", "?")}')
    return bad


def main():
    bad_shadow = _shadowed_subpackages()
    if bad_shadow:
        print(f'[checks_sentinel]  [FAIL] src 的兼容导出遮蔽了子包：{bad_shadow}'
              f'  ← 会让 import src.<子包>.<模块> as <别名> 报错')
    else:
        print('[checks_sentinel]  [ ok ] 兼容导出未遮蔽子包（src.core / src.ui）')
    files = sorted(f for f in os.listdir(_HERE)
                   if f.endswith('.py') and os.path.isfile(
                       os.path.join(_HERE, f)))
    n_bad = len(bad_shadow)
    print(f'[checks_sentinel] 扫描 {len(files)} 个脚本（checks/_archive/ 除外）')
    for f in files:
        err, bad, sib = _scan(os.path.join(_HERE, f))
        if err:
            n_bad += 1
            print(f'  [FAIL] {f}  语法/解析错误：{err}')
            continue
        if bad:
            n_bad += 1
            print(f'  [FAIL] {f}  不可解析导入：{bad}')
            continue
        if sib:
            print(f'  [note] {f}  同目录隐式依赖：{sib}'
                  f'  ← 归档/改名这些模块前先查这里')
        else:
            print(f'  [ ok ] {f}')
    if n_bad:
        print(f'[checks_sentinel]  FAIL={n_bad} —— 有脚本已腐烂，需要修或归档')
        return n_bad
    print('[checks_sentinel]  全部可解析 ✅（同目录隐式依赖已列出，非错误）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
