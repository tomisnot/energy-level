# -*- coding: utf-8 -*-
"""代码新鲜度（T3 / 待办 A12）：磁盘上的 `src/` 是否比**本进程加载的代码**更新。

**为什么需要它**（G4 / 待办 §A12）：引擎与服务端都是**进程内**代码——改了
`src/` 不重启，AI 看到的仍是旧行为，而且毫无提示（本会话因此白跑过整轮：
`hasattr(evolve_mixed,'_H_SNAPSHOT_MAX_MB')==False` 但源码里明明有）。

**判据只依赖文件 mtime + 一个进程基准时刻**：
    stale = 存在 src/**/*.py（排除 __pycache__/_archive）的 mtime > 基准时刻

基准时刻（`loaded_at`）语义：**本进程第一次用到本模块的时刻**。GUI 里由
`src/ai/server.py` 在模块导入时（≈ 启动 `_init_ai`）先盖一次戳 → 就是进程启动
时刻；脚本/自检里若晚导入，仍可用 `loaded_at=` 显式传基准（自检与演示都这么用）。

输出（`code_status()`，全部 JSON 友好）：
    loaded_at / scanned_at / n_files / n_stale / stale / age_s
    src_core_mtime / src_ai_mtime / src_ui_mtime / src_other_mtime / app_mtime
    newest = [{path, mtime, stale}, …]（最多 5 个，按 mtime 降序）
    hint    = 一句话结论（给人/AI 看的"要不要重启"）
"""
import os
import time

_LOADED_AT = None          # 进程基准（first-wins；server.py 启动时盖戳）
_SELF_IMPORT_AT = time.time()
_CACHE = {'t': 0.0, 'd': None}
_CACHE_TTL = 2.0           # 秒：脚本里连调 summary() 时别反复扫盘

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# 分组：路径片段 → 输出键（保证 src_core_mtime / src_ai_mtime 一定存在）
_GROUPS = (('src_core_mtime', (os.path.join('src', 'core'),)),
           ('src_ai_mtime', (os.path.join('src', 'ai'),)),
           ('src_ui_mtime', (os.path.join('src', 'ui'),)),
           ('app_mtime', ('app_entry.py', 'main.py')))
_SKIP_DIRS = ('__pycache__', '_archive', '.git')


def mark_loaded(ts=None):
    """盖进程基准戳（**first-wins**）：由 server.py 在启动导入时调用。

    幂等且只认第一次——否则"进程运行中第一次调 code_status"会把基准推到
    编辑之后，stale 永远为 False（假阴性，比没有更坏）。
    """
    global _LOADED_AT
    if _LOADED_AT is None:
        _LOADED_AT = float(ts if ts is not None else time.time())
    return _LOADED_AT


def baseline():
    """当前进程基准时刻（mark_loaded 未盖戳时用本模块导入时刻）。"""
    return _LOADED_AT if _LOADED_AT is not None else _SELF_IMPORT_AT


def _iter_py(root):
    """只扫**进程内真正加载的代码**：`src/**/*.py` + 顶层 `app_entry.py`/`main.py`。

    ⚠ 刻意不扫 `checks/`、`scripts/tmp/`、`runtime/`、`save_state/` 等——改那些文件**不影响**
    正在跑的进程，若算作 stale 就是**假警报**（实测踩过：扫全仓库时 400 个 .py，
    刚写完一个自检就报 stale）。
    """
    tops = [(os.path.join(root, 'src'), 'src')]
    for fn in ('app_entry.py', 'main.py'):
        p = os.path.join(root, fn)
        if os.path.isfile(p):
            yield p, fn
    for base, label in tops:
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                if fn.endswith('.py'):
                    p = os.path.join(dirpath, fn)
                    yield p, os.path.relpath(p, root).replace('\\', '/')


def _rel_to(path, prefix):
    p = path.replace('\\', '/')
    pre = prefix.replace('\\', '/').rstrip('/')
    if pre in ('app_entry.py', 'main.py'):
        return p == pre or p.endswith('/' + pre)
    return p == pre or p.startswith(pre + '/')


def code_status(root=None, loaded_at=None, use_cache=True):
    """磁盘代码 vs 进程基准 → 状态 dict（见模块 docstring）。"""
    root = os.path.abspath(root or _ROOT)
    base = float(loaded_at) if loaded_at is not None else baseline()
    if use_cache and loaded_at is None:
        now = time.time()
        if _CACHE['d'] is not None and now - _CACHE['t'] < _CACHE_TTL:
            return _CACHE['d']
    items, groups = [], {k: None for k, _p in _GROUPS}
    groups['src_other_mtime'] = None
    for p, rel in _iter_py(root):
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        items.append((mt, rel))
        for key, prefixes in _GROUPS:
            if any(_rel_to(rel, pre) for pre in prefixes):
                groups[key] = max(groups[key] or 0.0, mt)
        if rel.startswith('src/') and not any(
                _rel_to(rel, os.path.join('src', g))
                for g in ('core', 'ai', 'ui')):
            groups['src_other_mtime'] = max(groups['src_other_mtime'] or 0.0,
                                            mt)
    items.sort(reverse=True)
    stale_items = [(mt, rel) for (mt, rel) in items if mt > base + 1e-6]
    newest = [{'path': rel, 'mtime': round(mt, 3), 'stale': mt > base + 1e-6}
              for (mt, rel) in items[:5]]
    age = (items[0][0] - base) if (items and items[0][0] > base) else 0.0
    d = {'loaded_at': round(base, 3), 'scanned_at': round(time.time(), 3),
         'n_files': len(items), 'n_stale': len(stale_items),
         'stale': bool(stale_items), 'age_s': round(age, 1),
         'newest': newest,
         'hint': ('⚠ 代码已陈旧：src/ 里有 %d 个文件比本进程加载的代码新'
                  '（最新 %s，新 %.0f s）——引擎/服务端是**进程内**代码，'
                  '**需要重启软件**才生效；当前结果仍来自旧代码。'
                  % (len(stale_items), stale_items[0][1], age)
                  if stale_items else
                  '代码与进程一致（无需重启）')}
    d.update(groups)
    if use_cache and loaded_at is None:
        _CACHE.update(t=time.time(), d=d)
    return d


def stale_warning(root=None, loaded_at=None):
    """陈旧时给一句话；新鲜时 None（run_python 的 `stale_code_warning` 用它）。"""
    try:
        d = code_status(root=root, loaded_at=loaded_at, use_cache=False)
    except Exception:                                   # noqa: BLE001
        return None
    return d['hint'] if d.get('stale') else None
