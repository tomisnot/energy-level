# -*- coding: utf-8 -*-
"""只读存档面（mecha-session-v1 JSON 快照）——补"AI 完全够不到存档"的缺口（**只读**）。

存档的**写**归人/GUI（`AppBridge.save`，M5：存档属人管）；本模块只提供**列出/读取**，
让 AI 能"读某个存档的参数 → 经 `session.set_many` 复原接着做实验"，而不给 AI 直接写档/
落盘的能力。格式 = `AppBridge.save` 的 mecha-session-v1：
`{'format','mode','params':{扁平 registry 键→值}}`。

目录默认 `<cwd>/save_state`（app/判据都从仓库根跑）；base 给了用 `<base>/save_state`。
"""
import json
import os

__all__ = ['DEFAULT_DIR', 'archives_dir', 'list_archives', 'read_archive']

DEFAULT_DIR = 'save_state'


def archives_dir(base=None):
    """存档目录（默认 <cwd>/save_state）。"""
    return os.path.join(base or os.getcwd(), DEFAULT_DIR)


def _resolve(name, base=None):
    """name 可为 'freeze' / 'freeze.json' / 绝对路径 → 绝对路径。

    相对名限存档目录内（防越界）；绝对路径须 .json。
    """
    d = os.path.abspath(archives_dir(base))
    if os.path.isabs(str(name)):
        p = os.path.abspath(str(name))
    else:
        p = os.path.abspath(os.path.join(d, str(name)))
        if os.path.commonpath([p, d]) != d:
            raise ValueError('存档名越界（限存档目录内）：%r' % name)
    if not p.endswith('.json'):
        p += '.json'
    return p


def list_archives(base=None):
    """→ {'ok,dir,archives:[{name,n_keys,mode}]}（按名排序；目录不存在→空+提示）。"""
    d = archives_dir(base)
    if not os.path.isdir(d):
        return {'ok': True, 'dir': d, 'archives': [],
                'note': '存档目录不存在：%s' % d}
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.json'):
            continue
        try:
            with open(os.path.join(d, fn), encoding='utf-8') as f:
                snap = json.load(f)
            out.append({'name': fn[:-5],
                        'n_keys': len((snap or {}).get('params') or {}),
                        'mode': (snap or {}).get('mode')})
        except Exception:                                # noqa: BLE001
            out.append({'name': fn[:-5], 'n_keys': None, 'mode': None,
                        'error': '读失败（非 JSON 或损坏）'})
    return {'ok': True, 'dir': d, 'archives': out}


def read_archive(name, base=None):
    """读存档 → {'ok,name,mode,params'}（**只读、不 apply**；复原请 session.set_many）。"""
    p = _resolve(name, base)
    if not os.path.exists(p):
        raise FileNotFoundError(
            '存档不存在：%r（先 list_archives 看有哪些）' % os.path.basename(p))
    with open(p, encoding='utf-8') as f:
        snap = json.load(f)
    if not isinstance(snap, dict) or 'params' not in snap:
        raise ValueError('不是 mecha-session-v1 存档（缺 params）：%r' % p)
    return {'ok': True, 'name': os.path.basename(p)[:-5], 'mode': snap.get('mode'),
            'params': snap['params'],
            'note': '只读快照；复原请 session.set_many(params, reason=...)'}
