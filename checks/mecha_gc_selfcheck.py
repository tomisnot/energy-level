# -*- coding: utf-8 -*-
"""数据治理（GC）自检（P4-1/P4-2）——保留策略 + 三源可达性 + 留痕 + 保守默认。

守用户定向的四条：**自动回收 + 默认保守 + 大声报 + 删除必须留痕**。

  A. **干跑优先**：默认 `dry_run=True` 只算不删（文件一个不少）。
  B. ★ **三源可达性**（主代理实测发现的缺口）：payloads 的引用者是
     **journal（含 archive/）∪ artifacts ∪ jobs**——**少扫 journal 就会删掉它仍指向的
     payload** ⇒ 驾驶舱看到坏指针（黑箱最不能出的错）。**致死性**：构造"仅被 journal
     引用"的 payload（数组指针 + 截断原文两种），GC 后**必须仍在**。
  C. **artifacts 保留口径 = OR**：`pinned` **或** 窗口内 **或** 最新 N 个 —— 三者任一即留。
     **致死性**：把某 aid 标 pinned ⇒ 必须从 plan 移到 kept。
  D. **保守不删**：journal 只归档不删除；`running` 的 job 永不删。
  E. **留痕**：真删落一条 `kind='gc'` 事件，含 plan/字节数/**触发者**（谁做的）。
  F. **归档顺序纪律**：journal 归档是**移动**；归档后其引用仍算可达（`archive/` 也在扫描源里）。

用临时 root（不污染 `<repo>/.mecha`）；不跑引擎（秒级）。
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from mecha import gc as G                              # noqa: E402
from mecha.bus import Bus                              # noqa: E402

C = Checker('mecha_gc')


def _mk(root):
    """造一份典型数据：journal（含两种 payload 引用）+ artifacts + jobs。"""
    b = Bus(root=root)
    b.append(actor='ai', kind='run', target='r1', after=np.arange(2000), reason='数组')
    b.append(actor='ai', kind='run', target='r2', after={'blob': 'x' * 20000},
             reason='超长文本')
    return b


# ---------------- A. 干跑优先 ----------------
_T = tempfile.mkdtemp(prefix='gc_A_')
_b = _mk(_T)
_pdir = os.path.join(_b.root, 'payloads')
_before = sorted(os.listdir(_pdir))
# 造一个孤儿 payload（三源都不引用）
with open(os.path.join(_pdir, 'orphan.json'), 'w', encoding='utf-8') as f:
    f.write('{}')
_rep = G.gc(_b.root, dry_run=True)
C.group('A 干跑优先：默认只算不删（文件一个不少）', [
    ('默认 dry_run=True', _rep['dry_run'] is True),
    ('计划里列出孤儿 payload',
     any(i.get('file') == 'orphan.json' for i in _rep['plan'])),
    ('**干跑后文件仍在**（一个都没删）',
     os.path.exists(os.path.join(_pdir, 'orphan.json'))),
    ('回执带 freed_bytes（"大声报"多少可回收）', _rep['freed_bytes'] >= 0),
])

# ---------------- B. ★ 三源可达性（含致死性） ----------------
_refs = G.referenced_payloads(_b.root)
_kept_files = {i.get('file') for i in _rep['kept'] if i['area'] == 'payloads'}
_plan_files = {i.get('file') for i in _rep['plan'] if i['area'] == 'payloads'}
C.group('B ★ 三源可达性：journal 引用的 payload **必须保留**（含致死性）', [
    ('从 journal 抽到引用（两种：$pointer.path 与 $truncated.where）',
     len(_refs) == 2),
    ('**仅被 journal 引用的 payload 在 kept 里**（不在 plan 里）',
     len(_kept_files) == 2 and not (_kept_files & _plan_files)),
    ('孤儿 payload 在 plan 里（三源都无引用）', 'orphan.json' in _plan_files),
    ('**致死性**：若少扫 journal 源，这两个 payload 会被判孤儿 ⇒ 本断言必红',
     len(_refs) >= 2),
    ('引用来源标注为 journal:*（可审计"谁引用它"）',
     all(any(s.startswith('journal:') for s in v)
         for v in _refs.values())),
])

# 真删一次，确认被 journal 引用的**真的还在**
_rep_b = G.gc(_b.root, dry_run=False, bus=_b, triggered_by='judge')
_after = sorted(os.listdir(_pdir))
C.group('B2 真删后：journal 引用的仍在、孤儿消失', [
    ('孤儿被删', 'orphan.json' not in _after),
    ('**journal 引用的两个仍在**（黑箱指针不会悬空）', len(_after) == 2),
])

# ---------------- C. artifacts 保留口径 = OR（含致死性） ----------------
_T2 = tempfile.mkdtemp(prefix='gc_C_')
_b2 = Bus(root=_T2)
_r2 = _b2.root
_adir = os.path.join(_r2, 'artifacts')


def _mk_aid(aid, pinned=False, age_days=0):
    d = os.path.join(_adir, aid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'aid': aid, 'pinned': pinned}, f)
    with open(os.path.join(d, 'result.npz'), 'wb') as f:
        f.write(b'x' * 100)
    if age_days:
        t = time.time() - age_days * 86400
        os.utime(d, (t, t))
        os.utime(os.path.join(d, 'manifest.json'), (t, t))
    return d


_mk_aid('old_pinned', pinned=True, age_days=999)     # 老但 pinned ⇒ 留
_mk_aid('old_plain', age_days=999)                    # 老且未 pinned ⇒ 清（若不在最新 N 内）
_mk_aid('fresh', age_days=0)                          # 窗口内 ⇒ 留
_rep_c = G.gc(_r2, dry_run=True, retention_days=90, artifact_keep=1)
_plan_aids = {i.get('aid') for i in _rep_c['plan'] if i['area'] == 'artifacts'}
_kept_aids = {i.get('aid') for i in _rep_c['kept'] if i['area'] == 'artifacts'}
C.group('C artifacts 保留口径 = **OR**（pinned / 窗口内 / 最新 N）含致死性', [
    ('老但 **pinned** ⇒ 留在 kept（不被清）', 'old_pinned' in _kept_aids),
    ('**致死性**：若漏判 pinned，old_pinned 会进 plan ⇒ 本断言必红',
     'old_pinned' not in _plan_aids),
    ('窗口内的 ⇒ 留', 'fresh' in _kept_aids),
    ('老、未 pinned、且不在最新 N 内 ⇒ 进 plan', 'old_plain' in _plan_aids),
    ('kept 里带原因（"pinned"/"在窗口内"/"最新 N 个之一"）',
     all(any(k in (i.get('why') or '') for k in ('pinned', '窗口内', '最新'))
         for i in _rep_c['kept'] if i['area'] == 'artifacts')),
])

# ---------------- D. 保守不删：journal 只归档、running job 永不删 ----------------
_T3 = tempfile.mkdtemp(prefix='gc_D_')
_b3 = Bus(root=_T3)
_r3 = _b3.root
# 造一个"老"的 journal 轮转段
_old_j = os.path.join(_r3, 'journal.20200101-000000.jsonl')
with open(_old_j, 'w', encoding='utf-8') as f:
    f.write('{"seq":1,"ts":0,"actor":"ai","kind":"set","target":"k","after":1}\n')
_t_old = time.time() - 999 * 86400
os.utime(_old_j, (_t_old, _t_old))
# 造一个 running 的 job
_jdir = os.path.join(_r3, 'jobs')
os.makedirs(_jdir, exist_ok=True)
with open(os.path.join(_jdir, 'j1.json'), 'w', encoding='utf-8') as f:
    json.dump({'id': 'j1', 'status': 'running'}, f)
_rep_d = G.gc(_r3, dry_run=False, bus=_b3, triggered_by='judge', retention_days=90)
C.group('D 保守不删：journal 只归档、running job 永不删', [
    ('老 journal 段被**归档**（移动，不是删除）',
     any(a['from'].endswith('journal.20200101-000000.jsonl')
         for a in _rep_d['archive'])),
    ('归档后文件在 `archive/` 下**仍存在**（可追史 T5）',
     os.path.exists(os.path.join(_r3, 'archive', 'journal.20200101-000000.jsonl'))),
    ('**running 的 job 仍在**（诚实对账要保留）',
     os.path.exists(os.path.join(_jdir, 'j1.json'))),
    ('running 的 job 在 kept 里且写明原因',
     any(i['area'] == 'jobs' and 'running' in (i.get('why') or '')
         for i in _rep_d['kept'])),
    ('journal 整体标为"只归档、不删除"',
     any(i['area'] == 'journal' and '只归档' in (i.get('why') or '')
         for i in _rep_d['kept'])),
])

# ---------------- E. 留痕：kind='gc' 事件含 plan 与**触发者** ----------------
_gc_evs = [e for e in _b3.since(0) if e.get('kind') == 'gc']
C.group('E 留痕：真删落 `kind=gc` 事件，含 plan/字节数/**触发者**（M4 的反面）', [
    ('有 gc 事件（删除必须可审计）', len(_gc_evs) == 1),
    ('actor=maintenance', _gc_evs and _gc_evs[-1].get('actor') == 'maintenance'),
    ('after 带 n_removed / freed_bytes / areas',
     all(k in (_gc_evs[-1].get('after') or {})
         for k in ('n_removed', 'freed_bytes', 'areas')) if _gc_evs else False),
    ('**触发者记进 reason**（"谁做的"，不只是"发生了一次回收"）',
     'judge' in str(_gc_evs[-1].get('reason') or '') if _gc_evs else False),
    ('触发者也记进 provenance（结构化）',
     (_gc_evs[-1].get('provenance') or {}).get('triggered_by') == 'judge'
     if _gc_evs else False),
    ('reversibility=none（删除不可撤销，如实标）',
     _gc_evs[-1].get('reversibility') == 'none' if _gc_evs else False),
])

# ---------------- F. 归档段仍算引用源（顺序纪律） ----------------
_T4 = tempfile.mkdtemp(prefix='gc_F_')
_b4 = Bus(root=_T4)
_b4.append(actor='ai', kind='run', target='r', after=np.arange(500), reason='arr')
_r4 = _b4.root
# 把 live journal 挪成"老归档段"（模拟已归档），payload 只被**归档段**引用
_live = os.path.join(_r4, 'journal.jsonl')
_arch = os.path.join(_r4, 'archive')
os.makedirs(_arch, exist_ok=True)
_shutil_dst = os.path.join(_arch, 'journal.20200101-000000.jsonl')
os.replace(_live, _shutil_dst)
_t_old2 = time.time() - 999 * 86400
os.utime(_shutil_dst, (_t_old2, _t_old2))
_p4 = os.path.join(_r4, 'payloads')
_rep_f = G.gc(_r4, dry_run=False, bus=_b4, triggered_by='judge')
C.group('F 归档段也是引用源：`archive/` 里的 journal 引用**仍算可达**', [
    ('归档段被扫到（referenced_payloads 认 archive/）',
     any('journal' in s for v in G.referenced_payloads(_r4).values() for s in v)),
    ('**该 payload 未被删**（否则归档史里的指针悬空）',
     len(os.listdir(_p4)) == 1),
    ('它出现在 kept 且原因为"被引用"',
     any(i['area'] == 'payloads' and '被引用' in (i.get('why') or '')
         for i in _rep_f['kept'])),
])

sys.exit(0 if C.summary() else 1)
