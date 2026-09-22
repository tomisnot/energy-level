# -*- coding: utf-8 -*-
"""驾驶舱读**归档段**自检 —— GC 归档后**不丢史**（T5 黑匣子完整性）。

**背景（主代理发现的真回归）**：归档有两个来源，落点不同——
  · `bus._maybe_rotate`（体积超限）→ 归档写在**根目录** `journal.<stamp>.jsonl`；
  · `mecha.gc`（保留策略）→ 归档**移动**到 `<root>/archive/`。
`cockpit.backend.journal_files` 旧版**只 glob 根目录**，却在自己的 docstring 里声称
"含归档、不丢史"——**代码与文档互相矛盾**。后果：**跑一次 `gc`，驾驶舱静默丢掉那段历史**
（文件还在，黑匣子看不见）⇒ T5"死前全貌"缺一块。

本判据守四件事：
  A. `journal_files` 同时返回 live 与 `archive/` 两处，且带 `archived` 标记；
  B. `read_events` 合并后**仍按 seq 有序**（归档段 seq 更小 ⇒ 必须归并排序，
     否则 `diff-since-seq` 语义被破坏）；
  C. 端到端：**归档一段 journal → 跑 gc 把它移进 archive/ → 驾驶舱仍读得到那些事件**；
  D. **致死性**：把 `archive/` 从 glob 里去掉 ⇒ C 必须变红（那正是当前的 bug）。

用临时 root（不污染 `<repo>/.mecha`）；不跑引擎（秒级）。
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from cockpit import backend                            # noqa: E402
from mecha import gc as G                              # noqa: E402
from mecha.bus import Bus                              # noqa: E402

C = Checker('cockpit_archive')

# ---- 造数据 ----
# ⚠ 顺序很重要：**先建 Bus 并写 live 段，再把"老段"挪成归档段**。
#   若先放老段再建 Bus，Bus 的 `_load` 会把它读进内存并据此续 seq，
#   于是"live 段"与"归档段"的 seq 会重叠（写出的夹具反而制造了假故障）。
_TMP = tempfile.mkdtemp(prefix='ck_arch_')
_b = Bus(root=_TMP)
_root = _b.root
for i in (1, 2):                      # live 段：seq 1,2（由 Bus 自己分配）
    _b.append(actor='ai', kind='set', target='k%d' % i, after=i, reason='live 段')
_live_seqs = {int(e['seq']) for e in _b.since(0)}
# 再把一份"老"轮转段放进根目录（模拟 bus 体积轮转的产物），随后由 gc 归档
_old = os.path.join(_root, 'journal.20200101-000000.jsonl')
with open(_old, 'w', encoding='utf-8') as f:
    for i in (101, 102):              # 用明显更大的 seq 区间，避免与 live 混淆
        f.write(json.dumps({'seq': i, 'ts': 0.0, 'actor': 'ai', 'kind': 'set',
                            'target': 'old%d' % i, 'after': i,
                            'reason': '旧段'}) + '\n')
_t_old = time.time() - 999 * 86400
os.utime(_old, (_t_old, _t_old))
_arch_seqs = {101, 102}
# **先跑一次 gc 把它归档进 archive/**：A/B 两组要验的正是"归档段已被读到"，
# 故必须在归档**之后**再断言（否则文件还在根目录，archived 标记自然全是 False）。
_rep0 = G.gc(_root, dry_run=False, triggered_by='judge-fixture',
             retention_days=90)
assert _rep0['archive'], '夹具失败：gc 未归档旧段'

# ---------------- A. journal_files 含两处 + archived 标记 ----------------
_files = backend.journal_files(_root)
_live_n = sum(1 for _f, a in _files if not a)
_arch_n = sum(1 for _f, a in _files if a)
C.group('A journal_files：live + archive/ 两处都返回，且带 `archived` 标记', [
    ('返回 (路径, archived) 对（%d 个）' % len(_files),
     all(isinstance(x, tuple) and len(x) == 2 for x in _files)),
    ('含 live 段（%d）' % _live_n, _live_n >= 1),
    ('**含 archive/ 段**（%d）——旧版只 glob 根目录，这里必红' % _arch_n,
     _arch_n >= 1),
    ('archived 标记能区分两段（不把归档误当 live）',
     _arch_n >= 1 and _live_n >= 1),
])

# ---------------- B. read_events 合并后按 seq 有序 ----------------
_evs = backend.read_events(_root, 0)
_seqs = [int(e.get('seq', 0)) for e in _evs]
C.group('B read_events 合并两源后**仍按 seq 有序**（diff-since-seq 语义不被破坏）', [
    ('读到两段的事件（%d 条）' % len(_evs), len(_evs) >= 4),
    ('seq 严格递增（归并排序生效）', _seqs == sorted(_seqs) and len(set(_seqs)) == len(_seqs)),
    ('归档段事件带 `archived=True`（前端可区分）',
     any(e.get('archived') is True for e in _evs)),
    ('live 段事件带 `archived=False`',
     any(e.get('archived') is False for e in _evs)),
    ('diff-since-seq 仍只回增量（since=100 → 只剩归档段 seq>100）',
     {int(e['seq']) for e in backend.read_events(_root, 100)} == _arch_seqs),
])

# ---------------- C. 端到端：跑 gc 归档后**仍读得到**（T5） ----------------
# 归档动作已在夹具里发生（`_rep0`）；这里断言"归档后驾驶舱仍读得到"，
# 并顺带验**幂等**：再跑一次 gc 不应重复归档（旧段已在 archive/）。
_before_seqs = {int(e['seq']) for e in _b.since(0)} | _arch_seqs
_after_evs = backend.read_events(_root, 0)
_after_seqs = {int(e['seq']) for e in _after_evs}
_rep2 = G.gc(_root, dry_run=False, bus=_b, triggered_by='judge', retention_days=90)
C.group('C 端到端：**gc 归档一段 → 驾驶舱仍读得到**（T5）＋ 幂等', [
    ('gc 确实把旧段归档了（夹具那次 archive 计划非空）', len(_rep0['archive']) >= 1),
    ('旧段文件确实在 `archive/` 下',
     os.path.exists(os.path.join(_root, 'archive',
                                 'journal.20200101-000000.jsonl'))),
    ('**归档段的 seq 仍读得到**（只移动、不丢史）', _arch_seqs <= _after_seqs),
    ('归档后这些事件被标 `archived=True`（明确显示这是归档段）',
     any(e.get('archived') is True and int(e['seq']) in _arch_seqs
         for e in _after_evs)),
    ('**幂等**：再跑一次 gc 不重复归档（旧段已在 archive/）',
     not _rep2['archive']),
])

# ---------------- D. 致死性：去掉 archive/ 的 glob ⇒ C 必红 ----------------
# 先**重取基线**：C 里那次 gc 又追加了一条 gc 事件（seq 变了），
# 故"还原后照旧"要拿**当前**读数比，不能拿 C 之前的旧快照比。
_baseline_seqs = {int(e['seq']) for e in backend.read_events(_root, 0)}
_orig_glob = backend.glob.glob


def _no_archive_glob(pat, *a, **k):
    """变异：把 archive/ 的 glob 掐掉（= 旧版行为）。"""
    if 'archive' in str(pat).replace('\\', '/'):
        return []
    return _orig_glob(pat, *a, **k)


backend.glob.glob = _no_archive_glob
try:
    _mut_evs = backend.read_events(_root, 0)
    _mut_seqs = {int(e['seq']) for e in _mut_evs}
    _mut_files = backend.journal_files(_root)
finally:
    backend.glob.glob = _orig_glob

C.group('D **致死性**：去掉 archive/ 的 glob（= 旧版）⇒ 归档史立刻读不到', [
    ('变异后**读不到**归档段的 seq（证明本判据真在守这件事）',
     not (_arch_seqs <= _mut_seqs)),
    ('变异后事件数变少（丢史）', len(_mut_seqs) < len(_baseline_seqs)),
    ('变异后 journal_files 不含 archived=True 的段',
     not any(a for _f, a in _mut_files)),
    ('还原后一切照旧（无残留副作用）',
     {int(e['seq']) for e in backend.read_events(_root, 0)} == _baseline_seqs),
])

sys.exit(0 if C.summary() else 1)
