# -*- coding: utf-8 -*-
"""自检：**journal 的 detail 体积闸**（2026-09-14 用户裁决"巨型 detail 瘦身 + 修写入侧"）。

来历（不是假想）：`set_param('dynamic.waves_A')` 的 detail.old/new 曾带**整条波形
包络**，实测**单条 748 KB**；`_journal/2026-09-10.jsonl` 一天写到 **37 MB**，
审计文件变成不可读的 blob 堆。现在 `src/ai/journal.py` 的 `Op` 在 record 时就把
detail 过 `slim_detail()`；本自检是它的闸——**"以后不再产生 MB 级记录"必须可验证**。

覆盖
  A `slim_detail` 纯函数：巨型 detail → ≤ 上限，且**键名/长度/sha1** 都留着
    （审计靠 sha1 比对；顶层标量键 `path`/`job` 必须原样，GUI 状态栏与 AI 归因靠它）
  B 未超限的 detail **原对象返回**（不加任何包装，行为与改前逐位一致）
  C `OpJournal.record` → 落盘单行 ≤ 上限 + 固定开销；内存里的 op.detail 同样已瘦
  D 仓库现有 `_journal/*.jsonl`：每行都合法 JSON 且 ≤ 阈值（历史迁移的回归闸）
  E 幂等：瘦过的 detail 再瘦一次**不再变**（迁移脚本可重跑）
  F 非 dict detail（超长字符串 / 巨型列表）也能收口
  G **undo 快照不受影响**：detail 瘦身绝不碰 `snapshot`（撤销的绝对正确性优先）

用法：python checks/journal_size_cap_selfcheck.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                                  # noqa: E402
from src.ai.journal import (OpJournal, slim_detail, _json_bytes,     # noqa: E402
                            _MAX_DETAIL_BYTES)

C = Checker('journal_size_cap')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = _MAX_DETAIL_BYTES
SLACK = 2048              # 单行里除 detail 外的固定开销（seq/ts/actor/kind/reason…）


def _huge_detail():
    """仿真实的 `set_param('dynamic.waves_A')` detail：两条波形各带 10 万点包络。"""
    env = ['file', [0.0] * 100000]
    w = {'name': 's=0.05 tc=0.5', 'freq_mhz': 295724187.476,
         'intensity': 100605857.52753377, 'pol': 'π', 'offset_ns': 0.0,
         'env': env}
    return {'path': 'dynamic.waves_A', 'old': {'n': 1, 'head': [dict(w)]},
            'new': {'n': 1, 'head': [dict(w)]}}


print('A) slim_detail 纯函数：巨型 detail → 摘要（键名/长度/sha1 都要留）')
big = _huge_detail()
n_big = _json_bytes(big)
slim, meta = slim_detail(big)
C.check('A1 巨型 detail 本身确实超标（前提成立）', n_big > CAP * 50,
        '%d bytes' % n_big)
C.check('A2 摘要 ≤ 上限', _json_bytes(slim) <= CAP,
        '%d bytes' % _json_bytes(slim))
C.check('A3 meta 记原始字节数 + sha1 + 键名清单',
        isinstance(meta, dict) and meta.get('orig_bytes') == n_big
        and len(meta.get('sha1') or '') == 16
        and meta.get('keys') == ['path', 'old', 'new'], str(meta)[:160])
C.check('A4 顶层标量键 path 原样保留（GUI 状态栏 / AI 归因靠它）',
        slim.get('path') == 'dynamic.waves_A')
C.check('A5 巨型序列降级为 {形状 + 长度 + 前 8 项}',
        slim['old']['head'][0]['env'][0] == 'file'
        and slim['old']['head'][0]['env'][1].get('__slim__') == 'seq'
        and slim['old']['head'][0]['env'][1]['n'] == 100000
        and len(slim['old']['head'][0]['env'][1]['head']) == 8,
        str(slim['old']['head'][0]['env'])[:120])
C.check('A6 摘要体积至少缩小 50 倍', n_big / max(1, _json_bytes(slim)) > 50,
        '%d → %d' % (n_big, _json_bytes(slim)))

print('\nB) 未超限 → 原对象返回（与改前逐位一致）')
small = {'path': 'field.B_gauss', 'old': 21.7, 'new': 22.0}
s2, m2 = slim_detail(small)
C.check('B1 小 detail 原样（同一对象、meta=None）', s2 is small and m2 is None)
tiny = {'path': 'x', 'note': 'y' * (CAP - 200)}
C.check('B2 恰好在上限内 → 不动', slim_detail(tiny)[0] is tiny,
        '%d bytes' % _json_bytes(tiny))

print('\nC) OpJournal.record：落盘单行与内存 detail 都收口')
tmp = tempfile.mkdtemp(prefix='_jrnl_cap_')
try:
    j = OpJournal(tmp)
    op = j.record('ai', 'set_param', _huge_detail(), reason='体积闸自检')
    f = os.path.join(tmp, '_journal')
    lines = []
    for nm in os.listdir(f):
        with io.open(os.path.join(f, nm), encoding='utf-8') as fh:
            lines += [x for x in fh if x.strip()]
    rec = json.loads(lines[-1])
    C.check('C1 落盘单行 ≤ 上限 + 固定开销（%d）' % (CAP + SLACK),
            len(lines[-1].encode('utf-8')) <= CAP + SLACK,
            '%d bytes' % len(lines[-1].encode('utf-8')))
    C.check('C2 记录带 detail_slim（原始体积 + sha1 可回溯）',
            isinstance(rec.get('detail_slim'), dict)
            and rec['detail_slim'].get('orig_bytes') == n_big)
    C.check('C3 落盘 detail 仍是 dict 且 path 可读（GUI 那条链路）',
            rec['detail'].get('path') == 'dynamic.waves_A')
    C.check('C4 内存 op.detail 同样已瘦（不然 200 条 op 会吃 GB）',
            _json_bytes(op.detail) <= CAP)
    C.check('C5 get_activity 视图（since）带同名摘要字段',
            (j.since(0)[-1].get('detail') or {}).get('path') == 'dynamic.waves_A')
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('\nD) 仓库现有 _journal/*.jsonl：行合法 + 单行 ≤ 16 KB（历史迁移回归闸）')
jd = os.path.join(ROOT, '_journal')
worst, nlines, bad = 0, 0, 0
if os.path.isdir(jd):
    for nm in sorted(os.listdir(jd)):
        if not nm.endswith('.jsonl'):
            continue
        with io.open(os.path.join(jd, nm), encoding='utf-8') as fh:
            for x in fh:
                if not x.strip():
                    continue
                nlines += 1
                worst = max(worst, len(x.encode('utf-8')))
                try:
                    json.loads(x)
                except Exception:                          # noqa: BLE001
                    bad += 1
C.check('D1 每行都是合法 JSON（无损坏）', bad == 0, 'bad=%d' % bad)
C.check('D2 最大单行 ≤ 16 KB', worst <= 16 * 1024,
        'max=%d bytes（%d 行）' % (worst, nlines))
C.check('D3 记录数仍是 5 位数量级（迁移没丢条目）', nlines > 2000,
        'n=%d' % nlines)

print('\nE) 幂等：瘦过的 detail 再瘦一次不再变（迁移脚本可重跑）')
again, m3 = slim_detail(slim)
C.check('E1 二次 slim 无变化、meta=None', _json_bytes(again) == _json_bytes(slim)
        and m3 is None, '%d vs %d' % (_json_bytes(again), _json_bytes(slim)))

print('\nF) 非 dict detail 也收口')
S = slim_detail('x' * 500000)
L = slim_detail([{'k': i, 'v': 'y' * 100} for i in range(5000)])
C.check('F1 超长字符串 → 摘要 ≤ 上限且有 n_chars/head',
        _json_bytes(S[0]) <= CAP and S[1] and 'n_chars' in S[0],
        str(S[0])[:100])
C.check('F2 巨型列表 → 摘要 ≤ 上限且有 n/head',
        _json_bytes(L[0]) <= CAP and L[1] and L[0].get('__slim__') == 'seq',
        '%d bytes' % _json_bytes(L[0]))

print('\nG) undo 快照不受 detail 瘦身影响（撤销的正确性是第一优先）')
tmp = tempfile.mkdtemp(prefix='_jrnl_snap_')
try:
    j = OpJournal(tmp)
    snap = {'params': {'dynamic.waves_A': {'n': 1, 'head': [{'env':
            ['file', [1.0] * 50000]}]}}, 'meta': {'version': 3}}
    j.record('ai', 'set_param', _huge_detail(), reason='快照自检', snapshot=snap)
    seq, got = j.last_snapshot('ai')
    C.check('G1 快照逐项未改动（同尺寸、内容可读）',
            got is snap and _json_bytes(got) == _json_bytes(snap)
            and len(got['params']['dynamic.waves_A']['head'][0]['env'][1])
            == 50000, '%d bytes' % _json_bytes(got or {}))
    C.check('G2 undo_last 仍能弹回该快照', j.undo_last('ai')[1] is snap)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

ok = C.summary()
sys.exit(0 if ok else 1)
