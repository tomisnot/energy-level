# -*- coding: utf-8 -*-
"""操作日志（OpJournal）：每次状态变更记一条带来源的 Op，支持撤销。

原则 P3（可逆与可归因）：每个 op = {seq, ts, actor, kind, path, before,
after, reason}。journal 同时是三样东西的数据源：
  - GUI"AI 动态"面板（P3 协同 UI）
  - MCP get_activity 工具（人/AI 都可读）
  - 落盘审计 save_state/_journal/<日期>.jsonl

undo 采用"整 op 快照回滚"：op 记录变更前的可序列化状态（before 为
SimState.to_dict 的浅快照），撤销即 restore 回那个快照。粒度粗但绝对
正确，避免字段级合并的复杂度（草案 §2.2 Phase 2 约定）。

**detail 体积上限（2026-09-14 用户裁决："巨型 detail 瘦身 + 修写入侧"）**：
历史教训——`set_param('dynamic.waves_A')` 的 detail 里 old/new 带着**整条波形
包络**（实测单条 **748 KB**），09-10 一天就写掉 37 MB。现在 `record()` 一律先过
`slim_detail()`：超过 `_MAX_DETAIL_BYTES` 的值只留**形状/键名/长度/前缀 + 原始
字节数与 sha1**（可审计、可在 git/存档里找回），顶层标量键（`path`/`job`…）原样
保留（GUI 状态栏与 AI 归因都靠它们）。自检
`checks/journal_size_cap_selfcheck.py`；历史文件用 `scripts/slim_journal.py` 迁移。
"""
import hashlib
import json
import os
import time

_MAX_HISTORY = 200          # 内存中保留的可撤销 op 数
_MAX_DETAIL_BYTES = 4096    # 单条 detail 上限（内存 + 落盘同一条闸）
_SLIM_HEAD = 8              # 摘要保留的序列前 N 项
_SLIM_STR = 256             # 摘要保留的字符串前缀长度
_SLIM_KEYS = 64             # 摘要保留的 dict 键数上限
_SLIM_DEPTH = 6             # 摘要递归深度上限


def _json_bytes(obj):
    """序列化字节数（不可序列化 → -1，绝不抛）。"""
    try:
        return len(json.dumps(obj, ensure_ascii=False, default=str)
                   .encode('utf-8'))
    except Exception:                                      # noqa: BLE001
        return -1


def _sha1(obj):
    try:
        txt = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha1(txt.encode('utf-8')).hexdigest()[:16]
    except Exception:                                      # noqa: BLE001
        return ''


def _slim_value(v, depth=0):
    """超长值 → 摘要；保留形状（键名 / 长度 / 前若干项），便于人读与审计。"""
    if depth >= _SLIM_DEPTH:
        if isinstance(v, (list, tuple, dict)):
            return {'__slim__': 'deep', 'type': type(v).__name__,
                    'n': len(v)}
        return v
    if isinstance(v, str):
        if len(v) > _SLIM_STR:
            return {'__slim__': 'str', 'n_chars': len(v),
                    'head': v[:_SLIM_STR]}
        return v
    if isinstance(v, (list, tuple)):
        head = [_slim_value(x, depth + 1) for x in list(v)[:_SLIM_HEAD]]
        if len(v) > _SLIM_HEAD:
            return {'__slim__': 'seq', 'n': len(v), 'head': head}
        return head
    if isinstance(v, dict):
        out = {}
        for k in list(v)[:_SLIM_KEYS]:
            out[k] = _slim_value(v[k], depth + 1)
        if len(v) > _SLIM_KEYS:
            out['__slim_more_keys__'] = len(v) - _SLIM_KEYS
        return out
    return v


def slim_detail(detail, max_bytes=_MAX_DETAIL_BYTES):
    """detail → (落盘/留存用摘要, meta)；未超限时**原样返回**（meta=None）。

    meta = {'orig_bytes', 'slim_bytes', 'sha1', 'keys'}——只记摘要，不记内容：
    大型 blob（波形包络、结果数组）本就在状态存档/git 里，审计靠 sha1 比对。
    """
    orig = _json_bytes(detail)
    if orig < 0 or orig <= max_bytes:
        return detail, None
    slim = _slim_value(detail)
    if _json_bytes(slim) > max_bytes:
        # 摘要本身还超限 → 只留顶层标量键 + 键名清单（GUI/归因要的 path/job 在标量里）
        keep = {}
        if isinstance(slim, dict):
            for k, v in slim.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    keep[k] = v
        if isinstance(detail, dict):
            keep['__slim_keys__'] = list(detail)[:32]
        slim = keep or {'__slim__': 'opaque', 'type': type(detail).__name__}
    meta = {'orig_bytes': orig, 'slim_bytes': _json_bytes(slim),
            'sha1': _sha1(detail),
            'keys': (list(detail)[:32] if isinstance(detail, dict) else [])}
    return slim, meta


class Op:
    __slots__ = ('seq', 'ts', 'actor', 'kind', 'detail', 'detail_meta',
                 'reason', 'snapshot')

    def __init__(self, seq, actor, kind, detail, reason, snapshot):
        self.seq = seq
        self.ts = time.time()
        self.actor = actor          # 'user' | 'ai:<session>'
        self.kind = kind            # 'set_param' | 'add_group' | 'run_compute' ...
        self.detail, self.detail_meta = slim_detail(detail)   # 人类可读摘要 dict
        self.reason = reason or ''
        self.snapshot = snapshot    # 变更前 SimState.to_dict()（undo 用）

    def to_public(self):
        """对外（GUI/工具）视图：不含 snapshot（太大）。"""
        out = {'seq': self.seq, 'ts': self.ts, 'actor': self.actor,
               'kind': self.kind, 'detail': self.detail,
               'reason': self.reason}
        if self.detail_meta:
            out['detail_slim'] = self.detail_meta
        return out


class OpJournal:
    def __init__(self, root_dir=None):
        self._ops = []
        self._seq = 0
        self._root = root_dir
        self._listeners = []        # callable(op) —— GUI 订阅
        self.suppress_snapshot = False   # CodeAct 期间：op 照记但不带快照
                                          # （整段撤销由一条 run_python op 承担）

    # ---------- 记录 ----------
    def record(self, actor, kind, detail, reason='', snapshot=None):
        if self.suppress_snapshot:
            snapshot = None
        self._seq += 1
        op = Op(self._seq, actor, kind, detail, reason, snapshot)
        self._ops.append(op)
        if len(self._ops) > _MAX_HISTORY:
            self._ops.pop(0)
        self._append_disk(op)
        for cb in self._listeners:
            try:
                cb(op)
            except Exception:       # noqa: BLE001 - 监听器异常不阻断
                pass
        return op

    def subscribe(self, cb):
        self._listeners.append(cb)

    # ---------- 查询 ----------
    def since(self, since_seq=0):
        return [o.to_public() for o in self._ops if o.seq > since_seq]

    def last_snapshot(self, actor_prefix='ai'):
        """最近一个指定来源 op 的变更前快照（undo 用）；无则 None。"""
        for o in reversed(self._ops):
            if o.snapshot is not None and (
                    actor_prefix is None or o.actor.startswith(actor_prefix)):
                return o.seq, o.snapshot
        return None, None

    def undo_last(self, actor_prefix='ai'):
        """弹出最近一个带快照的 op，返回其变更前快照（供 SimState 回滚）。"""
        for o in reversed(self._ops):
            if o.snapshot is not None and (
                    actor_prefix is None or o.actor.startswith(actor_prefix)):
                self._ops.remove(o)
                return o.to_public(), o.snapshot
        return None, None

    # ---------- 磁盘 ----------
    def _append_disk(self, op):
        if not self._root:
            return
        try:
            d = os.path.join(self._root, '_journal')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, time.strftime('%Y-%m-%d') + '.jsonl'),
                      'a', encoding='utf-8') as f:
                f.write(json.dumps(op.to_public(), ensure_ascii=False) + '\n')
        except OSError:
            pass                    # 审计落盘失败不影响主流程


_JOURNAL = None


def get_journal(root_dir=None):
    global _JOURNAL
    if _JOURNAL is None:
        _JOURNAL = OpJournal(root_dir)
    return _JOURNAL
