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
"""
import json
import os
import time

_MAX_HISTORY = 200          # 内存中保留的可撤销 op 数


class Op:
    __slots__ = ('seq', 'ts', 'actor', 'kind', 'detail', 'reason', 'snapshot')

    def __init__(self, seq, actor, kind, detail, reason, snapshot):
        self.seq = seq
        self.ts = time.time()
        self.actor = actor          # 'user' | 'ai:<session>'
        self.kind = kind            # 'set_param' | 'add_group' | 'run_compute' ...
        self.detail = detail        # 人类可读摘要 dict
        self.reason = reason or ''
        self.snapshot = snapshot    # 变更前 SimState.to_dict()（undo 用）

    def to_public(self):
        """对外（GUI/工具）视图：不含 snapshot（太大）。"""
        return {'seq': self.seq, 'ts': self.ts, 'actor': self.actor,
                'kind': self.kind, 'detail': self.detail,
                'reason': self.reason}


class OpJournal:
    def __init__(self, root_dir=None):
        self._ops = []
        self._seq = 0
        self._root = root_dir
        self._listeners = []        # callable(op) —— GUI 订阅

    # ---------- 记录 ----------
    def record(self, actor, kind, detail, reason='', snapshot=None):
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
# AI 写入窗口：显式布尔覆盖写入全程（含 run_compute 的秒级预处理泵），
# 尾巴时间戳 0.6s 覆盖写完成后才发射的 field 防抖信号（180ms）等级联。
_ai_active = False
_ai_busy_until = 0.0


def set_ai_writing(on):
    global _ai_active, _ai_busy_until
    _ai_active = bool(on)
    if not on:
        _ai_busy_until = time.time() + 0.6


def is_ai_writing():
    return _ai_active or time.time() < _ai_busy_until


def get_journal(root_dir=None):
    global _JOURNAL
    if _JOURNAL is None:
        _JOURNAL = OpJournal(root_dir)
    return _JOURNAL
