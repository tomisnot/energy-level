# -*- coding: utf-8 -*-
"""append-only 事件总线（纲领 I2「独立表征数=1」的唯一真值日志 / M4「不可抵赖」）。

一条 append-only 事件流，其余（HUD / 告警 / 飞行记录仪 / 驾驶舱）皆派生投影。
每条事件 = `{seq, ts, actor, kind, target, before, after, reason, provenance,
reversibility}`。三条硬纪律：

- **I3 大载荷走指针**：数组/波形/结果永不内联，落 `.mecha/payloads/<sha1>.npz`，
  事件里只给 `{path, sha1, shape, dtype}`——既省日志体积，又让"内容被动过"可被
  sha1 检出。
- **I4 体积闸 + 截断必告知**：任何字段超过 `max_inline_bytes` 一律收缩，且**明确
  记下截了多少、原文在哪**（不能事后才知道）。
- **M4 append-only**：只追加、永不改写既有行；seq 单调递增；进程重启后从盘上
  续 seq（AI 不能"打扫现场"）。

**T5 黑匣子**：journal 落盘 ⇒ 驾驶舱作为**独立进程**直接读文件即可在项目死后
还原死前全貌，无需项目进程存活。
"""
import glob
import hashlib
import json
import os
import threading
import time

__all__ = ['Bus', 'default_root', 'sha1_bytes', 'payload_pointer', 'gate_value']

# 单条事件里任一字段内联进 journal 的字节上限；超过则收缩（指针或截断）。
_MAX_INLINE_BYTES = 4096
# journal 单文件字节上限；超过则轮转归档（**不删**，驾驶舱仍读得到全史）。
_MAX_JOURNAL_BYTES = 8 * 1024 * 1024


def default_root():
    """默认运行时数据根目录 `<repo>/.mecha`（gitignore 忽略）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))     # src/mecha → src → repo
    return os.path.join(repo, '.mecha')


def sha1_bytes(b):
    return hashlib.sha1(b).hexdigest()


def _is_array(x):
    """鸭子判 numpy 数组（不硬 import numpy，避免总线层依赖）。"""
    return hasattr(x, 'shape') and hasattr(x, 'dtype') and hasattr(x, 'tobytes')


def payload_pointer(root, arr, subdir='payloads'):
    """大数组 → 落 npz，返回 `{path, sha1, shape, dtype}` 指针（I3）。

    sha1 取自数组原始字节 ⇒ 内容被动过可检出。同名（同 sha1）二次落盘零成本命中
    （harvest：内容寻址缓存）。
    """
    import numpy as np
    d = os.path.join(root, subdir)
    os.makedirs(d, exist_ok=True)
    a = np.ascontiguousarray(arr)
    raw = a.tobytes()
    h = sha1_bytes(raw)
    path = os.path.join(d, h + '.npz')
    if not os.path.exists(path):
        np.savez_compressed(path, arr=a)
    return {'$pointer': True, 'path': os.path.relpath(path, root),
            'sha1': h, 'shape': list(a.shape), 'dtype': str(a.dtype)}


def _shrink(root, val, max_bytes=_MAX_INLINE_BYTES):
    """体积闸（I4）：大值收缩为指针或截断，并**明确记下截了多少、原文在哪**。

    - numpy 数组 → payload 指针（无损，可 sha1 校验）。
    - 其它（str/list/dict）超 max_bytes → 落一份原文到 payloads，字段里只留头部 +
      `{$truncated, kept, total, where}`。
    - 小值原样返回。
    journal 事件字段用小预算（_MAX_INLINE_BYTES）；工具回程用较大预算（分页后
    正常回程不该被截）——两者同一逻辑，只 max_bytes 不同。
    """
    if _is_array(val):
        return payload_pointer(root, val)
    try:
        enc = json.dumps(val, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        enc = repr(val)
    if len(enc.encode('utf-8')) <= max_bytes:
        return val
    # 太大：落原文，字段里留头部 + 截断告知
    d = os.path.join(root, 'payloads')
    os.makedirs(d, exist_ok=True)
    h = sha1_bytes(enc.encode('utf-8'))
    where = os.path.join(d, h + '.json')
    if not os.path.exists(where):
        with open(where, 'w', encoding='utf-8') as f:
            f.write(enc)
    head = enc[:512]
    return {'$truncated': True, 'head': head, 'kept': len(head),
            'total': len(enc), 'where': os.path.relpath(where, root),
            'sha1': h}


def gate_value(root, val, max_bytes=_MAX_INLINE_BYTES):
    """体积闸的公共口（I4）：所有回程/事件字段走**同一把闸**。

    server.py 的每个工具回程、bus.append 的 before/after 都调它 ⇒ 截断口径
    一致（安全口径/单位/时间戳形态一致，截了多少+原文在哪都明确告知）。
    工具回程可传较大 max_bytes（分页后的正常回程不该被截）。
    """
    return _shrink(root, val, max_bytes)


class Bus:
    """append-only 事件总线：JSONL 落盘 + 进程内 seq 索引。

    用法：
        bus = Bus()                       # 用 <repo>/.mecha
        ev = bus.append(actor='ai', kind='set', target='field.B_gauss',
                        before=0.0, after=21.7, reason='Zeeman 调谐')
        bus.since(0)                      # diff-since-seq（读通道头号原语）
    """

    def __init__(self, root=None, journal_glob='journal*.jsonl'):
        self.root = root or default_root()
        os.makedirs(self.root, exist_ok=True)
        self.journal_path = os.path.join(self.root, 'journal.jsonl')
        self._journal_glob = journal_glob
        self._events = []          # 进程内全量索引（含从盘重载的历史）
        self._seq = 0
        self._subscribers = []     # 进程内实时订阅回调（SSE 端点用，Phase 3）
        self._lock = threading.Lock()   # async job 跨线程写总线：seq+落盘+索引须原子
        self._load()

    # ---------------- 持久化 ----------------
    def _journal_files(self):
        """当前 + 归档的 journal 文件（按名字排序；轮转归档不丢史）。"""
        return sorted(glob.glob(os.path.join(self.root, self._journal_glob)))

    def _load(self):
        """从盘重载全部事件（重启后续 seq；驾驶舱读全史靠这个）。"""
        self._events = []
        self._seq = 0
        for fp in self._journal_files():
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue        # 半行（崩溃时）→ 跳过，不让总线哑掉
                    self._events.append(ev)
                    self._seq = max(self._seq, int(ev.get('seq', 0)))
        self._events.sort(key=lambda e: e.get('seq', 0))

    def _write_line(self, ev):
        with open(self.journal_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(ev, ensure_ascii=False, default=repr) + '\n')

    def _maybe_rotate(self):
        try:
            sz = os.path.getsize(self.journal_path)
        except OSError:
            return
        if sz < _MAX_JOURNAL_BYTES:
            return
        # 归档当前文件（**不删**），后续写入落到新的 journal.jsonl
        stamp = time.strftime('%Y%m%d-%H%M%S')
        arch = os.path.join(self.root, 'journal.%s.jsonl' % stamp)
        os.replace(self.journal_path, arch)

    # ---------------- 写入口 ----------------
    def append(self, *, actor, kind, target=None, before=None, after=None,
               reason=None, provenance=None, reversibility=None):
        """追加一条事件（唯一写入口）。返回带 seq/ts 的完整事件 dict。

        before/after 过体积闸（大数组自动转指针，大文本截断并告知）。
        **线程安全**：async job 在工作线程里也走这里，seq 递增+落盘+索引在同一把锁内
        原子完成（否则并发 append 会撞 seq / 交错写行）；订阅回调在锁外触发（防死锁）。
        """
        with self._lock:
            self._seq += 1
            ev = {
                'seq': self._seq,
                'ts': time.time(),
                'actor': actor,
                'kind': kind,
                'target': target,
                'before': _shrink(self.root, before),
                'after': _shrink(self.root, after),
                'reason': reason,
                'provenance': provenance,
                'reversibility': reversibility,
            }
            self._write_line(ev)
            self._events.append(ev)
            self._maybe_rotate()
        for cb in list(self._subscribers):
            try:
                cb(ev)
            except Exception:                        # noqa: BLE001
                pass                                 # 订阅者异常不能拖垮写入
        return ev

    # ---------------- 读出口（只读、幂等） ----------------
    def since(self, seq):
        """diff-since-seq：只回 seq 之后的事件（不重灌全量，R1 头号原语）。"""
        seq = int(seq or 0)
        return [e for e in self._events if e.get('seq', 0) > seq]

    def tail(self, n=20):
        return self._events[-max(int(n), 0):]

    @property
    def last_seq(self):
        return self._seq

    def __len__(self):
        return len(self._events)

    def subscribe(self, cb):
        """注册实时回调（SSE 端点用）；返回取消函数。"""
        self._subscribers.append(cb)
        return lambda: self._subscribers.remove(cb) if cb in self._subscribers else None

    # ---------------- 定位包（R1 attach/orientation） ----------------
    def orientation(self, since_seq=0, tail_n=10):
        """中途加入会话的**有界定位包**：最近 N 条 + 当前 seq（不灌全量）。"""
        return {'last_seq': self._seq, 'n_events': len(self._events),
                'recent': self.tail(tail_n),
                'new_since': len(self.since(since_seq))}
