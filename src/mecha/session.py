# -*- coding: utf-8 -*-
"""会话态 + **写 choke point**（纲领 I1「语义层是不二写入口，no side doors」）。

会话态 = 当前配置快照（喂 `runspec.build_run` 的登记册 dict）+ 撤销栈 + 惰性
ArcCache。**状态管理住在引擎之外**：core 保持纯函数，本模块持有"现在值是多少、
谁改的"（见 `docs/引擎层该有状态管理吗.md` 结论）。

三条硬纪律：
- **I1 唯一写入口**：快照 `_snapshot` 是私有的，**唯一变更口是 `_write`**（`set`
  / `undo` 都经它）。任何绕过 `set` 的裸写入 = 一个监控盲区，由 `mecha_kernel_
  selfcheck` 的 AST 守卫致死性看住。
- **D5 报错不静默**：`set` 先 `registry.validate`（类型/单位/值域/枚举/必填），
  非法 → 抛 `InvalidValue` 且**快照不变**（自愈回退，绝不静默兜底）。
- **M2 声称 vs 实测**：`set` 回执带 `readback`——由 choke point（而非 AI 自己）
  写完后**读回**的值，监控器把"AI 声称写入的" 与 "实测读回的" 并排显示。

**人机同一条代码路径（I5）**：`effective()` 产出的快照，与"人点运行"喂给
`runspec.build_run` 的是同一份；`ops.run` 复用它，不另起炉灶。
"""
from . import registry
from .bus import Bus
from .errors import MechaError

__all__ = ['Session', 'MISSING']


class _Missing:
    """哨兵：区分"键不存在"与"值就是 None"（与 runspec._required 同一纪律）。"""
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return '<MISSING>'

    def __bool__(self):
        return False


MISSING = _Missing()


class Session:
    """一次会话的配置快照 + 写 choke point。

    参数：
      bus  : 事件总线（None → 用默认 `<repo>/.mecha` 新建一个）
      root : 运行时数据根（bus 为 None 时用；测试可指临时目录）
    """

    def __init__(self, bus=None, root=None, snapshot=None, authority=None):
        self._snapshot = {}                 # ← 私有；唯一变更口是 _write
        self._undo = []                      # [(key, before_value_or_MISSING)]
        self._bus = bus if bus is not None else Bus(root=root)
        self._cache = None                   # 惰性引擎缓存句柄（经 runtime.engine）
        self._authority = None               # 控制权门（None=不门控；装载快照时不拦）
        if snapshot:
            for k, v in dict(snapshot).items():
                self.set(k, v, actor='session', reason='init from snapshot')
        self._authority = authority          # 快照装载后再挂门（init 写入不受门控）

    # ---------------- 只读投影 ----------------
    @property
    def bus(self):
        return self._bus

    @property
    def authority(self):
        """控制权门（None=不门控）。挂上后 set/set_many/undo 先过 gate(actor)。"""
        return self._authority

    def get(self, key):
        """读快照里的键；不存在 → None（I6：契约外字段读出来永远是 None）。"""
        return self._snapshot.get(key)

    def effective(self):
        """→ 完整配置快照的**副本**（喂 runspec.build_run 的那份；I5 同路径）。"""
        return dict(self._snapshot)

    def cache(self):
        """惰性引擎缓存句柄（经注入的 engine；mecha 不认识 arc/ArcCache）。"""
        if self._cache is None:
            from . import runtime
            self._cache = runtime.engine().make_cache()
        return self._cache

    def orientation(self, since_seq=0):
        """中途加入会话的**有界定位包**（R1）：配置摘要 + 最近事件 + 当前 seq。"""
        snap = self._snapshot
        return {
            'n_keys': len(snap),
            'keys': sorted(snap),
            'method': snap.get('dynamic.method'),
            'mode': snap.get('dynamic.mode'),
            'bus': self._bus.orientation(since_seq=since_seq),
        }

    # ---------------- 唯一变更口 ----------------
    def _write(self, key, value):
        """**唯一**改 `_snapshot` 的地方（I1 choke point 的物理落点）。

        value is MISSING → 删键；否则写键。set/undo 都经它，别无二路。
        AST 守卫（mecha_kernel_selfcheck）证明：全 mecha 里改 `_snapshot` 的
        语句只出现在 `__init__` 与本方法内。
        """
        if value is MISSING:
            self._snapshot.pop(key, None)
        else:
            self._snapshot[key] = value

    # ---------------- 写 choke point ----------------
    def set(self, key, value, *, actor, reason):
        """写一个键（经校验 → 落快照 → 发事件）。返回带 `resolved`/`readback` 的回执。

        actor/reason 是**必填**（M1 归因：每条事件带谁、为什么）。value=None 视为
        "清除该键"（写 MISSING）。非法值 → 抛错且快照不变（D5 自愈回退不静默）。
        """
        if not actor:
            raise MechaError('set 需要 actor（M1 归因：这个变化是谁做的）')
        if not reason:
            raise MechaError('set 需要 reason（M1 归因：为什么改）——'
                             '无理由的写入不进总线')
        if self._authority is not None:
            self._authority.gate(actor)          # 控制权门控（写权互斥；§2）
        norm = registry.validate(key, value)     # 非法在此抛，快照尚未动
        return self._commit(key, norm, actor, reason)

    def set_many(self, patch, *, actor, reason):
        """批量写（harvest：句柄 + 增量 patch）。**先全校验，再全落地**（原子）：
        任一键非法 → 一个都不写（不留半拉子状态）。每键各发一条事件。
        """
        if not isinstance(patch, dict):
            raise MechaError('set_many 需要 dict（{key: value}）')
        if self._authority is not None:
            self._authority.gate(actor)          # 控制权门控
        norms = {k: registry.validate(k, v) for k, v in patch.items()}  # 全校验
        return [self._commit(k, n, actor, reason) for k, n in norms.items()]

    def _commit(self, key, norm, actor, reason):
        """校验后落地 + 记账 + 发事件（set/set_many 共用）。"""
        before = self._snapshot.get(key, MISSING)
        after = MISSING if norm is None else norm
        self._write(key, after)
        self._undo.append((key, before))
        ev = self._bus.append(
            actor=actor, kind='set', target=key,
            before=(None if before is MISSING else before),
            after=(None if after is MISSING else after),
            reason=reason, provenance='session.set',
            reversibility='undo')
        readback = self._snapshot.get(key)       # ← choke point 产出的实测读回（M2）
        return {'ok': True, 'key': key,
                'resolved': (None if after is MISSING else after),
                'readback': readback,           # 与 resolved 并排显示，AI 伪造不了
                'before': (None if before is MISSING else before),
                'seq': ev['seq']}

    # ---------------- 撤销（L2 可逆） ----------------
    def can_undo(self):
        return bool(self._undo)

    def undo(self, *, actor, reason='undo'):
        """撤销最近一次 set：恢复到 before，发一条 kind='undo' 的反向事件。"""
        if self._authority is not None:
            self._authority.gate(actor)          # 控制权门控（undo 也是写）
        if not self._undo:
            raise MechaError('没有可撤销的写入（撤销栈空）',
                             hint='undo 只回退经 session.set 的写入')
        key, before = self._undo.pop()
        current = self._snapshot.get(key, MISSING)
        self._write(key, before)                 # 恢复（仍经唯一变更口）
        ev = self._bus.append(
            actor=actor, kind='undo', target=key,
            before=(None if current is MISSING else current),
            after=(None if before is MISSING else before),
            reason=reason, provenance='session.undo',
            reversibility='redo')
        return {'ok': True, 'key': key,
                'restored': (None if before is MISSING else before),
                'readback': self._snapshot.get(key), 'seq': ev['seq']}

    # ---------------- 活动查询（读通道，只读幂等） ----------------
    def activity(self, since_seq=0):
        """diff-since-seq：seq 之后的事件（驾驶舱/AI tail 总线维持心智模型）。"""
        return self._bus.since(since_seq)
