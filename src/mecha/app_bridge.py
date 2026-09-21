# -*- coding: utf-8 -*-
"""GUI ↔ mecha 对接缝（app 级门面）——《回归主仓库-迁移计划》§2/§4.3 的落点。

把 `Bus + Authority + Session + Ops + JobRunner + registry` 捆成 GUI 需要的**一个对象**，
取代旧仓的 `pm_setup.new_registry` + `archive` + `src.ai.server` + `MainWindow.engine_overrides`
/`god_mode`。它是 Qt-agnostic 的纯 Python（GUI 侧 import 它，它不 import Qt）。

设计要点（对齐纲领）：
- **进程拓扑（T4）**：人启动 GUI = 启动项目 = 权威。仿真档 GUI 进程内托管 mecha；
  `mcp_server()` 给出 (srv, tools)，由 app_entry 在后台线程跑起来（进程内 MCP）。
- **单一 choke point（I1/I5）**：GUI 写经 `write()` → `session.set(actor='gui')`，与 AI
  写走同一条口、同一条事件总线；人机同路径。
- **两模式写权门控（§2）**：`authority` 挂在 session 上；AI 模式下 GUI 写被 gate 拒
  （GUI 侧据此把控件设只读），人工模式下 AI 写被拒。
- **模式切换是人的动作**：`switch_mode()` 由 GUI 调；**AI 不能给自己授权**（MCP 面不暴露
  switch）。AI→人 交接前要求 `in_flight()==0`（先静默 AI）。
- **实时刷新（M2/黑匣子分工）**：GUI `subscribe(cb)` 订阅 bus，AI 写入即时重渲染；
  文件级黑匣子仍是 cockpit 的职责（独立进程读 journal）。
- **新存档格式（M5）**：`save/load` 就是 `session.effective()` 的 JSON——扁平 registry 键
  → 值的快照。旧 save_state 由一次性转换脚本转成这个格式（不写运行时旧档适配器）。
"""
import json
import os

from .authority import Authority
from .bus import Bus
from .jobs import JobRunner
from .ops import Ops
from .session import Session
from . import registry
from . import runtime

__all__ = ['AppBridge']


class AppBridge:
    """GUI 的 mecha 门面。默认人工模式（人先开车；交给 AI 由人 switch_mode('ai')）。"""

    def __init__(self, root=None, engine=None, mode='human'):
        if engine is not None:
            runtime.set_engine(engine)     # 注入领域引擎（adapter.EngineAdapter）
        self.bus = Bus(root=root)
        self.authority = Authority(bus=self.bus, mode=mode)
        self.session = Session(bus=self.bus, authority=self.authority)
        self.ops = Ops(self.session)
        self.jobs = JobRunner(self.ops)
        self._fields = None                # 建场层惰性（engine.make_fields）

    @property
    def fields(self):
        """建场层（领域可选能力，经 engine.make_fields）；无则 None。"""
        if self._fields is None:
            try:
                self._fields = runtime.engine().make_fields(self.session)
            except NotImplementedError:
                self._fields = False
        return self._fields or None

    # ---------------- 只读投影 ----------------
    @property
    def cache(self):
        return self.session.cache()

    @property
    def mode(self):
        return self.authority.mode

    def read(self, key):
        return self.session.get(key)

    def effective(self):
        return self.session.effective()

    def params(self, **kw):
        """参数表（供检视器）；人可 include_internal=True 深挖（M2/D-B）。"""
        return registry.list_params(**kw)

    def describe(self, key):
        return registry.describe(key)

    def activity(self, since_seq=0):
        return self.session.activity(since_seq)

    def orientation(self, since_seq=0):
        return self.session.orientation(since_seq=since_seq)

    def subscribe(self, cb):
        """订阅 bus（GUI 实时刷新）；返回取消函数。"""
        return self.bus.subscribe(cb)

    # ---------------- 写（经 choke point，受写权门控） ----------------
    def write(self, key, value, reason, actor='gui'):
        return self.session.set(key, value, actor=actor, reason=reason)

    def write_many(self, patch, reason, actor='gui'):
        return self.session.set_many(patch, actor=actor, reason=reason)

    def undo(self, reason='undo', actor='gui'):
        return self.session.undo(actor=actor, reason=reason)

    def can_write(self, actor='gui'):
        """GUI 据此把控件设只读（AI 模式下 can_write('gui')==False）。"""
        return self.authority.can_write(actor)

    # ---------------- 运行 / 预览 ----------------
    def run(self, reason, actor='gui', diag=False, progress_arr=None):
        return self.ops.run(actor=actor, reason=reason, diag=diag,
                            progress_arr=progress_arr)

    def estimate(self):
        return self.ops.estimate()

    def preview(self):
        return self.ops.preview()

    # ---------------- 模式切换（人的动作） ----------------
    def in_flight(self):
        """当前在飞的 job 数（AI→人 交接前必须为 0）。"""
        return sum(1 for j in self.jobs.list_jobs() if j.get('status') == 'running')

    def switch_mode(self, new_mode, reason, actor='human'):
        return self.authority.switch(new_mode, actor=actor, reason=reason,
                                     in_flight=self.in_flight())

    # ---------------- 存档（新格式 = session 快照） ----------------
    def save(self, path, reason='save'):
        snap = {'format': 'mecha-session-v1', 'mode': self.mode,
                'params': self.session.effective()}
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(snap, f, ensure_ascii=False, indent=1, default=str)
        self.bus.append(actor='gui', kind='save', target=path, before=None,
                        after={'n_keys': len(snap['params'])}, reason=reason,
                        provenance='app_bridge.save', reversibility='none')
        return {'ok': True, 'path': path, 'n_keys': len(snap['params'])}

    def load(self, path, reason='load', actor='gui'):
        with open(path, encoding='utf-8') as f:
            snap = json.load(f)
        if not isinstance(snap, dict) or 'params' not in snap:
            raise ValueError(
                '不是 mecha-session-v1 存档（缺 params）：%r——旧格式请先用一次性'
                '转换脚本转成新格式（M5，不写运行时旧档适配器）' % path)
        params = snap['params'] or {}
        recs = self.session.set_many(params, actor=actor,
                                     reason='%s: %s' % (reason, path))
        return {'ok': True, 'path': path, 'n_keys': len(params),
                'seqs': [r['seq'] for r in recs]}

    # ---------------- 进程内 MCP（仿真档：GUI 进程 = 项目进程） ----------------
    def mcp_server(self, actor='ai'):
        """→ (srv, tools)：由 app_entry 在后台线程跑起来（进程内托管 MCP，T4）。"""
        from .server import create_server
        return create_server(self.session, self.ops, actor=actor)
