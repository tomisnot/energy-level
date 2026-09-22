# -*- coding: utf-8 -*-
"""HubBridge ⇄ Hub 契约自检（A1）——两种 transport 的**同名方法形状必须一致**。

**背景（A1，功能不可用级）**：`HubBridge` 是 GUI 的另一种 transport（Hub 模式），
`AppBridge` 是 in-process 模式。两者是"**同一门面的两种 transport**"，
形状不一致就是静默坏。实测曾有三处：
  ① `read`/`effective` 读 `get_state()['params']`，而 `Ops.state()` 回的是
     `{'snapshot', 'orientation', 'mode'}` ⇒ **恒空** ⇒ GUI 在 Hub 模式**显示不出任何配置**；
  ② `mode` 读 `_state()['mode']`（当时不存在）⇒ 恒回落 `'human'` ⇒ **AI 模式下 GUI 还以为可写**；
     ⚠ 注意**不能**改读 `orientation['mode']`——那是**配置里的 `dynamic.mode`（物理：0/1）**，
     同名不同义（本判据专门钉这一条）；
  ③ `subscribe` 回调回 **changed-keys list**，而 `_mecha_bind._on_event(ev)` 立刻调
     `ev.get('kind')` ⇒ **list 没有 `.get` → AttributeError** ⇒ Hub 模式实时刷新直接抛异常。

  A. `effective()`/`read()` 读 **`snapshot`**（与 `Ops.state()` 一致），非空；
  B. `mode` 取**顶层 authority `mode`**，且**不误取** `orientation['mode']`（物理 mode）；
  C. `subscribe` 回调形状是**事件 dict**（含 kind/target/actor），与 `AppBridge` 一致；
  D. **同名方法契约一致**：`effective`/`read`/`mode`/`subscribe`/`save`/`load`/`undo`/`can_write`
     两侧都有且形状同类（本判据用假 client 免起进程）；
  E. **致死性**：把读键改回 `params`、或把回调改回 list ⇒ 必红。

用**假 client**（让 `get_state` 回**真实 `Ops.state()` 的形状**）⇒ 与真 Hub 一致且不依赖起进程。
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from mecha import runtime                              # noqa: E402
from mecha.app_bridge import AppBridge                 # noqa: E402
from mecha.authority import Authority                  # noqa: E402
from mecha.bus import Bus                              # noqa: E402
from mecha.hub_bridge import HubBridge                 # noqa: E402
from mecha.ops import Ops                              # noqa: E402
from mecha.session import Session                      # noqa: E402
from src.adapter import EngineAdapter                  # noqa: E402

C = Checker('hub_bridge_contract')

# ---- 造一个真 session，取**真实** Ops.state() 形状，再用假 client 喂给 HubBridge ----
runtime.set_engine(EngineAdapter())
_bus = Bus(root=tempfile.mkdtemp(prefix='hb_'))
_auth = Authority(bus=_bus, mode='human')
_sess = Session(bus=_bus, authority=_auth)
_sess.set('field.B_gauss', 21.7, actor='gui', reason='夹具')
_sess.set('dynamic.mode', 1, actor='gui', reason='夹具：物理 mode=1（两体）')
_ops = Ops(_sess)
_REAL_STATE = _ops.state()          # ← 权威形状


class _FakeClient:
    """假 Hub 客户端：`get_state` 回**真实 Ops.state()**；`get_activity` 回真事件。"""

    def __init__(self, state, session):
        self._state = state
        self._sess = session
        self.url = 'http://127.0.0.1:1/mcp'
        self.timeout = 5.0
        self.calls = []

    def call(self, name, **kw):
        self.calls.append((name, kw))
        if name == 'get_state':
            return dict(self._state)
        if name == 'get_activity':
            since = int(kw.get('since_seq') or 0)
            return {'since_seq': since, 'last_seq': self._sess.bus.last_seq,
                    'events': self._sess.activity(since)}
        if name == 'list_params':
            return {'params': []}
        return {}

    def close(self):
        pass


_fc = _FakeClient(_REAL_STATE, _sess)
_hb = HubBridge.__new__(HubBridge)          # 不走 __init__（那会真连 Hub）
_hb.client = _fc
import threading                                # noqa: E402
_hb._stop = threading.Event()

# ---------------- A. 读键：snapshot ----------------
C.group('A HubBridge 读键 = `snapshot`（与 `Ops.state()` 一致），非空', [
    ('`Ops.state()` 顶层键含 snapshot/orientation/mode（%s）'
     % sorted(_REAL_STATE), 'snapshot' in _REAL_STATE),
    ('**`effective()` 非空且与 Hub 侧一致**',
     _hb.effective().get('field.B_gauss') == 21.7),
    ('`read(key)` 读得回真值', _hb.read('field.B_gauss') == 21.7),
    ('`effective()` 不含 `params` 那个不存在的键（旧版据此恒空）',
     'params' not in _REAL_STATE),
])

# ---------------- B. mode：取顶层 authority mode，不误取物理 mode ----------------
_ori_mode = _REAL_STATE['orientation'].get('mode')
C.group('B `mode` 取**顶层 authority mode**；**不误取** `orientation.mode`（物理）', [
    ('顶层 `mode` = %r（authority）' % _REAL_STATE.get('mode'),
     _REAL_STATE.get('mode') == 'human'),
    ('`orientation.mode` = %r（**物理** dynamic.mode，同名不同义）' % _ori_mode,
     _ori_mode == 1),
    ('**HubBridge.mode == 顶层 authority mode**（不是物理 mode）',
     _hb.mode == 'human'),
    ('**致死性**：若误取 `orientation.mode` 会得到 1（非 ai/human）⇒ 本断言必红',
     _hb.mode in ('ai', 'human') and _hb.mode != _ori_mode),
    ('can_write(gui) 与 mode 一致（human ⇒ True）',
     _hb.can_write('gui') is True),
])

# ---------------- C. subscribe 回调形状 = 事件 dict ----------------
_got = []
_unsub = _hb.subscribe(lambda ev: _got.append(ev), interval=0.2)
time.sleep(0.3)
_sess.set('field.E_vpcm', 3.3, actor='gui', reason='触发订阅')   # human 模式 ⇒ 人侧 actor
time.sleep(1.2)
_unsub()
_last = _got[-1] if _got else {}
C.group('C `subscribe` 回调形状 = **事件 dict**（与 `AppBridge` 一致；含 kind/target/actor）', [
    ('回调**真被触发**（%d 次）' % len(_got), len(_got) >= 1),
    ('**形状是 dict**（旧版回 list ⇒ `_mecha_bind` 的 `ev.get` 会 AttributeError）',
     isinstance(_last, dict)),
    ('带 kind/target/actor（`Binder` 归因高亮正需要这三个）',
     _last.get('kind') == 'set' and _last.get('target') == 'field.E_vpcm'
     and _last.get('actor') == 'gui'),
    ('**致死性**：若回 changed-keys list，`isinstance(ev, dict)` 为假 ⇒ 本断言必红',
     isinstance(_last, dict) and not isinstance(_last, list)),
    ('用的是 `get_activity`（能给出完整事件，而非差分）',
     any(c[0] == 'get_activity' for c in _fc.calls)),
])

# ---------------- D. 两种 transport 的同名方法契约一致 ----------------
_br = AppBridge(root=tempfile.mkdtemp(prefix='ab_'), engine=EngineAdapter(),
                mode='human')
_SHARED = ['effective', 'read', 'mode', 'subscribe', 'save', 'load', 'undo',
           'can_write', 'activity', 'params', 'describe', 'run', 'estimate',
           'preview', 'in_flight', 'switch_mode']
_missing_hub = [m for m in _SHARED if not hasattr(_hb, m)]
_missing_app = [m for m in _SHARED if not hasattr(_br, m)]
C.group('D `HubBridge` ⇄ `AppBridge` 同名方法契约一致（同一门面的两种 transport）', [
    ('共有方法 %d 个都在两侧（Hub 缺：%s / App 缺：%s）'
     % (len(_SHARED), _missing_hub or '无', _missing_app or '无'),
     not _missing_hub and not _missing_app),
    ('两侧 `effective()` 都回 dict（形状同类）',
     isinstance(_br.effective(), dict) and isinstance(_hb.effective(), dict)),
    ('两侧 `mode` 都回 str',
     isinstance(_br.mode, str) and isinstance(_hb.mode, str)),
    ('两侧 `can_write` 都回 bool',
     isinstance(_br.can_write('gui'), bool)
     and isinstance(_hb.can_write('gui'), bool)),
])

# ---------------- E. 致死性：把读键改回 `params` ⇒ 必红 ----------------
# 直接模拟**旧版读法**（读 `params`）对着**真实 state 形状**跑一遍：
# 真值在 `snapshot` 里，`params` 键不存在 ⇒ 旧版必然拿到空。
_old_effective = (_REAL_STATE.get('params') or {})       # ← 旧版 `hub_bridge.effective()`
_new_effective = (_REAL_STATE.get('snapshot') or {})     # ← 现行实现
C.group('E **致死性**：读键改回 `params`（= 旧版）⇒ 必红', [
    ('真实 `state()` 里**没有** `params` 键（%s）' % sorted(_REAL_STATE),
     'params' not in _REAL_STATE),
    ('**旧版读法拿到空**（%r）⇒ 正是"GUI 显示不出配置"的根因'
     % (_old_effective or {}), not _old_effective),
    ('现行读法拿到真值（21.7）', _new_effective.get('field.B_gauss') == 21.7),
    ('对照：两键集合不同 ⇒ 读错键必然拿不到真值（本判据因此是载重的）',
     set(_REAL_STATE) != {'params'}),
])

sys.exit(0 if C.summary() else 1)
