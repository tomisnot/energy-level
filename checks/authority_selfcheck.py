# -*- coding: utf-8 -*-
"""控制权 token 自检（两模式写权互斥 + 交接协议 + 致死性）。

守《回归主仓库-迁移计划》§2 的架构支点：
  A. actor→side 映射（job:<id> 归 AI；未知→None→拒）。
  B. **向后兼容**：Session(authority=None) 完全不门控（既有判据不受影响）。
  C. human 模式：人侧可写、AI 侧被拒**且快照不变**、拒绝可教学（kind/hint/suggest）。
  D. ai 模式：AI 侧可写、人侧被拒。
  E. switch：每次真切换发一条 authority_switch 事件；**AI→人 且 in_flight>0 被拒**
     （交接前必须静默 AI）；同模式重复切换幂等（changed=False）。
  F. undo 也过控制权门。
  G. **致死性**：B（无门控时 AI 能写）+ C/D/F（有门控时越权被拒）合起来证明
     这道门是**载重的**，不是永不触发的哑门（D4）。

用**临时目录**当 bus root（不污染 `<repo>/.mecha`）。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from src.mecha import session as SESS                      # noqa: E402
from src.mecha.authority import Authority, AuthorityError, side_of  # noqa: E402
from src.mecha.bus import Bus                              # noqa: E402
from src.mecha import runtime                              # noqa: E402
from src.adapter import EngineAdapter                      # noqa: E402

runtime.set_engine(EngineAdapter())   # 注入领域引擎（mecha 零领域 import）

C = Checker('authority')

# ---------------- A. actor → side 映射 ----------------
C.group('A actor→side 映射（job:<id> 归 AI；未知→None）', [
    ('ai/codeact/mcp/job → ai',
     all(side_of(a) == 'ai' for a in ('ai', 'codeact', 'mcp', 'job'))),
    ('job:<id> → ai', side_of('job:1712-ab12cd') == 'ai'),
    ('human/gui/user/session → human',
     all(side_of(a) == 'human' for a in ('human', 'gui', 'user', 'session'))),
    ('未知 actor → None（安全默认：谁都不是→不可写）',
     side_of('martian') is None and side_of('') is None),
])

# ---------------- B. 向后兼容：无 authority 不门控 ----------------
tmpB = tempfile.mkdtemp(prefix='authB_')
sb = SESS.Session(root=tmpB)                 # 无 authority
rb_ai = sb.set('field.B_gauss', 1.0, actor='ai', reason='无门控时 AI 写')
rb_gui = sb.set('field.E_vpcm', 0.5, actor='gui', reason='无门控时人写')
C.group('B 无 authority 时任何 actor 都能写（向后兼容既有判据）', [
    ('ai 写入成功（无门控）', rb_ai['ok'] and rb_ai['resolved'] == 1.0),
    ('gui 写入成功（无门控）', rb_gui['ok'] and rb_gui['resolved'] == 0.5),
])

# ---------------- C~G：挂 authority 的会话（共享一条 bus） ----------------
tmpA = tempfile.mkdtemp(prefix='authA_')
bus = Bus(root=tmpA)
auth = Authority(bus=bus, mode='human')
sh = SESS.Session(bus=bus, authority=auth)

# C. human 模式：人可写、AI 被拒且不落地
ok_human = sh.set('field.B_gauss', 10.0, actor='gui', reason='人调磁场')
snap_after_human = sh.effective()
err = None
try:
    sh.set('field.B_gauss', 99.0, actor='ai', reason='AI 越权偷写')
    ai_rejected = False
except AuthorityError as e:
    ai_rejected = True
    err = e.to_dict()
snap_unchanged = (sh.effective() == snap_after_human)
C.group('C human 模式：人侧可写、AI 侧被拒且不落地（写权互斥）', [
    ('gui 写入成功', ok_human['ok'] and ok_human['resolved'] == 10.0),
    ('ai 写入被拒（AuthorityError）', ai_rejected),
    ('被拒后快照逐位不变（AI 没写进去）', snap_unchanged),
    ('拒绝可教学：kind=authority_denied + hint + suggest',
     bool(err) and err.get('kind') == 'authority_denied'
     and bool(err.get('hint')) and bool(err.get('suggest'))),
])

# D. ai 模式：AI 可写、人被拒
sw1 = auth.switch('ai', actor='human', reason='交给 AI 开车')
ok_ai = sh.set('field.B_gauss', 20.0, actor='ai', reason='AI 调磁场')
try:
    sh.set('field.B_gauss', 88.0, actor='gui', reason='人越权偷写')
    human_rejected = False
except AuthorityError:
    human_rejected = True
C.group('D ai 模式：AI 侧可写、人侧被拒', [
    ('切到 ai 成功（changed=True）', sw1['ok'] and sw1['changed'] and auth.mode == 'ai'),
    ('ai 写入成功', ok_ai['ok'] and ok_ai['resolved'] == 20.0),
    ('gui 写入被拒', human_rejected),
])

# E. switch：记账 + AI 静默闸 + 幂等
try:
    auth.switch('human', actor='human', reason='夺权', in_flight=2)
    quiesce_blocked = False
except AuthorityError:
    quiesce_blocked = True          # AI→人 且在飞 job>0 → 必须先静默
sw2 = auth.switch('human', actor='human', reason='AI 已静默，夺权', in_flight=0)
sw_noop = auth.switch('human', actor='human', reason='重复切换')
switch_events = [e for e in bus.since(0) if e.get('kind') == 'authority_switch']
C.group('E switch：原子 + 记账 + AI 静默闸（交接协议）', [
    ('AI→人 且 in_flight>0 被拒（交接前必须静默 AI）', quiesce_blocked),
    ('in_flight=0 时切换放行、changed=True、mode=human',
     sw2['ok'] and sw2['changed'] and sw2['mode'] == 'human'),
    ('同模式重复切换 changed=False（幂等，不发多余事件）',
     sw_noop['changed'] is False),
    ('两次真切换各发一条 authority_switch 事件（共 %d）' % len(switch_events),
     len(switch_events) == 2),
    ('switch 事件带 before/after（human↔ai）与 reason',
     all(e.get('reason') and e.get('before') and e.get('after')
         for e in switch_events)),
])

# F. undo 也过控制权门（当前 human 模式；ai undo 应被拒）
try:
    sh.undo(actor='ai')
    undo_gated = False
except AuthorityError:
    undo_gated = True
C.check('F undo 也过控制权门（ai 在 human 模式 undo 被拒）', undo_gated)

# G. 致死性：门控是载重的（B 证明无门控时 AI 能写；C/D/F 证明有门控时越权被拒）
C.check('G 致死性：越权写在有门控时确实被拦（非哑门；对照 B 无门控可写）',
        ai_rejected and human_rejected and undo_gated
        and rb_ai['ok'] and rb_ai['resolved'] == 1.0)

sys.exit(0 if C.summary() else 1)
