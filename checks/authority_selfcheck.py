# -*- coding: utf-8 -*-
"""写权模式机判据（mecha v2：Authority + Gate + Channel，两模式写权互斥 + 致死性）。

守总纲 §5.2⑤ 与硬纪律 4（actor 由通道钉死）：
  A. actor→side 映射（V2Bridge 适配器层：gui/human→human；ai/mcp/codeact/job→ai；
     未知 actor→拒 unknown_actor）。
  B. **出厂 LOCKED**（DEFAULT_MODE）：任何通道都不能写（nothing ships enabled）。
  C. human 模式：人侧可写、AI 侧被拒**且快照不变**、拒绝可教学
     （kind=authority_mode_mismatch + hint + suggest）。
  D. ai 模式：AI 侧可写、人侧被拒。
  E. switch_mode：**只有 human 侧能切**（AI 不能自解锁，kind=authority_switch_denied）；
     切到 LOCKED 全员禁写；未知 actor 拒。
     （v2 语义：mode 是操作者协调态，不落 History——mecha/authority.py
     docstring 已声明；v1 的 authority_switch 事件断言随之退役。）
  F. v1 的 undo 门不存在于 v2（History append-only，撤销=反向 set 记新事件，
     决策见 mecha 仓 ADR `EL能力面取舍` 第 3 条）。
  G. **致死性**：B（LOCKED 拒一切）+ C/D（越权被拒）合起来证明这道门是**载重的**。

纯内存装配（Gate/History/Authority 手搭 + V2Bridge 测试注入构造），不碰 EL 引擎、
不起 cache——判据测的是**控制权机制本身**（<1s）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from mecha.authority import Authority, DEFAULT_MODE, Mode  # noqa: E402
from mecha.errors import GateDenied                        # noqa: E402
from mecha.gate import Channel, Gate, accept_all           # noqa: E402
from mecha.history import History                          # noqa: E402

from examples.energy_level.bridge import V2Bridge, _ACTOR_CHANNEL  # noqa: E402

C = Checker('authority')


def _mk_bridge(mode: Mode) -> V2Bridge:
    """纯内存装配（engine=None）：只测控制权，不碰物理。"""
    auth = Authority(mode=mode)
    hist = History()
    gate = Gate(auth, hist, validate=accept_all)
    return V2Bridge(gate, hist, auth)


def _denied(fn):
    """执行 fn，返回 (是否被 GateDenied, 异常或 None)。"""
    try:
        fn()
        return False, None
    except GateDenied as e:
        return True, e


# ---------------- A. actor → side 映射（适配器层） ----------------
_a1, _e1 = _denied(lambda: _mk_bridge(Mode.HUMAN).write(
    'field.B_gauss', 1.0, reason='火星人越权', actor='martian'))
C.group('A actor→side 映射（桥层表：gui/human→human；ai 系→ai；未知→拒）', [
    ('ai/codeact/mcp/job → ai 侧',
     all(_ACTOR_CHANNEL[a] == 'ai' for a in ('ai', 'codeact', 'mcp', 'job'))),
    ('gui/human → human 侧',
     all(_ACTOR_CHANNEL[a] == 'human' for a in ('gui', 'human'))),
    ('细标签归因进史（Channel("gui","human").actor=gui, side=human）',
     Channel('gui', 'human').actor == 'gui' and Channel('gui', 'human').side == 'human'),
    ('未知 actor → 写被拒（kind=unknown_actor，安全默认）',
     _a1 and _e1.kind == 'unknown_actor'),
])

# ---------------- B. 出厂 LOCKED：谁都不可写 ----------------
brB = _mk_bridge(DEFAULT_MODE)
_b_gui, _eb = _denied(lambda: brB.write('field.B_gauss', 1.0, reason='人写', actor='gui'))
_b_ai, _ = _denied(lambda: brB.write('field.B_gauss', 2.0, reason='AI 写', actor='ai'))
C.group('B 出厂 LOCKED（DEFAULT_MODE）：任何通道都不能改域状态', [
    ('默认模式就是 locked（nothing ships enabled）',
     DEFAULT_MODE is Mode.LOCKED and brB.mode == 'locked'),
    ('human 侧写被拒（kind=authority_locked + hint）',
     _b_gui and _eb.kind == 'authority_locked' and bool(_eb.hint)),
    ('ai 侧写被拒', _b_ai),
    ('LOCKED 下两侧 can_write 都是 False',
     brB.can_write('gui') is False and brB.can_write('ai') is False),
    ('被拒后快照一字节没动', brB.effective() == {}),
])

# ---------------- C~E：human 起步的桥 ----------------
br = _mk_bridge(Mode.HUMAN)

# C. human 模式：人可写、AI 被拒且不落地
rec_human = br.write('field.B_gauss', 10.0, reason='人调磁场', actor='gui')
snap_after_human = br.effective()
_c_ai, _ec = _denied(lambda: br.write('field.B_gauss', 99.0,
                                       reason='AI 越权偷写', actor='ai'))
C.group('C human 模式：人侧可写、AI 侧被拒且不落地（写权互斥）', [
    ('gui 写入成功（回执 {ok,resolved,seq}）',
     rec_human['ok'] and rec_human['resolved'] == 10.0 and rec_human['seq'] >= 1),
    ('ai 写入被拒（GateDenied）', _c_ai),
    ('被拒后快照逐位不变（AI 没写进去）', br.effective() == snap_after_human),
    ('拒绝可教学：kind=authority_mode_mismatch + hint + suggest',
     _ec.kind == 'authority_mode_mismatch' and bool(_ec.hint) and bool(_ec.suggest)),
])

# D. ai 模式：AI 可写、人被拒
br.switch_mode(Mode.AI, actor='human')          # 人开闸
rec_ai = br.write('field.B_gauss', 20.0, reason='AI 调磁场', actor='ai')
_d_gui, _ = _denied(lambda: br.write('field.B_gauss', 88.0,
                                      reason='人越权偷写', actor='gui'))
C.group('D ai 模式：AI 侧可写、人侧被拒', [
    ('切到 ai 成功（mode=ai）', br.mode == 'ai'),
    ('ai 写入成功', rec_ai['ok'] and rec_ai['resolved'] == 20.0),
    ('gui 写入被拒', _d_gui),
    ('ai 模式下 can_write：ai=True / gui=False',
     br.can_write('ai') is True and br.can_write('gui') is False),
])

# E. switch_mode：AI 不能自解锁；LOCKED 全员禁写；未知 actor 拒
_e_self, _es = _denied(lambda: br.switch_mode(Mode.HUMAN, actor='ai'))
br.switch_mode(Mode.HUMAN, actor='human')       # 人夺回
br.switch_mode(Mode.LOCKED, actor='gui')        # gui 标签在 human 侧 → 可切
_e_locked_gui, _ = _denied(lambda: br.write('field.B_gauss', 1.0,
                                            reason='locked 下人写', actor='gui'))
_e_locked_ai, _ = _denied(lambda: br.write('field.B_gauss', 1.0,
                                           reason='locked 下 AI 写', actor='ai'))
br.switch_mode(Mode.HUMAN, actor='human')       # 解回
_e_unknown, _eu = _denied(lambda: br.switch_mode(Mode.AI, actor='martian'))
C.group('E switch_mode：只有人类侧能切（AI 不能自解锁）+ LOCKED 语义', [
    ('AI 侧切模式被拒（kind=authority_switch_denied）',
     _e_self and _es.kind == 'authority_switch_denied'),
    ('人类夺回：切回 human 放行', br.mode == 'human'),
    ('切到 LOCKED 后两侧都写不进（维护/危险态）',
     _e_locked_gui and _e_locked_ai),
    ('解回 human 后 gui 恢复可写', br.can_write('gui') is True),
    ('未知 actor 切模式被拒（unknown_actor）',
     _e_unknown and _eu.kind == 'unknown_actor'),
    ('mode 不落 History（协调态非域状态；authority.py 声明）',
     all(e.key == 'field.B_gauss' for e in br.history.events())),
])

# F. v1 undo 门在 v2 不存在（如实登记，不是断言失败）
C.check('F v2 无 undo 口（撤销=反向 set 记新事件，ADR EL能力面取舍；'
        '写权门的覆盖面因此不需要 undo 分支）',
        not hasattr(br, 'undo'))

# G. 致死性：B 的 LOCKED 拒一切 + C/D 的越权被拒 ⇒ 门是载重的
C.check('G 致死性：无门可写（human 模式人写成功）对照越权必拒——门不是哑门',
        rec_human['ok'] and rec_ai['ok'] and _c_ai and _d_gui
        and _b_gui and _b_ai and _e_self)

sys.exit(0 if C.summary() else 1)
