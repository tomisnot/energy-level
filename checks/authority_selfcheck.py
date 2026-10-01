# -*- coding: utf-8 -*-
"""EL **适配层**的写权映射判据（只留 EL 自己写的那一层）。

## 为什么只剩这些（2026-10-01 收窄）

写权门 / 模式机**本身**的语义是**框架的**，且框架判据已经在实测：
- `mecha/checks/phase1_assembly_selfcheck.py`：LOCKED 下 seed 落地与校验、OPEN 两侧可写、
  既有模式互斥、`authority_switch_denied`、拒绝不留痕、快照与 fold(史) 一致、批量回滚；
- `mecha/checks/phase5_bridge_selfcheck.py`：**对 EL 的真 `V2Bridge`** 实测 can_write、
  写权互斥（ai 模式下 gui 被拒 + 抛 GateDenied）、`switch_mode(actor='ai')` 被拒、
  `switch_mode(actor='gui')` 经适配器映射到 human 侧放行、订阅刷新与 journal 归因。

⇒ 本判据**不再重测那些**（同义重复），只守**框架不知道的那部分**——EL 的
`_ACTOR_CHANNEL` 表与它的边界行为：

- **A** 表的内容与完整性（谁映到哪一侧；只有两种 side）；
- **B** 未知 actor：`write` / `switch_mode` 都拒（`unknown_actor`）、`can_write` 为 False；
- **C** 映射的**效果**：细标签进史做归因、side 定写权（human / ai 互为对照）；
- **D** `watch_mode` 只是把 `Authority.watch` 透出去（真变化才通知 + disposer 生效）。

⚠ 保留棘轮：这几条删掉，EL 的适配层就没人守了（框架判据测框架语义，它不知道 EL 的表长什么样）。
⚠ 未覆盖（如实登记）：`watch_mode` 在**生产**里没有消费者（见交接文档 §8 挂起项）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from mecha.authority import Authority, Mode                # noqa: E402
from mecha.errors import GateDenied                        # noqa: E402
from mecha.gate import Gate, accept_all                    # noqa: E402
from mecha.history import History                          # noqa: E402

from mecha_v2.bridge import V2Bridge, _ACTOR_CHANNEL       # noqa: E402

C = Checker('authority')


def _mk_bridge(mode: Mode) -> V2Bridge:
    """纯内存装配（engine=None）：只测控制权，不碰物理。"""
    auth = Authority(mode=mode)
    hist = History()
    return V2Bridge(Gate(auth, hist, validate=accept_all), hist, auth)


def _denied(fn):
    """执行 fn → (是否被 GateDenied, 异常或 None)。"""
    try:
        fn()
        return False, None
    except GateDenied as e:
        return True, e


# ---------------- A. EL 的映射表（框架不知道它长什么样） ----------------
C.group('A EL 适配层的 actor→side 映射表', [
    ('ai 系四个标签 → ai',
     all(_ACTOR_CHANNEL[a] == 'ai' for a in ('ai', 'codeact', 'mcp', 'job'))),
    ('human 侧两个标签 → human',
     all(_ACTOR_CHANNEL[a] == 'human' for a in ('gui', 'human'))),
    ('表里只有这两种 side（没有第三种写权面）',
     set(_ACTOR_CHANNEL.values()) == {'human', 'ai'}),
])

# ---------------- B. 未知 actor：fail closed ----------------
br = _mk_bridge(Mode.HUMAN)
_a_den, _a_err = _denied(lambda: br.write('field.B_gauss', 1.0,
                                          reason='火星人越权', actor='martian'))
_b_den, _b_err = _denied(lambda: br.switch_mode(Mode.AI, actor='martian'))
C.group('B 未知 actor → 一切入口都拒（不静默、不默认放行）', [
    ('write 被拒且 kind=unknown_actor', _a_den and _a_err.kind == 'unknown_actor'),
    ('switch_mode 被拒且 kind=unknown_actor', _b_den and _b_err.kind == 'unknown_actor'),
    ('can_write 未知标签 = False（控件据此只读）', br.can_write('martian') is False),
    ('两处被拒后快照仍空（没有偷偷落地）', br.effective() == {}),
])

# ---------------- C. 映射的效果：归因 + 写权（一正一反） ----------------
rec = br.write('field.B_gauss', 10.0, reason='人调磁场', actor='gui')
ev = br.history.events()[-1]
_c_den, _c_err = _denied(lambda: br.write('field.B_gauss', 99.0,
                                          reason='AI 越权偷写', actor='ai'))
br.switch_mode(Mode.AI, actor='human')          # 人开闸（看门人语义）
rec_ai = br.write('field.B_gauss', 20.0, reason='AI 调磁场', actor='ai')
_d_den, _ = _denied(lambda: br.write('field.B_gauss', 88.0,
                                     reason='人越权偷写', actor='gui'))
_rec = br.history.events()[-1]
C.group('C 映射的效果：细标签进史 + side 定写权（human / ai 互为对照）', [
    ('gui 写在 human 模式放行，且**史里的 actor 是细标签 gui**（没被抹成 side）',
     rec['ok'] and ev.actor == 'gui' and ev.target == 'field.B_gauss'),
    ('ai 在 human 模式被拒，且可教学（kind + hint + suggest）',
     _c_den and _c_err.kind == 'authority_mode_mismatch'
     and bool(_c_err.hint) and bool(_c_err.suggest)),
    ('ai 模式：ai 可写、gui 被拒（互斥是真的）', rec_ai['ok'] and _d_den),
    ('ai 写的史 actor 是 ai（面板/端点据此高亮归因）', _rec.actor == 'ai'),
])

# ---------------- D. watch_mode = Authority.watch 的透出 ----------------
_seen = []
_unwatch = br.watch_mode(lambda m: _seen.append(m))
br.switch_mode(Mode.HUMAN, actor='human')       # 真变化 → 通知
_after_one = list(_seen)
_unwatch()
br.switch_mode(Mode.AI, actor='human')          # 已退订 → 不再收
C.group('D watch_mode 透出模式变更（"此刻谁能写"的订阅口）', [
    ('真变化时回调收到 mode 字符串', _after_one == ['human']),
    ('disposer 退订后不再收（切换本身仍生效）', len(_seen) == 1 and br.mode == 'ai'),
])

sys.exit(0 if C.summary() else 1)
