# -*- coding: utf-8 -*-
"""控制权 token（两模式互斥的写权门控；《回归主仓库-迁移计划》§2 / 原则 §5）。

**单一控制权（single control authority）**：任一时刻只有一方（AI 或 人）能写语义状态。
**互斥的是"写权"，不是"状态"**——状态/事件总线/choke point 始终只有一个；本模块只是一
道门，决定"此刻谁的写入被放行"。

- `mode='ai'`    ：AI 监工模式——AI 侧 actor（ai/codeact/job/mcp）可写；人侧被拒（GUI 只读）。
- `mode='human'` ：自己干模式——人侧 actor（human/gui/user/session）可写；AI 侧被拒
                  （MCP 写通道门控，AI 写 → 可教学报错"人工模式，AI 无写权"）。

切换是**原子 + 记账 + 静默**：`switch()` 发一条 `authority_switch` 事件；AI→人 前要求
AI 静默（`in_flight==0`，即在飞的 async job 已收尾/取消）。人→AI 后由调用方让 AI 用
定位包 / diff-since-seq 重新站定（R1）。

**可选、默认不门控**：`Session(authority=None)` 时完全不 gate（向后兼容，既有判据不受
影响）；只有显式挂上 `Authority` 才启用门控。泛化：这个写权 token 就是硬件档（原则 §5/§6）
的 L3 控制权交接。
"""
from .errors import MechaError

__all__ = ['MODES', 'Authority', 'AuthorityError', 'side_of']

MODES = ('ai', 'human')

_AI_ACTORS = {'ai', 'codeact', 'job', 'mcp'}
_HUMAN_ACTORS = {'human', 'gui', 'user', 'session'}


class AuthorityError(MechaError):
    """写入被控制权门控拒绝（当前模式不持写权）。"""
    kind = 'authority_denied'


def side_of(actor):
    """actor → 'ai' | 'human' | None（未知）。`job:<id>` 归 AI 侧。"""
    a = str(actor or '')
    if a.startswith('job:') or a in _AI_ACTORS:
        return 'ai'
    if a in _HUMAN_ACTORS:
        return 'human'
    return None


class Authority:
    """写权 token。挂在 Session 上后，set/set_many/undo 都会先过 gate(actor)。"""

    def __init__(self, bus=None, mode='human'):
        if mode not in MODES:
            raise AuthorityError('未知模式 %r（合法：%s）' % (mode, list(MODES)),
                                 hint="mode 只能是 'ai' 或 'human'")
        self._mode = mode
        self._bus = bus

    @property
    def mode(self):
        return self._mode

    def can_write(self, actor):
        """actor 侧 == 当前模式 才可写；未知 actor 一律拒（安全默认）。"""
        return side_of(actor) == self._mode

    def gate(self, actor):
        """放行或抛可教学 AuthorityError（D2：错误消息即 LLM 的 UI）。"""
        if self.can_write(actor):
            return
        side = side_of(actor)
        raise AuthorityError(
            '当前是 %s 模式（写权在 %s 侧）；actor=%r 属 %s 侧，无写权'
            % (self._mode, self._mode, actor, side or '未知'),
            hint=('要写就先交接控制权：switch(%r)；或让持权方来操作'
                  % ('ai' if side == 'human' else 'human')),
            suggest=list(MODES))

    def switch(self, new_mode, *, actor, reason, in_flight=0):
        """切换控制权（原子 + 记账 + 静默）。

        in_flight：当前在飞的 AI async job 数；**AI→人** 前必须为 0（先把 AI 静默，
        否则人在 AI 还在写的时候夺权 = 并发写冲突）。
        """
        if new_mode not in MODES:
            raise AuthorityError('未知模式 %r（合法：%s）' % (new_mode, list(MODES)),
                                 hint="mode 只能是 'ai' 或 'human'")
        if not reason:
            raise AuthorityError('switch 需要 reason（M1 归因：为什么交接控制权）')
        old = self._mode
        if old == 'ai' and new_mode == 'human' and int(in_flight or 0) > 0:
            raise AuthorityError(
                'AI→人 交接前必须先静默 AI：还有 %d 个在飞 job' % int(in_flight),
                hint='先 cancel/等待 job 收尾（jobs.cancel / poll）再 switch')
        if old == new_mode:
            return {'ok': True, 'mode': new_mode, 'changed': False, 'from': old,
                    'seq': None}
        self._mode = new_mode
        seq = None
        if self._bus is not None:
            ev = self._bus.append(
                actor=actor, kind='authority_switch', target='mode',
                before=old, after=new_mode, reason=reason,
                provenance='authority.switch', reversibility='switch')
            seq = ev['seq']
        return {'ok': True, 'mode': new_mode, 'changed': True, 'from': old,
                'seq': seq}
