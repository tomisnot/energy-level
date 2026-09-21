# -*- coding: utf-8 -*-
"""互锁（硬件档强制层）scaffold——L3 不可逆动作的最后一道闸。

铁律二：不可逆动作过"人 + 硬件互锁"双闸，且**互锁不能是软件/AI 策略**（限值在固件/硬件
层强制）。本模块是软件侧的"互锁接口 + 仿真档自动放行 + 硬件档默认拒绝"的 scaffold；
真硬件绑定（GPIO/PLC/固件握手）留待硬件档接入，此处只定契约与默认安全态。

- SimInterlock：仿真档（可逆、零风险）自动放行。
- HardwareInterlock：默认**未武装 ⇒ 一律拒**；放行需 armed + human_confirmed 双闸齐过。
"""
__all__ = ['Interlock', 'SimInterlock', 'HardwareInterlock']


class Interlock:
    """互锁契约：allow(action) → {'ok': bool, 'why': str}。"""

    def allow(self, action):
        raise NotImplementedError


class SimInterlock(Interlock):
    def allow(self, action):
        return {'ok': True, 'why': 'sim：可逆、零风险，自动放行'}


class HardwareInterlock(Interlock):
    def __init__(self, armed=False):
        self.armed = bool(armed)

    def allow(self, action):
        if not self.armed:
            return {'ok': False, 'why': '未武装（默认安全态，拒绝一切不可逆动作）'}
        if not (action or {}).get('human_confirmed'):
            return {'ok': False, 'why': '已武装但缺人确认（L3 双闸不齐）'}
        return {'ok': True, 'why': 'armed + 人确认，双闸齐过'}
