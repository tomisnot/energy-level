# -*- coding: utf-8 -*-
"""互锁判据（Phase 7 出口）：sim 自动放行；hardware 未武装拒 / armed 缺人确认拒 / 双闸齐过放行。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checklib import Checker                              # noqa: E402
from src.mecha.interlock import HardwareInterlock, SimInterlock  # noqa: E402

C = Checker('interlock')


def main():
    C.check('sim 档自动放行（可逆零风险）', SimInterlock().allow({}).get('ok') is True)
    h = HardwareInterlock()
    C.check('hardware 未武装 ⇒ 一律拒（默认安全态）',
            h.allow({}).get('ok') is False)
    h.armed = True
    C.check('armed 但缺人确认 ⇒ 拒（L3 双闸不齐）',
            h.allow({}).get('ok') is False)
    C.check('armed + 人确认 ⇒ 放行（双闸齐过）',
            h.allow({'human_confirmed': True}).get('ok') is True)
    ok = C.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
