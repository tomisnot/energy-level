# -*- coding: utf-8 -*-
"""分级自治 + 后端路由（纲领 §5 / §6 硬件档两条铁律 / Phase 5 **只预留接口**）。

**统一判据**（§5）：*这一步错了能不能撤销、撤销多贵* → 决定交给 AI 的自主度。
可逆且廉价 → 放手；不可逆或昂贵 → "AI 提议、人 + 互锁放行"（单向门必须人把关）。

**后端路由**（harvest 的 DISPATCH 北极星）：一张带 `backend` 列的表，`'sim'`（现状）
/ `'hardware'`（未来）。**AI 侧零改动**：同一套 `set_param`/`run_compute` 语义面，
`backend` 只是路由细节——这正是"仿真引擎就是硬件的数字孪生 / what-if oracle"（§6
统一点）：AI 提议序列 → 先在 sim 里跑（可逆、零风险、给预测）→ 预测+风险+代价摆给
人 → 人确认 → 确定性运行时在硬件上执行**同一份配方**。

**两条铁律（硬件档；本模块只留结构，不实现）：**
- 铁律一：AI 永远不在硬实时回路里——只在上层声明配方/参数/序列，由确定性运行时执行。
- 铁律二：不可逆动作过"人 + 硬件互锁"双闸，**互锁不能是软件/AI 策略**（固件/硬件层）。

⚠ **Phase 5 不实现**：真实设备驱动、硬实时内核、物理互锁固件——不在 Re0 范围。
本模块提供的是"结构就位"：路由表 + 自治分级 + L3 双闸 + 数字孪生 dry-run 的骨架。
"""
from .errors import MechaError

__all__ = ['LEVELS', 'BACKENDS', 'backends', 'route', 'classify', 'l3_gate',
           'ReservedBackend', 'ARMED']

# 硬件档默认**未武装**：即使结构就位，没有人的显式武装 + 互锁，任何 L3 都不放行。
ARMED = False


class ReservedBackend(MechaError):
    """请求了预留但未实现的后端（如 'hardware'）——明确报错，不静默退化成 sim。"""
    kind = 'reserved_backend'


# 自治级别（§5）
LEVELS = {
    'L1': dict(label='只读/查询', ai='自由',
               ops=['list_params', 'get_state', 'get_activity', 'estimate_cost',
                    'preview', 'describe', 'orientation']),
    'L2': dict(label='可逆/有界致动', ai='允许 + 自动互锁（校验）+ 全程记录',
               ops=['set_param', 'set', 'set_many', 'undo']),
    'L3': dict(label='不可逆/高能量/安全攸关', ai='**只能提议**，人确认 + 硬件互锁',
               ops=['出光', '加压', '朝人/样品移动', '加热', '淬灭', '删档']),
}

# 后端路由表（`backend` 列 = harvest 北极星；加一列即接真机，AI 侧零改动）
BACKENDS = {
    'sim': dict(label='仿真（数字孪生）', implemented=True, reversible=True,
                ground_truth='math', l3_gate='auto_pass', realtime='none',
                note='可逆、可重跑；undo=快照；run 属 L2'),
    'hardware': dict(label='实验室硬件控制', implemented=False, reversible=False,
                     ground_truth='none', l3_gate='propose_confirm_interlock',
                     realtime='hard', armed=ARMED,
                     note='Phase 5 只预留接口；不可逆、不可复现、部分可观测'),
}

_L1 = set(LEVELS['L1']['ops'])
_L2 = set(LEVELS['L2']['ops'])
# 运行/致动类：在可逆后端上是 L2，在不可逆后端上升级 L3
_ACTUATE = {'run', 'run_compute', 'actuate', 'arm', 'fire', 'execute'}


def backends():
    """→ 路由表副本（供查询/展示；带 backend 列的语义）。"""
    return {k: dict(v) for k, v in BACKENDS.items()}


def route(backend):
    """backend → 路由信息；**未实现的后端明确报 Reserved**（不静默当 sim）。"""
    if backend not in BACKENDS:
        raise ReservedBackend(
            '未知 backend %r（合法：%s）' % (backend, sorted(BACKENDS)),
            hint="现状只有 'sim' 可用；'hardware' 是 Phase 5 预留")
    b = BACKENDS[backend]
    if not b.get('implemented'):
        raise ReservedBackend(
            "backend=%r 已预留但**未实现**（%s）——Phase 5 只留接口，不接真机"
            % (backend, b['label']),
            hint="数字孪生 dry-run 请用 backend='sim'；接真机需先实现驱动 + 互锁固件")
    return b


def classify(op, backend='sim'):
    """把一个操作归到 L1/L2/L3（分级自治）。

    - 只读 → L1；可逆写 → L2；
    - 运行/致动在**可逆后端**（sim）→ L2，在**不可逆后端**（hardware）→ **L3**；
    - 未知操作按**最严**（L3）——安全默认（宁可多要一次人确认）。
    """
    if op in _L1:
        return 'L1'
    b = BACKENDS.get(backend, {})
    reversible = b.get('reversible', True)
    if op in _L2:
        return 'L2'
    if op in _ACTUATE:
        return 'L2' if reversible else 'L3'
    return 'L3'


def l3_gate(backend, *, confirmed=False, interlock_ok=False, armed=None):
    """L3 不可逆动作的闸 → `(allow, reason)`。

    - sim（auto_pass）：可逆、可重跑，默认放行。
    - hardware（propose_confirm_interlock）：**必须** 人确认（confirmed）AND 硬件
      互锁 OK（interlock_ok，独立于软件/AI）AND 已武装（armed）——三者缺一即拒。
      这落实铁律二：AI 只能提议，放行的是"人 + 硬件互锁"双闸，不是软件策略。
    """
    b = BACKENDS.get(backend)
    if b is None:
        return False, '未知 backend %r' % backend
    if b.get('l3_gate') == 'auto_pass':
        return True, 'sim 档：可逆可重跑，L3 默认放行（auto_pass）'
    _armed = b.get('armed', False) if armed is None else bool(armed)
    if not _armed:
        return False, 'hardware 档：未武装（ARMED=False）——AI 只能提议，不能放行'
    if not confirmed:
        return False, 'hardware 档：缺人确认（单向门必须人把关）'
    if not interlock_ok:
        return False, 'hardware 档：硬件互锁未 OK（互锁独立于软件，是最后一道闸）'
    return True, 'hardware 档：人确认 + 硬件互锁 + 已武装 → 放行'
