# -*- coding: utf-8 -*-
"""引擎参数单源：**注册表 + 默认值 + 每次运行的覆写**。

============================================================
怎么用（三条规则）
============================================================
1) **读**：一律 `P.get('NAME')`（`from . import params as _P`），
   **不要**在函数签名默认值里写 `P.NAME`——默认值在 import 时求值，会绕过覆写。
   签名里要可覆写就写 `arg=None`，函数体内 `arg = P.get('NAME') if arg is None else arg`。
2) **覆写**：`with P.override({'PHASE_TOL': 1e-5}): engine(...)`。
   覆写存在**线程本地栈**里 → 每次运行一个快照、天然线程安全、可嵌套、退出即还原。
   GUI 的 `EvolveWorker` 就是这么做的（上帝模式"引擎参数"页 → 每次运行一个快照）。
3) 模块级 `P.NAME` 仍然可用（PEP 562 `__getattr__` 回落到 `get()`），
   故旧代码/脚本不必改；但**它只在"无覆写生效"时等于默认值**，
   要真正吃到覆写必须走规则 1。

============================================================
分层（与 §A~§E 一致）
============================================================
  【A】精度控制面 —— 5 个无量纲量 + 积分容差。**这是唯一的精度旋钮集。**
  【B】由【A】派生的运行时有量纲量（不是独立自由度）
  【C】结构 / 物理事实常量
  【D】模型边界（改它 = 换模型）
  【E】性能 / 数值保护（只影响速度与鲁棒性）

判据推导与历史动因见 `docs/参数与判据决策史.md`；
逐参数登记表见 `docs/引擎参数总表.md`。

本模块零依赖（只用标准库）：不 import 任何 src 子模块。
"""
import contextlib
import threading

# ============================================================
# 元数据：ParamSpec / SPECS
# ============================================================


class ParamSpec:
    """一个引擎参数的完整描述（默认值 + 单位 + 分层 + UI 元数据）。"""

    __slots__ = ('name', 'default', 'tier', 'kind', 'label', 'unit', 'desc',
                 'lo', 'hi', 'log', 'settable')

    def __init__(self, name, default, tier, label, desc='', unit='',
                 kind='float', lo=None, hi=None, log=False, settable=True):
        self.name = name
        self.default = default
        self.tier = tier
        self.kind = kind               # 'float' | 'int' | 'tuple'
        self.label = label
        self.desc = desc
        self.unit = unit
        self.lo = lo                   # UI 提示范围（不做硬校验）
        self.hi = hi
        self.log = log                 # 跨量级 → UI 用对数步长
        self.settable = settable       # False = 派生量/别名，不出现在引擎参数页


TIERS = (
    ('A', '精度控制面（无量纲）——引擎"算多准"的唯一旋钮集'),
    ('C', '结构 / 物理事实常量（不是精度旋钮）'),
    ('D', '模型边界（改它 = 换模型）'),
    ('E', '性能 / 数值保护（只影响速度与鲁棒性）'),
)

SPECS = (
    # ---------------- 【A】精度控制面 ----------------
    ParamSpec(
        'PHASE_TOL', 1e-4, 'A', '弱耦合可忽略门',
        'Θ = C·T·|V|²/max(|ΔE|,|V|) ≥ 本值 → 保留。V_dd 与微波对共用同一判据，'
        '各引擎保留集自动一致。共振极限 Θ=C·T·|V|，远失谐 Θ=C·T·|V|²/Δ。'
        '调大 → 丢更多弱耦合（更快，但可能丢多光子通道）。',
        unit='rad', lo=1e-9, hi=1e-1, log=True),
    # ⚠ PHASE_BUDGET（成本预算）已**迁出引擎参数面**：它是运行级成本旋钮，
    # 注册在 `solve.COST_BUDGET`（见 pm_setup._SOLVE_SPECS），默认 None
    # = 不限制。引擎侧仍读模块级默认 None（无人指定 → 不做成本剔除），
    # 真实存在的耦合一律保留；"可忽略"由上面的 Θ 判据回答。
    ParamSpec(
        'R_CRIT', 0.0135, 'A', 'AC 化边界 r=Ω/ω',
        'r > 本值的远强对从光子块剔除，位移走 far_sos 注入。',
        lo=1e-5, hi=1.0, log=True),
    ParamSpec(
        'RWA_TOL', 1e-4, 'A', 'RWA 有效性 Ω/ωL',
        'Ω/ωL < 本值 → RWA 有效（ip4 遗留引擎）。与 R_CRIT 是同一族的两个'
        '不同要求（块截断 vs 近似有效性），刻意分开。',
        lo=1e-8, hi=1e-1, log=True),
    ParamSpec(
        'M_EST_TOL', 1e-6, 'A', '光子块截断容差',
        '(r/4)^{2(M−1)} ≤ 本值 → M 收敛。mixed 路径用本值；sambe/两场用各自'
        '入口的 tol（默认同值）。', lo=1e-12, hi=1e-1, log=True),
    ParamSpec(
        'DEFAULT_RTOL', 1e-8, 'A', '积分相对容差',
        'mixed 引擎入口的默认 rtol。atol 由它按 ATOL_RTOL_RATIO 派生。',
        lo=1e-14, hi=1e-2, log=True),
    ParamSpec(
        'ATOL_RTOL_RATIO', 1e-2, 'A', 'atol/rtol 比',
        'atol = rtol × 本值（派生用，非独立旋钮）。',
        lo=1e-6, hi=1.0, log=True),
    ParamSpec(
        'DEFAULT_ATOL', None, 'A', '积分绝对容差（派生）',
        '= DEFAULT_RTOL × ATOL_RTOL_RATIO。只读：要改请改那两个。',
        settable=False),

    # ---------------- 【C】结构 / 物理事实 ----------------
    ParamSpec(
        'MW_MAX_MHZ', 1e5, 'C', '微波/光场分界',
        '载频 ≤ 本值按微波处理，> 按光场；兼作微波候选窗的绝对上界'
        '（兜 T→0）。', unit='MHz', lo=1e3, hi=1e9, log=True),
    ParamSpec(
        '_M_START', 2, 'C', '光子块 M 起点',
        'M 自适应起点（实际自 M=_M_START−1 起步）。', kind='int', lo=1, hi=8),
    ParamSpec(
        '_M_MAX', 8, 'C', '光子块 M 上限',
        '截断上限；达上限会记"截断误差可能 > tol"警告。',
        kind='int', lo=1, hi=16),
    ParamSpec(
        'DIRECT_SOS_N_WINDOW', 3, 'C', 'Floquet 偶极伙伴窗',
        '静态 AC-Stark 的 Floquet 基组自动纳入偶极伙伴 |Δn| ≤ 本值。',
        kind='int', lo=0, hi=12),
    ParamSpec(
        'RYDBERG_N_MIN', 20, 'C', '里德堡下限',
        'n ≥ 本值视为里德堡态（超精细可忽略）——物理事实。',
        kind='int', lo=1, hi=60),
    ParamSpec(
        'AC_EFFECT_THRESHOLD_MHZ', 0.05, 'C', 'UI「受影响」阈值',
        '|AC| > 本值 → UI 标记"受寻址光影响"（**显示阈值**，非引擎精度）。',
        unit='MHz', lo=1e-4, hi=1e3, log=True),

    # ---------------- 【D】模型边界 ----------------
    ParamSpec(
        'BW_OPT', 2e4, 'D', '光场对过滤带宽（mixed）',
        'mixed φ-帧光场对 |ΔD| < 本值保留。是**双峰分离器**（共振对 ~0 vs '
        '反旋伙伴 ~2ω≈7.6e8，差 4 个量级）→ 在 [1e3,1e6] 内取谁都一样；'
        '调小会让更多对进精确计算。', unit='MHz', lo=1e3, hi=1e6, log=True),
    ParamSpec(
        'OPT_NRES_BW_MHZ', 2e3, 'D', '光场对近共振窗（两场）',
        '两场引擎把光场耦合冻结为常量矩阵 ½Ω → 本窗是**正确性边界**'
        '（远失谐对塞进去是系统性错误），不可换相位判据。',
        unit='MHz', lo=1e1, hi=1e6, log=True),
    ParamSpec(
        'AC_NEAR_CUTOFF_MHZ', 5e3, 'D', '近/远失谐分流（field_build）',
        '|Δ| ≥ max(本值, Ω) → 只走静态 SOS；否则相干驱动进 A 矩阵。'
        '5 GHz 是覆盖超精细/精细流形的保守地板。',
        unit='MHz', lo=1e1, hi=1e6, log=True),
    ParamSpec(
        'M_EST_BW_MHZ', 2e4, 'D', 'M 估计的 r 取样窗',
        '只在此窗内取 max|A| 定 r（窗越宽 r 只增不减 → 最保守）。'
        '不改变精度目标（那是 M_EST_TOL），只决定"哪些对算近共振"。',
        unit='MHz', lo=1e2, hi=1e6, log=True),
    ParamSpec(
        'FRAME_TOL_REL', 1e-6, 'D', '旋转系有效性（相对）',
        '|ΔE_rot−ΔE_lab| ≤ 本值·max(|ΔE_rot|,|ΔE_lab|) → 该 V_dd 元素在'
        '本征旋转系里静态、可用。布尔分离器（差值只有 ~0 与 ~±ω_opt 两档）。',
        lo=1e-9, hi=1e-1, log=True),

    # ---------------- 【E】性能 / 数值保护 ----------------
    ParamSpec(
        '_DENSE_MAX', 2200, 'E', '稠密 eigh 上限',
        '扩展维 ≤ 本值走稠密 eigh，否则稀疏 expm_multiply。',
        kind='int', lo=100, hi=20000),
    ParamSpec(
        '_TF_DENSE_MAX', 3000, 'E', '两场两体稠密上限',
        '标准两场两体 22 态为 2420 维，须留稠密，否则 backend 翻成 '
        'tf-sambe-sparse（有自检断言该契约）。',
        kind='int', lo=100, hi=20000),
    ParamSpec(
        '_TAB_N', 2 ** 17, 'E', 'hard 包络表点数（契约）',
        '线性插值表点数。**接口契约**：任意波形可驱动带宽 = '
        '0.5·(_TAB_N−1)/T，报给用户（waveforms.f_max_mhz）→ 缩小它会改变'
        '承诺的带宽。', kind='int', lo=1024, hi=2 ** 21),
    ParamSpec(
        '_RHS_ENV_TAB_N', 20001, 'E', 'mixed RHS 表点数',
        'mixed RHS 包络/相位查表点数（内存/构建预算）。',
        kind='int', lo=101, hi=2 ** 19),
    ParamSpec(
        '_AC_ENV_TAB_N', 4001, 'E', 'mixed AC 表点数',
        'AC 时变对角的独立包络表点数（AC 只跟随慢变光强）。',
        kind='int', lo=101, hi=2 ** 18),
    ParamSpec(
        '_ENV_PROBE_N', 2001, 'E', 'sambe 包络探针点数',
        'sambe 包络活性/恒常判断采样点数。', kind='int', lo=17, hi=100001),
    ParamSpec(
        '_ENV_SAMPLES', 64, 'E', 'field_build 包络探针数',
        'field_build 包络恒常判断采样数（粗粒度）。', kind='int', lo=8, hi=4097),
    ParamSpec(
        '_DT_MIN_FRAC', 2e5, 'E', '自适应细分下限分数',
        '自适应细分绝对下限 = T/本值（防无限细分）。', lo=1e2, hi=1e9, log=True),
    ParamSpec(
        '_SPARSE_SEG_MAX', 50000, 'E', '稀疏分段上限',
        '稀疏自适应细分总段数上限（防爆炸卡死）。',
        kind='int', lo=100, hi=10 ** 7),
    ParamSpec(
        '_SPARSE_MAX_DEPTH', 4, 'E', '稀疏二分深度上限',
        '稀疏自适应二分递归深度上限（正常 ≤3）。', kind='int', lo=1, hi=12),
    ParamSpec(
        '_NODE_LADDER', (16, 32, 64, 128), 'E', '求积节点阶梯',
        'Volterra/GL 求积节点阶梯（逗号分隔整数）。', kind='tuple',
        lo=None, hi=None),
    ParamSpec(
        '_VOLT_MAX_ITER', 60, 'E', 'GMRES 迭代上限',
        'GMRES 求解上限迭代数。', kind='int', lo=1, hi=2000),
    ParamSpec(
        '_VOLT_PICARD_FRAC', 0.05, 'E', 'GMRES 容差比',
        'GMRES atol = tol × 本值（Picard 残差容差比）。',
        lo=1e-4, hi=1.0, log=True),
    ParamSpec(
        '_VOLT_PRE_SWEEPS', 8, 'E', '预条件扫掠次数',
        '预条件因果（cumsum）Picard 扫掠次数。', kind='int', lo=1, hi=64),
)

SPEC_BY_NAME = {s.name: s for s in SPECS}

# 别名：旧名 → 规范名（不在 SPECS 里，故不出现在 UI；get() 会转发）
ALIASES = {'POP_PHASE_TOL': 'PHASE_TOL'}

# 派生量：先算覆写后的依赖，再套这个函数（不占独立自由度）
_DERIVED = {'DEFAULT_ATOL': lambda ef: ef['DEFAULT_RTOL'] * ef['ATOL_RTOL_RATIO']}


def _defaults():
    d = {s.name: s.default for s in SPECS}
    for k, fn in _DERIVED.items():
        d[k] = None                      # 占位，避免 __getattr__ 无限递归
    d['DEFAULT_ATOL'] = d['DEFAULT_RTOL'] * d['ATOL_RTOL_RATIO']
    # 成本预算已迁到运行级（solve.COST_BUDGET）；引擎侧仍需要这个**默认值**
    # None（= 不限制）供 coupling.mw_cap_mhz 在无人指定时使用。
    d['PHASE_BUDGET'] = None
    return d


DEFAULTS = _defaults()

# ============================================================
# 取值 / 覆写
# ============================================================
_OVR = threading.local()


def _stack():
    st = getattr(_OVR, 'stack', None)
    if st is None:
        st = _OVR.stack = []
    return st


def _resolve(name):
    """规范名（别名转发）。"""
    return ALIASES.get(name, name)


def get(name):
    """当前生效值：线程本地覆写栈（就近优先）→ 派生 → 默认。"""
    n = _resolve(name)
    for frame in reversed(_stack()):
        if n in frame:
            return frame[n]
    fn = _DERIVED.get(n)
    if fn is not None:
        return fn(effective())
    if n not in DEFAULTS:
        raise KeyError(f'未知引擎参数 {name!r}（见 params.SPECS）')
    return DEFAULTS[n]


def effective():
    """全部参数的当前生效值（dict，含派生量与别名不重复）。"""
    out = {}
    for frame in _stack():
        out.update(frame)
    for name, spec in SPEC_BY_NAME.items():
        out.setdefault(name, spec.default)
    for name, fn in _DERIVED.items():
        out[name] = fn(out)
    return out


def current_overrides():
    """当前覆写栈的合并结果（不含默认值）——UI 用它回填输入框。"""
    out = {}
    for frame in _stack():
        out.update(frame)
    return out


@contextlib.contextmanager
def override(values=None, **kw):
    """在本次（线程）内临时覆写引擎参数。

        with params.override({'PHASE_TOL': 1e-5}):   # 或 override(PHASE_TOL=1e-5)
            evolve_...()

    未知名字 → KeyError（早失败，别让 UI 打错字悄悄无效）。
    可嵌套；退出時无条件还原。
    """
    vals = {}
    for k, v in dict(values or {}).items():
        n = _resolve(k)
        if n not in SPEC_BY_NAME:
            raise KeyError(f'未知引擎参数 {k!r}（见 params.SPECS）')
        vals[n] = v
    for k, v in kw.items():
        n = _resolve(k)
        if n not in SPEC_BY_NAME:
            raise KeyError(f'未知引擎参数 {k!r}（见 params.SPECS）')
        vals[n] = v
    st = _stack()
    st.append(vals)
    try:
        yield vals
    finally:
        st.pop()


def coerce(name, raw):
    """UI 用：把输入框文本转成类型化值 → (value, err)。空串/None → (None, None)。

    只做类型与形状校验；**不做物理范围校验**（范围提示见 ParamSpec.lo/hi）。
    """
    n = _resolve(name)
    if n not in SPEC_BY_NAME:
        return None, f'未知参数 {name}'
    if raw is None:
        return None, None
    if isinstance(raw, (int, float, tuple)):
        return raw, None
    s = str(raw).strip()
    if not s:
        return None, None
    kind = SPEC_BY_NAME[n].kind
    try:
        if kind == 'int':
            return int(float(s)), None      # 容忍 "1e4" 这种写法
        if kind == 'float':
            return float(s), None
        if kind == 'tuple':
            parts = [p for p in s.replace('；', ',').replace(' ', ',').split(',')
                     if p]
            return tuple(int(float(p)) for p in parts), None
    except ValueError:
        return None, (f'{n} 解析失败：{s!r}（应为 {kind}；'
                      '留空即用默认值）')
    return None, f'{n} 类型 {kind} 不支持'


def specs(tier=None, settable_only=True):
    """按层取参数规格（UI 用）；settable_only=False 时含派生/别名。"""
    return [s for s in SPECS
            if (tier is None or s.tier == tier)
            and (s.settable or not settable_only)]


def __getattr__(name):
    """PEP 562：`params.PHASE_TOL` 回落到 get()（含覆写）；旧代码零改动。"""
    if name in DEFAULTS or name in ALIASES:
        return get(name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
