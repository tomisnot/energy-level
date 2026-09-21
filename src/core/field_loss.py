# -*- coding: utf-8 -*-
"""场致加热与损失——核心物理模型（L2 散射 / L3 加热逃逸 / L4 光电离）。

角色：把"寻址光远失谐导致的流失"从**静态读数**升级为**引擎可消费的一等物理**。
本模块只放**模型与速率公式**（绝热消去运动自由度，不建外态、不含演化器）；
把 `γ(t)` 打进传播内核属引擎侧（`coupling.rhs_phi_pairs` 的时变损耗块）。

三层（可独立开关）：
  L2 散射   γ_iw = (Γ_j/2)·s/(1+s)，s = 2Ω²/(4Δ²+Γ²)（含饱和上限 Γ/2）。
            **修正旧 F3**：源态 = 被光"掺入短寿命伙伴"的那个态、Γ 取**伙伴**
            的离开基组率——不再按能量分 lo/hi 取反（旧 `real_field` 在里德堡
            寻址几何 60S↔6P 下把源态/Γ 用反，scatter[60S] 恒 0）。
  L3 加热   散射事件计数 N_sc 累积 → 温度 ODE / 预算档 → Arrhenius 逃逸率 γ_esc。
  L4 光电离 R_PI = σ_PI·Φ（线性于光强、无饱和上限）。

双计规则：伙伴在基组内且已配自发辐射通道 ⇒ 该通道 out 记 0（相干 + 自发辐射
已承担该流失，不再叠一份散射计损）。近共振剔除：|Δ| < near_cutoff ⇒ 该对走
相干动力学，二阶散射公式无定义 ⇒ 整对跳过。

单位与全 core 一致：Γ/Ω/Δ 用 MHz，时间 ns；速率 = Γ×1e-3 /ns；SI 常数走
`scipy.constants`（与 `static_atom` 同一来源，不另造第二套）。
"""
import numpy as np
from scipy.constants import h as _H, hbar as _HBAR, c as _C, k as _KB, atomic_mass

_MHZ_TO_NS = 1e-3                              # Γ[MHz] → 速率[/ns]（core 通用）
_M_RB87 = 86.909180527 * atomic_mass           # kg（⁸⁷Rb 原子质量，物理事实）


# =====================================================================
# 原子反冲量（本仓参数：⁸⁷Rb，λ=1013.8 nm 寻址）
# =====================================================================
def recoil_energy_J(lam_m, mass_kg=_M_RB87):
    """单光子反冲能 E_r = ħ²k²/2m，k=2π/λ。返回 J。"""
    k = 2.0 * np.pi / lam_m
    return (_HBAR * k) ** 2 / (2.0 * mass_kg)


def recoil_temperature_K(lam_m, mass_kg=_M_RB87):
    """反冲温度 E_r/k_B（K）。λ=1013.8 nm → ~107 nK（锚 A7）。"""
    return recoil_energy_J(lam_m, mass_kg) / _KB


# =====================================================================
# L2 · 场致散射损耗（含饱和，修正旧 F3 取反）
# =====================================================================
def scatter_rate_mhz(Om_mhz, det_mhz, gam_partner_mhz):
    """远失谐散射把源态布居抽走的速率 γ_sc（MHz）。

    含饱和的正确形式：
        R_sc = (Γ/2)·s/(1+s)，  s = 2Ω²/(4Δ²+Γ²)
    - s≪1（弱场/远失谐）→ 回到线性式 Ω²Γ/(4Δ²)（锚 A8 左端）；
    - Ω→∞ → 上限 Γ/2（散射通道天花板）。
    - **源态 = 被掺入短寿命伙伴的那个态**，Γ 取**伙伴**的总衰变率
      （修正 F3：旧 field_build 把二者用反，里德堡几何下恒为 0）。
    """
    Om = abs(float(Om_mhz))
    D = float(det_mhz)
    G = float(gam_partner_mhz)
    if G <= 0.0:
        return 0.0
    s = 2.0 * Om * Om / (4.0 * D * D + G * G)
    return 0.5 * G * s / (1.0 + s)


def scatter_rate_linear_mhz(Om_mhz, det_mhz, gam_partner_mhz):
    """线性式 Ω²Γ/(4Δ²)（仅诊断对比用；强场会高估，见锚 A8）。"""
    Om = abs(float(Om_mhz))
    D = float(det_mhz)
    return Om * Om * float(gam_partner_mhz) / (4.0 * D * D)


# =====================================================================
# L4 · 光电离（线性于光强，无饱和上限）
# =====================================================================
def photoion_rate_mhz(intensity_mwcm2, lam_m, sigma_mb):
    """单光子电离率 R_PI = σ_PI·Φ（MHz）。

    Φ = I/(hν) 光子通量（/m²/s）；σ_PI 用 Mb（1 Mb = 1e-22 m²）。
    ARC 不提供光电离截面 ⇒ σ 必须由使用者给。sigma_mb=0 或 I≤0 → 关闭。
    """
    if sigma_mb <= 0.0 or intensity_mwcm2 <= 0.0:
        return 0.0
    I_Wm2 = float(intensity_mwcm2) * 10.0          # mW/cm² → W/m²
    nu = _C / lam_m                                # Hz
    phi = I_Wm2 / (_H * nu)                        # 光子通量 /m²/s
    sigma_m2 = float(sigma_mb) * 1e-22             # Mb → m²
    R_s = sigma_m2 * phi                           # /s
    return R_s * 1e-6                              # s⁻¹ → MHz


# =====================================================================
# 场致损耗配置：逐波 × 逐态 × env(t)²
# =====================================================================
class FieldLossConfig:
    """把"若干束光 → 每个基组态的时变损耗率"打包。

    逐波 × 逐态 × env(t)²（沿用 mixed 的 ac 形状）。每通道携带两个速率：
      out_mhz —— **离开基组**的率（计入 loss_scatter / loss_photoion）；
      tot_mhz —— **总散射事件率**（驱动 L3 加热，含基组内回流的那部分）。
    光电离：out=tot=σ·Φ（电离即丢失，不驱动反冲加热 → tot 计 0 于加热）。

    双计规则由调用方负责：伙伴态已在基组内且配了自发辐射通道 ⇒ 把该通道
    的 out_mhz 传 0（相干驱动 + 伙伴衰变已由演化自动产生流失）。
    """

    def __init__(self, N):
        self.N = int(N)
        # 每项：dict(src, env_fn, out_mhz, tot_mhz, kind)
        self.channels = []
        self.trap = None            # 可选 TrapModel（L3 加热逃逸，由装配器挂上）

    def add_scatter(self, src, env_fn, out_mhz, tot_mhz=None):
        """散射通道。tot_mhz 缺省 = out_mhz（全部散射都离开基组）。"""
        out_mhz = float(out_mhz)
        tot_mhz = out_mhz if tot_mhz is None else float(tot_mhz)
        if out_mhz > 0.0 or tot_mhz > 0.0:
            self.channels.append(dict(src=int(src), env_fn=env_fn,
                                      out_mhz=out_mhz, tot_mhz=tot_mhz,
                                      kind='scatter'))
        return self

    def add_photoion(self, src, env_fn, rate_mhz):
        """光电离通道（线性于光强、无上限）。"""
        rate_mhz = float(rate_mhz)
        if rate_mhz > 0.0:
            self.channels.append(dict(src=int(src), env_fn=env_fn,
                                      out_mhz=rate_mhz, tot_mhz=0.0,
                                      kind='photoion'))
        return self

    def rates_at(self, tv):
        """t(ns) → (g_sc_ns, g_tot_ns, g_pi_ns)：三个长度 N 的 /ns 向量。

        g_sc  = 散射离开基组的率（计损，进 −iΓ/2 对角，分项 scatter）；
        g_tot = 总散射事件率（驱动加热，含回流）；
        g_pi  = 光电离率（分项 photoion，同样进对角）。
        总对角损耗 = spont + g_sc + g_pi + g_esc（三者互斥、无双计）。
        """
        g_sc = np.zeros(self.N)
        g_tot = np.zeros(self.N)
        g_pi = np.zeros(self.N)
        for ch in self.channels:
            env = float(np.asarray(ch['env_fn'](np.atleast_1d(tv)),
                                   float).flat[0])
            w = env * env * _MHZ_TO_NS              # env² 且 MHz→/ns
            if ch['kind'] == 'photoion':
                g_pi[ch['src']] += ch['out_mhz'] * w
            else:
                g_sc[ch['src']] += ch['out_mhz'] * w
                g_tot[ch['src']] += ch['tot_mhz'] * w
        return g_sc, g_tot, g_pi

    def peak_loss_rates_mhz(self):
        """各态 env=1 时的峰值损耗率（MHz），供诊断/日志。"""
        out = np.zeros(self.N)
        for ch in self.channels:
            out[ch['src']] += ch['out_mhz']
        return out

    def is_empty(self):
        return not self.channels


def scatter_pair_rates(Om_mhz, det_mhz, gam_out_i, gam_out_j):
    """修正 F3 的逐对双向规则：光把伙伴"掺"进来 × 伙伴离开基组的率。

    返回 (γ_i, γ_j)（MHz）——两端**数值系数相同、各用对方的 Γ_out**：
        γ_i = R_sc(Ω, Δ, Γ_out_j)，γ_j = R_sc(Ω, Δ, Γ_out_i)
    与旧 field_build 的差别：不再按能量分 lo/hi、不再取错 Γ。
    里德堡寻址几何（短寿命 6P 在**下**）下 γ(60S)>0 —— 旧实现恒为 0。
    """
    return (scatter_rate_mhz(Om_mhz, det_mhz, gam_out_j),
            scatter_rate_mhz(Om_mhz, det_mhz, gam_out_i))


def add_scatter_pair(cfg, i, j, Om_mhz, det_mhz,
                     gam_out_i_mhz, gam_out_j_mhz, env_fn,
                     gam_tot_i_mhz=None, gam_tot_j_mhz=None,
                     near_cutoff_mhz=5000.0):
    """把一对 (i,j) 的双向散射损耗装进 cfg（out=离开基组计损，tot=驱动加热）。

    gam_tot 缺省 = gam_out（不区分回流）；伙伴在基组内已配自发辐射通道时，
    调用方把对应 gam_out 传 0（双计规则）。
    近共振剔除：|Δ| < near_cutoff → 该对走相干动力学，二阶散射公式无定义，
    整对跳过。
    """
    if abs(float(det_mhz)) < float(near_cutoff_mhz):
        return cfg
    gi, gj = scatter_pair_rates(Om_mhz, det_mhz,
                                gam_out_i_mhz, gam_out_j_mhz)
    ti, tj = scatter_pair_rates(Om_mhz, det_mhz,
                                gam_tot_i_mhz if gam_tot_i_mhz is not None
                                else gam_out_i_mhz,
                                gam_tot_j_mhz if gam_tot_j_mhz is not None
                                else gam_out_j_mhz)
    cfg.add_scatter(i, env_fn, gi, ti)
    cfg.add_scatter(j, env_fn, gj, tj)
    return cfg


# =====================================================================
# L3 · 反冲加热 → 阱逃逸
# =====================================================================
class TrapModel:
    """把散射注入的运动能量绝热消去成一条逃逸率（两档）。

    mode='budget'   （P-档·现象学）：每次散射以固定概率 p_esc=1/N_crit 逃逸，
                     N_crit = U/(κ·E_r) 为"烧穿阱深所需散射数"。
                     γ_esc = R_sc·p_esc → loss_esc = 1−exp(−N_sc/N_crit)。
                     复现：累积、阱深依赖、∝1/I 阈值感。
    mode='arrhenius'（T-档·温度 ODE）：dT/dt = (2/3)κ·E_r·R_sc − Λ·T，
                     γ_esc = R_sc·exp(−U/T)。复现：冷却稳态 + 指数陡峭阈值。
                     冷却取多普勒式 Λ·T（T0 为无光时的极限温度，仅作初值）。

    诚实声明：depth_K=None → 关闭，绝不给"加热损失"数字。
    """

    def __init__(self, lam_m, depth_K, kappa_geo=2.0, T0_K=0.0,
                 cool_rate_ns=0.0, mode='budget', boltzmann_floor=0.0):
        self.E_r_K = recoil_temperature_K(lam_m)    # 反冲温度（K）
        self.U_K = float(depth_K) if depth_K is not None else None
        self.kappa = float(kappa_geo)
        self.T0_K = float(T0_K)
        self.cool_rate_ns = float(cool_rate_ns)     # 冷却速率 Λ /ns
        self.mode = str(mode)
        self.floor = float(boltzmann_floor)

    @property
    def enabled(self):
        return self.U_K is not None and self.U_K > 0.0

    @property
    def n_crit(self):
        """烧穿阱深所需散射数 = U/(κ·E_r)（锚 A9：1 mK → ~4664）。"""
        if not self.enabled:
            return float('inf')
        return self.U_K / (self.kappa * self.E_r_K)

    def escape_rate_ns(self, R_tot_ns, n_sc, T_K):
        """瞬时逃逸率（/ns）。R_tot_ns=总散射事件率(/ns)。"""
        if not self.enabled or R_tot_ns <= 0.0:
            return 0.0
        if self.mode == 'budget':
            return R_tot_ns / self.n_crit
        # arrhenius：Boltzmann 尾
        T_K = max(float(T_K), 1e-9)
        p = np.exp(-self.U_K / T_K)
        p = max(p, self.floor)
        return R_tot_ns * p

    def heat_rate_K_per_ns(self, R_tot_ns):
        """散射加热升温速率 dT/dt（K/ns），不含冷却。R_tot_ns 单位 /ns。"""
        return (2.0 / 3.0) * self.kappa * self.E_r_K * R_tot_ns
