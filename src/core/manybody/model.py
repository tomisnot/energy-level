# -*- coding: utf-8 -*-
"""多体模型卡（纯数据，可序列化）：任意点云 + 逐原子 2 能级 + 项开关。

设计口径：
- 几何完全自由：atoms 就是逐原子 (x,y,z) 点云，无任何晶格对称性假设；可选 `periods_um`
  开启 xy 环面最小像与跨胞元 winding；缺省仍为开放边界；
- 每原子局部空间 = 2 个选定子能级 (g, r)，身份用 (n,l,j,m) 四元组；
- 相互作用 = 对空间 V_dd 全精确标定（assemble.py），项开关 I1/I2/I3；
- 单位：位置 µm，Ω/δ/V/J/A MHz，时间 ns。
"""
import json
import numpy as np

TEMPLATES = ('full', 'ising', 'xy', 'potts3')
TERMS = ('I1', 'I2', 'I3')          # I1=nn V, I2=交换 J, I3=双激发 A
# 模板 → `terms` 预设。template 只是"相互作用项开关"的预设选择器，不是独立物理开关。
# 教科书对应：Ising = 密度-密度 Σ V n_i n_j ⇒ I1；XY = 交换 Σ J(σ⁺σ⁻+h.c.) ⇒ I2。
# 横向场由逐原子 Ω（σˣ）提供，与模板无关。
# **显式 `terms` 优先**（见 `__init__`）：给 terms 就以 terms 为准，template 只供预设。
TEMPLATE_TERMS = {'full': TERMS, 'ising': ('I1',), 'xy': ('I2',)}


class ManyBodyModel:
    """多体页的"模型卡"：一次标定、多次运行、可存档往返。"""

    def __init__(self, template='ising', terms=None, B_gauss=0.0,
                 n_window=3, r_cut_um=None, periods_um=None):
        if template not in TEMPLATES:
            raise ValueError(f'template 须为 {TEMPLATES}，收到 {template!r}')
        # terms=None ⇒ 取 template 的预设；显式给 terms（含空表 = 不要任何相互作用项）
        # ⇒ **以 terms 为准**。两种写法语义明确、不重叠。
        if terms is None:
            terms = TEMPLATE_TERMS.get(template, ('I1',))
        bad = [t for t in terms if t not in TERMS]
        if bad:
            raise ValueError(f'terms 含未知项 {bad}，可选 {TERMS}')
        self.template = template
        self.terms = list(terms)
        self.B_gauss = float(B_gauss)
        self.n_window = int(n_window)
        self.r_cut_um = None if r_cut_um is None else float(r_cut_um)
        if periods_um is None:
            self.periods_um = None
        else:
            pp = tuple(map(float, periods_um))
            if len(pp) != 2 or any((not np.isfinite(x) or x <= 0) for x in pp):
                raise ValueError('periods_um 须为两个正数 (Lx,Ly)')
            self.periods_um = pp
        self.atoms = []            # [{id, g:(n,l,j,m), r:(n,l,j,m), pos:(x,y,z)}]
        self.beams = []            # I5 串扰标定用（M0 只占位，不接线）

    # ---- 摆点（生成器只是便利，产出仍是任意点云） ----
    def add_atom(self, g, r, pos, Omega_mhz=0.0, delta_mhz=0.0,
                 gamma_mhz=0.0, Omega_phase_rad=0.0):
        """g/r = (n,l,j,m) 四元组；pos = (x,y,z) µm；Ω/δ/Γ = 逐原子驱动/失谐/衰减（MHz）。

        Ω/δ/Γ 是**运行级**量（calibrate 只读 g/r/pos，不读驱动 ⇒ 改驱动不重标定）；
        存进原子条目只为"模型卡自带一份默认驱动"，build_run 时抽成驱动数组。
        返回原子 id。
        """
        g = tuple(map(float, g))
        r = tuple(map(float, r))
        for q in (g, r):
            if len(q) != 4:
                raise ValueError(f'能级须为 (n,l,j,m)，收到 {q!r}')
        pos = tuple(map(float, pos))
        if len(pos) != 3:
            raise ValueError(f'位置须为 (x,y,z) µm，收到 {pos!r}')
        i = len(self.atoms)
        # ``Omega_mhz`` remains the signed transverse amplitude for backward
        # compatibility.  A separate phase is the natural JSON/Runspec
        # representation of a complex transverse field:
        # H_Omega = (Omega sigma+ + Omega* sigma-)/2.
        if isinstance(Omega_mhz, complex):
            if abs(Omega_mhz.imag) > 1e-12 and float(Omega_phase_rad) == 0.0:
                Omega_phase_rad = float(__import__('math').atan2(
                    Omega_mhz.imag, Omega_mhz.real))
                Omega_mhz = abs(Omega_mhz)
            else:
                Omega_mhz = float(Omega_mhz.real)
        self.atoms.append(dict(id=i, g=g, r=r, pos=pos,
                               Omega_mhz=float(Omega_mhz),
                               Omega_phase_rad=float(Omega_phase_rad),
                               delta_mhz=float(delta_mhz),
                               gamma_mhz=float(gamma_mhz)))
        return i

    def add_array(self, kind, n, spacing_um=5.0, jitter=0.0, rng=None,
                  g=None, r=None, **kw):
        """便利摆点：chain/square/ring。全部原子共用同一对 (g,r) 能级。

        ⚠ **仅 Python API**：登记册路径（`runspec_mb.model_from_params`）只调
        `add_atom`（逐原子显式坐标），**不调本函数**；要复现抖动请显式给 `seed=`
        （经 `kw`）或自建 `rng`。登记册若要按模板摆阵列，需要新增 `manybody.array`
        键并把 seed 接上；当前不支持。
        """
        import numpy as np
        if g is None or r is None:
            raise ValueError('add_array 需要 g/r 能级四元组')
        pos = []
        if kind == 'chain':
            pos = [(i * spacing_um, 0.0, 0.0) for i in range(n)]
        elif kind == 'square':
            side = int(round(n ** 0.5))
            if side * side != n:
                raise ValueError('square 需要 n 为完全平方数')
            pos = [((i % side) * spacing_um, (i // side) * spacing_um, 0.0)
                   for i in range(n)]
        elif kind == 'ring':
            for i in range(n):
                a = 2 * np.pi * i / n
                R = n * spacing_um / (2 * np.pi)
                pos.append((R * np.cos(a), R * np.sin(a), 0.0))
        else:
            raise ValueError(f'未知生成器 {kind!r}')
        pos = np.asarray(pos, float)
        if jitter:
            if rng is None:
                rng = np.random.default_rng(kw.get('seed'))
            pos = pos + rng.normal(0.0, float(jitter), pos.shape)
        ids = [self.add_atom(g, r, p,
                             Omega_mhz=kw.get('Omega_mhz', 0.0),
                             Omega_phase_rad=kw.get('Omega_phase_rad', 0.0),
                             delta_mhz=kw.get('delta_mhz', 0.0),
                             gamma_mhz=kw.get('gamma_mhz', 0.0)) for p in pos]
        return ids

    def positions(self):
        import numpy as np
        return np.array([a['pos'] for a in self.atoms], float)

    # ---- 序列化（存档往返逐位一致，M0 锚 A0c） ----
    def to_dict(self):
        return dict(template=self.template, terms=list(self.terms),
                    B_gauss=self.B_gauss, n_window=self.n_window,
                    r_cut_um=self.r_cut_um,
                    periods_um=(None if self.periods_um is None
                                else list(self.periods_um)),
                    atoms=[dict(id=a['id'], g=list(a['g']), r=list(a['r']),
                                pos=list(a['pos']),
                                Omega_mhz=a.get('Omega_mhz', 0.0),
                                Omega_phase_rad=a.get('Omega_phase_rad', 0.0),
                                delta_mhz=a.get('delta_mhz', 0.0),
                                gamma_mhz=a.get('gamma_mhz', 0.0))
                           for a in self.atoms],
                    beams=list(self.beams))

    @classmethod
    def from_dict(cls, d):
        # `energy_window_mhz` 不是支持的键：能量窗需要 `ref_energy_mhz`，schema 里
        # 没有这个键；中间态截断由 `n_window`（伙伴态窗）+ `guard_mhz`（近简并剔除）
        # 覆盖。输入里若带它，明确忽略并留给调用方看见（不静默当作已生效）。
        m = cls(d['template'], d.get('terms'), d.get('B_gauss', 0.0),
                d.get('n_window', 3), d.get('r_cut_um'),
                d.get('periods_um'))
        for a in d['atoms']:
            m.atoms.append(dict(id=a['id'], g=tuple(a['g']),
                                r=tuple(a['r']), pos=tuple(a['pos']),
                                Omega_mhz=float(a.get('Omega_mhz', 0.0)),
                                Omega_phase_rad=float(a.get('Omega_phase_rad', 0.0)),
                                delta_mhz=float(a.get('delta_mhz', 0.0)),
                                gamma_mhz=float(a.get('gamma_mhz', 0.0))))
        m.beams = list(d.get('beams', []))
        return m

    def to_json(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
