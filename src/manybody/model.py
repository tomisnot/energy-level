# -*- coding: utf-8 -*-
"""多体模型卡（纯数据，可序列化）：任意点云 + 逐原子 2 能级 + 项开关。

设计口径（docs/多体页-调查与开发计划.md §3，2026-09-17 裁决）：
- 几何完全自由：atoms 就是逐原子 (x,y,z) 点云，无任何晶格对称性假设；
- 每原子局部空间 = 2 个选定子能级 (g, r)，身份用 (n,l,j,m) 四元组；
- 相互作用 = 对空间 V_dd 全精确标定（assemble.py），项开关 I1/I2/I3；
- 单位：位置 µm，Ω/δ/V/J/A MHz，时间 ns。
"""
import json

TEMPLATES = ('ising', 'xy', 'potts3')
TERMS = ('I1', 'I2', 'I3')          # I1=nn V, I2=交换 J, I3=双激发 A


class ManyBodyModel:
    """多体页的"模型卡"：一次标定、多次运行、可存档往返。"""

    def __init__(self, template='ising', terms=('I1',), B_gauss=0.0,
                 n_window=3, r_cut_um=None, energy_window_mhz=None):
        if template not in TEMPLATES:
            raise ValueError(f'template 须为 {TEMPLATES}，收到 {template!r}')
        bad = [t for t in terms if t not in TERMS]
        if bad:
            raise ValueError(f'terms 含未知项 {bad}，可选 {TERMS}')
        self.template = template
        self.terms = list(terms)
        self.B_gauss = float(B_gauss)
        self.n_window = int(n_window)
        self.r_cut_um = None if r_cut_um is None else float(r_cut_um)
        self.energy_window_mhz = (None if energy_window_mhz is None
                                  else float(energy_window_mhz))
        self.atoms = []            # [{id, g:(n,l,j,m), r:(n,l,j,m), pos:(x,y,z)}]
        self.beams = []            # I5 串扰标定用（M0 只占位，不接线）

    # ---- 摆点（生成器只是便利，产出仍是任意点云） ----
    def add_atom(self, g, r, pos):
        """g/r = (n,l,j,m) 四元组；pos = (x,y,z) µm。返回原子 id。"""
        g = tuple(map(float, g))
        r = tuple(map(float, r))
        for q in (g, r):
            if len(q) != 4:
                raise ValueError(f'能级须为 (n,l,j,m)，收到 {q!r}')
        pos = tuple(map(float, pos))
        if len(pos) != 3:
            raise ValueError(f'位置须为 (x,y,z) µm，收到 {pos!r}')
        i = len(self.atoms)
        self.atoms.append(dict(id=i, g=g, r=r, pos=pos))
        return i

    def add_array(self, kind, n, spacing_um=5.0, jitter=0.0, rng=None,
                  g=None, r=None, **kw):
        """便利摆点：chain/square/ring。全部原子共用同一对 (g,r) 能级。"""
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
        ids = [self.add_atom(g, r, p) for p in pos]
        return ids

    def positions(self):
        import numpy as np
        return np.array([a['pos'] for a in self.atoms], float)

    # ---- 序列化（存档往返逐位一致，M0 锚 A0c） ----
    def to_dict(self):
        return dict(template=self.template, terms=list(self.terms),
                    B_gauss=self.B_gauss, n_window=self.n_window,
                    r_cut_um=self.r_cut_um,
                    energy_window_mhz=self.energy_window_mhz,
                    atoms=[dict(id=a['id'], g=list(a['g']), r=list(a['r']),
                                pos=list(a['pos'])) for a in self.atoms],
                    beams=list(self.beams))

    @classmethod
    def from_dict(cls, d):
        m = cls(d['template'], d['terms'], d.get('B_gauss', 0.0),
                d.get('n_window', 3), d.get('r_cut_um'),
                d.get('energy_window_mhz'))
        for a in d['atoms']:
            m.atoms.append(dict(id=a['id'], g=tuple(a['g']),
                                r=tuple(a['r']), pos=tuple(a['pos'])))
        m.beams = list(d.get('beams', []))
        return m

    def to_json(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
