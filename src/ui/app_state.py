# -*- coding: utf-8 -*-
"""共享状态层：能级组 / 磁场 / 地址光 的单一数据源。

目标是从页面和控件间的裸属性访问收口到统一 store：

  LevelGroupStore  能级组列表 + 颜色分配 + 排序（唯一增删入口）
  FieldState       B 场 / 电场 / 地址光 cfg（唯一读写入口）

各页面组件持有 store 引用（构造注入），不直接持有对方内部状态；
store 变更经 pyqtSignal 广播（页面订阅刷新），不散布同步调用。
"""
from PyQt5.QtCore import QObject, pyqtSignal

from ..core import core
from .ui_single import COLOR_CYCLE


class LevelGroupStore(QObject):
    """能级组的单一数据源：增删 / 颜色 / 排序 / 变更事件。"""

    changed = pyqtSignal()          # 增删后广播（页面订阅刷新）

    def __init__(self, cache, parent=None):
        super().__init__(parent)
        self.cache = cache
        self.groups = []            # [{'n','l','j','label','color'}, ...]
        self._color_counter = 0

    # ---------- 查询 ----------
    def sorted_by_energy(self):
        """按能量升序排列的能级组列表。"""
        return sorted(
            self.groups,
            key=lambda g: self.cache.E_mhz(g['n'], g['l'], g['j']))

    def find(self, n, l, j):
        for g in self.groups:
            if (g['n'], g['l'], g['j']) == (n, l, j):
                return g
        return None

    # ---------- 变更（唯一入口） ----------
    def add(self, n, l, j):
        """添加能级组（校验 + 自动标签/颜色）；不合法返回 (False, msg)。"""
        ok, msg = core.validate_state(n, l, j)
        if not ok:
            return False, msg
        group = {
            'n': n, 'l': l, 'j': j,
            'label': core.auto_label(n, l, j),
            'color': self._assign_color(),
        }
        self.groups.append(group)
        self.changed.emit()
        return True, group

    def remove_at(self, idx):
        if 0 <= idx < len(self.groups):
            del self.groups[idx]
            self.changed.emit()

    def _assign_color(self):
        color = COLOR_CYCLE[self._color_counter % len(COLOR_CYCLE)]
        self._color_counter += 1
        return color


class FieldState(QObject):
    """磁场 / 电场 / 地址光 的单一数据源（读 field 面板 + 写 address_cfg）。"""

    changed = pyqtSignal()          # B/E/光变化后广播

    def __init__(self, field_panel, parent=None):
        super().__init__(parent)
        self.field = field_panel    # fields.FieldPanel（读 B/E/光场）
        self.address_cfg = {
            'enabled': False,
            'intensity': 0.0,
            'frequency_hz': 0.0,
            'polarization': 'π',
        }

    @property
    def B_gauss(self):
        return float(self.field.b_gauss())

    @property
    def E_vpcm(self):
        return float(self.field.e_vpcm())

    def sync_address_from_lights(self):
        """第一份光同步到 address_cfg（静态页能级图/频谱计算）。"""
        c = self.field.first_cfg()
        if c:
            self.address_cfg.update(
                frequency_hz=c['frequency_hz'],
                intensity=c['intensity'],
                polarization=c['polarization'])
            if c.get('elliptical'):
                self.address_cfg['elliptical'] = dict(c['elliptical'])
            else:
                self.address_cfg.pop('elliptical', None)
            self.address_cfg['enabled'] = True
        else:
            self.address_cfg['enabled'] = False
