# -*- coding: utf-8 -*-
"""GUI 绑定层自检（_mecha_bind）——offscreen 验证 widget↔mecha 双向绑定 + 两模式门控。

守《回归主仓库-迁移计划》§4：绑定层取代外部 param_manager，真值全在 mecha.session。
  A. 标准件工厂：初值来自 bridge、类型对。
  B. 改 widget → 经 choke point 写进 bridge + 发事件（actor=gui）。
  C. **AI 模式**：外部（AI）写 → widget 实时刷新（bus 订阅）；GUI 控件只读门控；
     GUI 改 widget 被写权门控吞掉（不落地）。
  D. register_external：非标参数（面板值 ↔ bridge）种子 + 读回。
  E. 致死性：AI 模式下 GUI 写确实被吞（对照 B 人工模式能写）——门控非摆设。

offscreen（QT_QPA_PLATFORM=offscreen）；用临时 root 不污染 .mecha。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from PyQt5.QtWidgets import QApplication, QDoubleSpinBox    # noqa: E402
from checks.checklib import Checker                         # noqa: E402
from examples.energy_level.bridge import V2Bridge           # noqa: E402
from src.ui import _mecha_bind as MB                        # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)     # offscreen
C = Checker('gui_bind')

tmp = tempfile.mkdtemp(prefix='guibind_')
EL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
br = V2Bridge.for_gui(el_root=EL_ROOT, data_root=tmp, mode='human')
binder = MB.Binder(br)

# ---------------- A. 标准件工厂 ----------------
br.write('field.B_gauss', 21.7, reason='种子', actor='gui')
sb = MB.spinbox(binder, 'field.B_gauss', 0.0, 0.0, 1e6, decimals=2,
                spec=MB.ParamSpec('float', unit='G'))
C.group('A 标准件工厂：初值来自 bridge + 类型对', [
    ('spinbox 初值 == bridge 值 21.7', abs(sb.value() - 21.7) < 1e-9),
    ('是 QDoubleSpinBox', isinstance(sb, QDoubleSpinBox)),
    ('绑定登记进 binder', any(k == 'field.B_gauss' for (_w, k, _s) in binder._binds)),
])

# ---------------- B. 改 widget → choke point 写 bridge ----------------
sb.setValue(30.0)
ev_b = [e for e in br.activity(0)
        if e.get('target') == 'field.B_gauss' and e.get('kind') == 'set']
C.group('B 人工模式：改 widget → 经 choke point 写 bridge + 发事件', [
    ('bridge 读到新值 30.0', br.read('field.B_gauss') == 30.0),
    ('产生 actor=gui 的 set 事件',
     any(e.get('actor') == 'gui' for e in ev_b)),
])

# ---------------- C. AI 模式：外部写→刷新 + 只读门控 ----------------
br.switch_mode('ai', actor='human')
binder.set_readonly(not br.can_write('gui'))
br.write('field.B_gauss', 42.0, reason='AI 调磁场', actor='ai')   # 外部写
refreshed = abs(sb.value() - 42.0) < 1e-9
sb.setValue(99.0)                      # GUI 在 AI 模式强改 → 应被写权门控吞掉
swallowed = (br.read('field.B_gauss') == 42.0)
C.group('C AI 模式：外部写→widget 实时刷新 + GUI 只读门控（§2 写权互斥）', [
    ('AI 写入后 spinbox 实时刷新到 42.0（bus 订阅）', refreshed),
    ('spinbox 已设只读（GUI 物理不可写）', sb.isReadOnly()),
    ('can_write(gui)==False', br.can_write('gui') is False),
    ('GUI 强改 widget 被门控吞掉（bridge 仍 42.0）', swallowed),
])

# ---------------- D. register_external（非标参数） ----------------
br.switch_mode('human', actor='human')
binder.set_readonly(False)
holder = {'v': [{'gamma_mhz': 1.0}]}
binder.register_external('dynamic.decays', read_fn=lambda: holder['v'],
                         write_fn=lambda val: holder.__setitem__('v', val))
seeded = br.read('dynamic.decays')
# 外部（AI）改 external 键 → 面板 apply（切 ai 模式测刷新）
br.switch_mode('ai', actor='human')
br.write('dynamic.decays', [{'gamma_mhz': 2.0}], reason='AI 改衰变', actor='ai')
applied = holder['v'] == [{'gamma_mhz': 2.0}]
C.group('D register_external：面板值 ↔ bridge（种子 + 外部改回刷面板）', [
    ('初次把面板值种进 bridge', seeded == [{'gamma_mhz': 1.0}]),
    ('AI 改该键 → write_fn 回刷面板', applied),
])

# ---------------- E. 致死性：门控非摆设 ----------------
C.check('E 致死性：人工模式能写(B) 而 AI 模式 GUI 写被吞(C)——门控确实生效',
        br.read('field.B_gauss') == 42.0 and swallowed)

sys.exit(0 if C.summary() else 1)
