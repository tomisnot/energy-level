# -*- coding: utf-8 -*-
"""GUI↔mecha 对接缝自检（app_bridge）——两模式写权门控 + 存档往返 + 实时刷新 + 致死性。

守《回归主仓库-迁移计划》§2/§4.3：
  A. 组装：bridge 捆好 bus/authority/session/ops/jobs；can_write 随模式翻。
  B. GUI 写经 choke point：事件 + 订阅回调（实时刷新）+ readback（M1/M2）。
  C. **AI 模式：GUI 写被拒（→ 据此设只读）、AI 写放行**（§2 写权互斥）+ authority_switch 记账。
  D. 交接：AI→人 需 in_flight==0（先静默 AI）。
  E. save/load 往返（新存档格式 = 扁平 session 快照，M5）。
  F. **旧格式存档被明确拒**（不写运行时旧档适配器，M5）——致死性：喂旧格式必须报错。
  G. params 表渐进披露 + 人可深挖 internal（M2/D-B）。

用临时目录当 root（不污染 `<repo>/.mecha`）；不跑引擎（快）。
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from mecha.app_bridge import AppBridge                 # noqa: E402
from mecha.authority import AuthorityError             # noqa: E402
from mecha import runtime                              # noqa: E402
from src.adapter import EngineAdapter                      # noqa: E402

runtime.set_engine(EngineAdapter())   # 注入领域引擎（mecha 零领域 import）

C = Checker('app_bridge')
tmp = tempfile.mkdtemp(prefix='bridge_')
br = AppBridge(root=tmp, mode='human')

# ---------------- A. 组装 ----------------
C.group('A 组装：bus/authority/session/ops/jobs 就位；can_write 随模式', [
    ('默认人工模式', br.mode == 'human'),
    ('human 模式 can_write(gui)=True', br.can_write('gui') is True),
    ('human 模式 can_write(ai)=False', br.can_write('ai') is False),
    ('五个部件都在',
     all(x is not None for x in (br.session, br.ops, br.jobs, br.bus, br.authority))),
])

# ---------------- B. GUI 写经 choke point + 订阅实时刷新 ----------------
seen = []
unsub = br.subscribe(lambda ev: seen.append(ev))
rec = br.write('field.B_gauss', 21.7, reason='人调磁场')      # actor='gui'
last = seen[-1] if seen else {}
C.group('B GUI 写经 choke point：事件 + 订阅回调 + readback（M1/M2/实时刷新）', [
    ('写入成功 resolved=21.7', rec['ok'] and rec['resolved'] == 21.7),
    ('readback==resolved（choke point 实测）', rec['readback'] == rec['resolved']),
    ('订阅回调收到该 set 事件（GUI 实时刷新靠它）',
     last.get('kind') == 'set' and last.get('target') == 'field.B_gauss'),
    ('事件 actor=gui、reason 非空（M1 归因）',
     last.get('actor') == 'gui' and bool(last.get('reason'))),
    ('effective() 反映写入', br.effective().get('field.B_gauss') == 21.7),
])
unsub()

# ---------------- C. AI 模式：GUI 只读、AI 可写 ----------------
sw = br.switch_mode('ai', reason='交给 AI 开车')
ai_rec = br.write('field.B_gauss', 30.0, reason='AI 调', actor='ai')
try:
    br.write('field.B_gauss', 99.0, reason='GUI 越权', actor='gui')
    gui_rejected = False
except AuthorityError:
    gui_rejected = True
C.group('C AI 模式：GUI 写被拒（据此设只读）、AI 写放行（§2 写权互斥）', [
    ('切到 ai 成功、mode=ai', sw['ok'] and br.mode == 'ai'),
    ('AI 写入成功 resolved=30.0', ai_rec['ok'] and ai_rec['resolved'] == 30.0),
    ('GUI 写入被拒（AuthorityError）', gui_rejected),
    ('can_write(gui) 现在为假', br.can_write('gui') is False),
    ('被拒后值仍是 AI 写的 30.0（GUI 没写进去）',
     br.effective().get('field.B_gauss') == 30.0),
    ('authority_switch 事件已记账',
     any(e.get('kind') == 'authority_switch' for e in br.activity(0))),
])

# ---------------- D. 交接回人工（需静默） ----------------
sw2 = br.switch_mode('human', reason='夺回控制')
C.group('D 交接：AI→人 需 in_flight==0 + 记账 + GUI 复写权', [
    ('无在飞 job 时切回成功', sw2['ok'] and br.mode == 'human'),
    ('in_flight()==0', br.in_flight() == 0),
    ('切回后 can_write(gui)=True', br.can_write('gui') is True),
])

# ---------------- E. save/load 往返（新格式 = 扁平 session 快照） ----------------
p = os.path.join(tmp, 'state.json')
br.save(p, reason='存档')
br.write('field.B_gauss', 1.0, reason='改乱')
lr = br.load(p, reason='恢复')
raw = json.load(open(p, encoding='utf-8'))
C.group('E save/load 往返（新存档格式 = 扁平 session 快照，M5）', [
    ('格式标记 mecha-session-v1', raw.get('format') == 'mecha-session-v1'),
    ('params 是扁平 registry 键（含 field.B_gauss）',
     isinstance(raw.get('params'), dict) and 'field.B_gauss' in raw['params']),
    ('load 后 B_gauss 恢复到存档值 30.0', br.effective().get('field.B_gauss') == 30.0),
    ('load 回执 n_keys 与 seqs 数一致', lr['ok'] and len(lr['seqs']) == lr['n_keys']),
])

# ---------------- F. 旧格式存档被拒（致死性；M5 不写运行时适配器） ----------------
badp = os.path.join(tmp, 'old.json')
with open(badp, 'w', encoding='utf-8') as f:
    json.dump({'groups': [], 'engine_overrides': {}, 'some_old_key': 1}, f)
try:
    br.load(badp)
    old_rejected = False
except ValueError:
    old_rejected = True
C.check('F 旧格式存档被明确拒（提示走一次性转换，不做运行时适配）', old_rejected)

# ---------------- G. params 表：渐进披露 + 人可深挖 internal ----------------
lp_def = br.params()
lp_all = br.params(include_internal=True)
C.group('G params 表：渐进披露 + 人可深挖 internal（M2/D-B）', [
    ('默认藏 internal（全量 > 默认）', lp_all['count'] > lp_def['count']),
    ('describe 单键可查（带 name）', 'name' in br.describe('field.B_gauss')),
])

sys.exit(0 if C.summary() else 1)
