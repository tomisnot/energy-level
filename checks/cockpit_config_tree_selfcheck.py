# -*- coding: utf-8 -*-
"""驾驶舱配置态结构树自检（config_tree）——需求②「AI 模式看 web 配置树」的数据层契约。

`config_tree` = 最近一条 `registry_schema` 事件的 schema **join** 每键从 `set` 事件重建的
现值。守黑匣子纯净（T5）：只读 journal、**不 import mecha**（本判据用 mecha.bus 只是**造
测试数据**，被测的 backend 不依赖它）。

  A. schema join 现值 → 按 family 分组的树；每键带 schema 元数据（unit/type…）+ 现值 + 归因。
  B. 契约内但未 set 的键 → value=None、set=None（与"设过又清除"可辨，不冒充有值）。
  C. **大 schema 被 I4 体积闸截成 `$truncated` 指针 → config_tree 顺 where 读回原文**（不丢
     schema）；致死性：先证明事件 after 确被截断，再证明树仍拿回全部键。
  D. **致死性**：无 registry_schema 事件 → has_schema=False、families 空（树不凭空造）；
     此时 set 过的键落 orphans（不静默丢）。

用临时 root；纯读，不跑引擎（快）。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from mecha.bus import Bus                              # noqa: E402  (仅用于造 journal)
from cockpit import backend                            # noqa: E402  (被测：只读 journal)

C = Checker('cockpit_config_tree')

# ---- 造一份 journal：大 schema（>4096B 触发体积闸）+ 两条 set ----
tmp = tempfile.mkdtemp(prefix='cfgtree_')
bus = Bus(root=tmp)
tmp = bus.root                          # P0-1：数据根 = <项目根>/.mecha
_PAD = 'x' * 5000                       # 确保 schema 的 after 超 _MAX_INLINE_BYTES 被截断
schema = [
    {'name': 'field.B_gauss', 'family': 'config', 'type': 'float', 'unit': 'Gauss',
     'desc': '静磁场（Zeeman）', 'tier': 'run'},
    {'name': 'dynamic.method', 'family': 'config', 'type': 'str',
     'enum': ['mixed', 'sambe', 'hard'], 'desc': '引擎'},
    {'name': 'BW_OPT', 'family': 'engine', 'type': 'float', 'readonly': True,
     'tier': 'run', 'desc': _PAD},
    {'name': 'solve.MAX_STEP', 'family': 'solve', 'type': 'float', 'unit': 'ns',
     'desc': '起步最大步长'},
]
bus.append(actor='system', kind='registry_schema', target='registry', after=schema,
           reason='schema', provenance='t', reversibility='none')
bus.append(actor='ai', kind='set', target='field.B_gauss', before=None, after=21.7,
           reason='Zeeman 调谐', provenance='session.set', reversibility='undo')
bus.append(actor='gui', kind='set', target='dynamic.method', before=None, after='sambe',
           reason='换引擎', provenance='session.set', reversibility='undo')

tree = backend.config_tree(tmp)
cfg = {r['name']: r for r in tree['families'].get('config', [])}
solve = {r['name']: r for r in tree['families'].get('solve', [])}
schema_ev = [e for e in bus.since(0) if e.get('kind') == 'registry_schema'][0]

C.group('A schema join 现值 → 树（family 分组 + 元数据 + 值 + 归因）', [
    ('has_schema=True', tree['has_schema'] is True),
    ('config 族含 field.B_gauss', 'field.B_gauss' in cfg),
    ('B_gauss 现值 == 21.7（从 set 事件 join）', cfg.get('field.B_gauss', {}).get('value') == 21.7),
    ('B_gauss 带 schema 元数据 unit=Gauss', cfg.get('field.B_gauss', {}).get('unit') == 'Gauss'),
    ('B_gauss 归因 reason 来自 set 事件',
     cfg.get('field.B_gauss', {}).get('set', {}).get('reason') == 'Zeeman 调谐'),
    ('engine / solve 各自成族', 'engine' in tree['families'] and 'solve' in tree['families']),
])

C.group('B 契约内但未 set → value=None + set=None（不冒充有值）', [
    ('solve.MAX_STEP 在树里', 'solve.MAX_STEP' in solve),
    ('未 set → value None', solve.get('solve.MAX_STEP', {}).get('value') is None),
    ('未 set → set None', solve.get('solve.MAX_STEP', {}).get('set') is None),
])

C.group('C 大 schema 经 I4 体积闸截断 → config_tree 读回原文（不丢）', [
    ('schema 事件 after 确被截成 $truncated 指针',
     isinstance(schema_ev.get('after'), dict) and schema_ev['after'].get('$truncated') is True),
    ('config_tree 仍读回全部 4 键（顺 where 取回原文）', tree['n_keys'] == 4),
    ('读回的 BW_OPT desc 完整（未被 head 截断残留）',
     any(r.get('desc') == _PAD for r in tree['families'].get('engine', []))),
])

# ---- D 致死性：无 schema 事件 → 空树 + orphans ----
tmp2 = tempfile.mkdtemp(prefix='cfgtree2_')
b2 = Bus(root=tmp2)
tmp2 = b2.root                          # P0-1：数据根 = <项目根>/.mecha
b2.append(actor='ai', kind='set', target='field.B_gauss', before=None, after=1.0,
          reason='r', provenance='p', reversibility='undo')
t2 = backend.config_tree(tmp2)
C.group('D 致死性：无 registry_schema → has_schema=False + families 空（树真依赖 schema 事件）', [
    ('has_schema=False', t2['has_schema'] is False),
    ('families 空（不凭空造树）', t2['families'] == {}),
    ('无 schema 的 set 键落 orphans（不静默丢）',
     any(o['name'] == 'field.B_gauss' for o in t2['orphans'])),
])

# ---- E 真路径：serve.py 的 registry_schema emit 表达式（真 registry schema）→ config_tree 读回 ----
from mecha import runtime                              # noqa: E402
from src.adapter import EngineAdapter                      # noqa: E402
from mecha import registry as REG                      # noqa: E402
from mecha.session import Session as _Sess             # noqa: E402

runtime.set_engine(EngineAdapter())
tmp3 = tempfile.mkdtemp(prefix='cfgtree3_')
s3 = _Sess(root=tmp3)
tmp3 = s3.bus.root                      # P0-1：数据根 = <项目根>/.mecha
# 与 src/adapter/serve.py 逐字一致的 emit（真 schema、真体积闸）
s3.bus.append(actor='system', kind='registry_schema', target='registry',
              after=REG.list_params(include_internal=True)['params'],
              reason='配置态 schema（供驾驶舱结构树）',
              provenance='adapter.serve', reversibility='none')
s3.set('field.B_gauss', 21.7, actor='ai', reason='Zeeman')
t3 = backend.config_tree(tmp3)
_c3 = {r['name']: r for r in t3['families'].get('config', [])}
C.group('E 真路径：serve.py 的 registry_schema emit（真 schema）→ config_tree 读回', [
    ('真 schema 落盘并被读回（has_schema）', t3['has_schema'] is True),
    ('config 族含真键 field.B_gauss', 'field.B_gauss' in _c3),
    ('engine 族非空（真只读常量族）', len(t3['families'].get('engine', [])) > 0),
    ('真 B_gauss 现值 21.7 + 归因 reason',
     _c3.get('field.B_gauss', {}).get('value') == 21.7
     and _c3.get('field.B_gauss', {}).get('set', {}).get('reason') == 'Zeeman'),
])

sys.exit(0 if C.summary() else 1)
