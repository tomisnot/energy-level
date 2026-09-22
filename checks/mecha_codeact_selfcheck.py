# -*- coding: utf-8 -*-
"""机甲 CodeAct + async + 可复现指针 + 行为预览自检（Phase 4）。

  A CodeAct 编排：一次调用多步（set→set→run）经 choke point，事件归因 actor=codeact。
  B CodeAct 边界（D1/D3，含致死性）：命名空间只暴露 registry/session/ops 门面；
    NOT_EXPOSED 显式列出裸 core 名；**绕过 session.set 直改状态 → 被拒**；
    import 裸引擎 → 被拒；安全内建有 len 但无 __import__/open/eval。
  C 行为预览（D7）：不跑演化答"基组够不够"（driven/basis_enough），且期间**无 run 事件**。
  D 可复现产物（D8）：run 落 config+RunSpec(pkl)+result(npz)+manifest；产物→配置链可查；
    rerun(aid) 从产物重现；内容寻址（同配置同 aid，二次提交缓存命中）。
  E async job：submit→poll→done；**断线重连**（新 JobRunner）按 id 取回结果；cancel 结构就位。

真跑最小演化（单体 mixed，6 态，10 ns）——秒级。用临时 root（不污染 .mecha）。
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from arc import Rubidium87                                 # noqa: E402
from src.core.static_atom import ArcCache                  # noqa: E402
from src.core import runspec as RS                         # noqa: E402
from src.core import waveforms as WF                       # noqa: E402
from src.core.atomic_levels import state_key               # noqa: E402
from mecha.session import Session                      # noqa: E402
from mecha.ops import Ops, load_artifact, artifact_dir  # noqa: E402
from mecha.jobs import JobRunner                       # noqa: E402
from mecha import codeact as CA                        # noqa: E402
from mecha import runtime                              # noqa: E402
from src.adapter import EngineAdapter                      # noqa: E402

runtime.set_engine(EngineAdapter())   # 注入领域引擎（mecha 零领域 import）

C = Checker('mecha_codeact')

# ---- 最小可运行配置（单体 mixed，60S/60P，共振光波，10 ns）----
_TMP = tempfile.mkdtemp(prefix='mecha_p4_')
_cache = ArcCache(Rubidium87())
_GROUPS = [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]
_st = RS.build_states(_cache, _GROUPS, 0.0, 0.0)
_freq = abs(_cache.E_mhz(60, 1, 1.5) - _cache.E_mhz(60, 0, 0.5))
_wave = WF.Wave('opt', freq_mhz=_freq,
                envelope=WF.SquareEnv(0.0, 1.0, 1.0, 8.0), pol='π',
                intensity_mwcm2=1.0)
_CFG = {'groups.list': _GROUPS, 'field.B_gauss': 0.0, 'field.E_vpcm': 0.0,
        'dynamic.mode': 0, 'dynamic.method': 'mixed', 'dynamic.t_max': 10.0,
        'dynamic.npts': 11, 'dynamic.decay_index': 1,
        'dynamic.psi_single': state_key(_st[0]),
        'dynamic.waves_single': WF.serialize_waves([_wave])}

_sess = Session(root=_TMP)
for _k, _v in _CFG.items():
    _sess.set(_k, _v, actor='setup', reason='判据初始化配置')
_ops = Ops(_sess)

# ---------------- A. CodeAct 编排（经 choke point） ----------------
code_ok = (
    "session.set('field.B_gauss', 3.0, reason='codeact 调 B')\n"
    "session.set('dynamic.t_max', 12.0, reason='codeact 加长')\n"
    "result = {'B': session.get('field.B_gauss'),\n"
    "          't': session.get('dynamic.t_max'),\n"
    "          'nkeys': len(registry.all_settable_keys())}\n")
ra = CA.run_codeact(code_ok, _sess, _ops)
codeact_events = [e for e in _sess.activity(0) if e['actor'] == 'codeact']
ra_math = CA.run_codeact("result = round(math.sin(math.pi / 2), 9)", _sess, _ops)
C.group('A CodeAct 多步编排经 choke point（表达自由 + 契约继承）', [
    ('脚本 ok=True、result 回传', ra['ok'] is True
     and ra.get('result', {}).get('B') == 3.0
     and ra.get('result', {}).get('t') == 12.0),
    ('两次 set 各产生一条事件（n_events==2）', ra.get('n_events') == 2),
    ('事件归因 actor=codeact（M1）', len(codeact_events) == 2),
    ('registry 门面可查（len(all_settable_keys)）',
     ra.get('result', {}).get('nkeys', 0) >= 20),
    ('math 可用（sin(pi/2)==1）——科学编排不再手写 PI',
     ra_math['ok'] is True and ra_math.get('result') == 1.0),
])

# ---------------- B. CodeAct 边界（D1/D3，致死性） ----------------
door = CA.run_codeact("session._snapshot['field.B_gauss'] = 999\nresult = 1",
                      _sess, _ops)
imp = CA.run_codeact("import src.core.runspec as RS\nresult = RS.SPEC_VERSION",
                     _sess, _ops)
opn = CA.run_codeact("f = open('x.txt', 'w')\nresult = 1", _sess, _ops)
bnd = CA.run_codeact("result = NOT_EXPOSED", _sess, _ops)
has_len = 'len' in CA.SAFE_BUILTINS
no_escape = all(k not in CA.SAFE_BUILTINS
                for k in ('__import__', 'open', 'eval', 'exec', 'compile'))
C.group('B CodeAct 边界：只调语义层、不碰裸引擎（D1/D3，含致死性）', [
    ('**致死性**：绕过 set 直改 _snapshot → 被拒（AttributeError）',
     door['ok'] is False and door.get('error', {}).get('kind') == 'AttributeError'),
    ('被拒的侧门写入**没有**改到快照（B_gauss 仍是 3.0）',
     _sess.get('field.B_gauss') == 3.0),
    ('import 裸 core → 被拒（无 __import__）',
     imp['ok'] is False and imp.get('error', {}).get('kind') in
     ('ImportError', 'NameError')),
    ('open 文件 → 被拒（无 open 内建）',
     opn['ok'] is False and opn.get('error', {}).get('kind') == 'NameError'),
    ('NOT_EXPOSED 显式列出裸 core 名（D3）',
     bnd['ok'] is True and 'runspec' in (bnd.get('result') or [])
     and 'evolve_run' in (bnd.get('result') or [])),
    ('安全内建有 len 但无 __import__/open/eval/exec/compile',
     has_len and no_escape),
])

# ---------------- C. 行为预览（D7，不跑演化） ----------------
seq_before = _sess.bus.last_seq
pv = _ops.preview()
run_events_during = [e for e in _sess.activity(seq_before) if e['kind'] == 'run']
C.group('C 行为预览：不跑演化答"基组够不够"（D7）', [
    ('preview 报基组规模 n_states>0', pv.get('n_states', 0) > 0),
    ('preview 找到被驱动的跃迁（driven>=1，含 det/rabi）',
     len(pv.get('driven', [])) >= 1
     and 'det_mhz' in pv['driven'][0] and 'rabi_rad_s' in pv['driven'][0]),
    ('preview 回答 basis_enough（基组够不够）',
     isinstance(pv.get('basis_enough'), bool)),
    ('**不跑演化**：preview 期间无 run 事件', len(run_events_during) == 0),
])

# ---------------- D. 可复现产物（D8）+ 内容寻址（rid） ----------------
r1 = _ops.run(actor='ai', reason='产物落地测试')
aid = r1['artifact_id']
# 产物落 `session.bus.root`（= <项目根>/.mecha，P0-1 后 root 语义 = 项目根）
_aroot = _sess.bus.root
adir = artifact_dir(_aroot, aid)
loaded = load_artifact(_aroot, aid)
files = sorted(os.listdir(adir)) if os.path.isdir(adir) else []
rr = _ops.rerun(aid, actor='ai', reason='可复现重跑')
r2 = _ops.run(actor='ai', reason='二次提交（同配置）', use_cache=True)
C.group('D 可复现产物：config+RunSpec+result 落盘、产物→配置链、rerun、内容寻址', [
    ('run 落了产物目录（manifest+runspec.pkl+result.npz）',
     'manifest.json' in files and 'runspec.pkl' in files and 'result.npz' in files),
    ('产物→配置链可查（manifest.config == 当次快照）',
     loaded is not None and loaded[2].get('config', {}).get('dynamic.method')
     == 'mixed'),
    ('rerun(aid) 从产物重现（不重新 build）',
     rr['ok'] is True and rr['summary'].get('n_states')
     == r1['summary'].get('n_states')),
    ('内容寻址：同配置 → 同 aid', r2.get('artifact_id') == aid),
    ('同配置二次提交缓存命中（零算力，cached=True）', r2.get('cached') is True),
])

# ---------------- E. async job（可发起/轮询/断线找回） ----------------
jr = JobRunner(_ops)
jid = jr.submit(reason='async 判据', actor='ai')
final = None
for _ in range(120):
    j = jr.poll(jid)
    if j['status'] in ('done', 'error', 'cancelled'):
        final = j
        break
    time.sleep(0.25)
# 断线重连：另起一个 JobRunner（模拟重连的客户端），按 id 取回
jr2 = JobRunner(_ops)
rec = jr2.poll(jid)
jobs_list = jr2.list_jobs()
C.group('E async job：发起/轮询/断线重连按 id 取回（R2 回合数）', [
    ('submit 返回 job_id', isinstance(jid, str) and len(jid) > 0),
    ('job 跑完（status=done）且结果 ok',
     final is not None and final['status'] == 'done'
     and (final.get('result') or {}).get('ok') is True),
    ('进度用三槽 [frac,cancel,stage]（对齐 evolve_run）',
     isinstance(final.get('progress'), list) and len(final['progress']) == 3),
    ('**断线重连**：新 JobRunner 按 id 取回同一结果',
     rec['status'] == 'done' and (rec.get('result') or {}).get('ok') is True),
    ('list_jobs 能列出在册 job（找回）',
     any(x['id'] == jid for x in jobs_list)),
    ('job 结果带 artifact_id（可复现链）',
     bool((rec.get('result') or {}).get('artifact_id'))),
])

sys.exit(0 if C.summary() else 1)
