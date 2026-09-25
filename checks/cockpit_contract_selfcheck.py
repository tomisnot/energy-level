# -*- coding: utf-8 -*-
"""监控端点**契约门**（迁移前置夹具）——把 `_mecha-extraction/cockpit-contract.md` §8
的「不可变行为清单」变成可执行判据 + "golden 骨架 + 逐字段 diff"能力。

## 为什么先建这个

框架化迁移（`src/v2_bind.py` 的监控段 → `mecha.cockpit`）与"去硬编码 8767"要分两批落地，
而**面板的"样子和内容"必须不变**（用户裁决）。人眼比对不可复现，所以在动任何源码之前
先把"什么算没坏"钉死：

  1. **契约断言**：对着**正在跑的端点**（进程内起一个，端口 0）逐条验形状——OPTIONS /
     4 条基础路由 / 404 两条 / CORS / 键集 / 事件排序 / `before` 语义 / `/config` 树与归因口径。
  2. **golden 骨架**：4 条路由在同一份夹具数据上的**完整 JSON 结构**（键名 + 类型 + 嵌套
     路径，**不含数值、不含数组长度**）落成 `checks/cockpit_golden.json`，并提供"拿另一个
     端点来比对"的入口（`--url`）⇒ 迁移提交可直接跑出逐字段 diff。

## 关键夹具设计（**为 `before` 的 fold 语义而生**）

写序列（5 条，seq 确定）：

    seq1 field.B_gauss = 21.7   actor=gui
    seq2 weird.orphan  = 1      actor=gui      ← 契约外键（进 orphans）
    seq3 dynamic.t_max = 200.0  actor=ai       ← 先 switch_mode('ai', 'human')
    seq4 field.B_gauss = 30.0   actor=ai       ← 同键**第二次**写 ⇒ seq1..seq3 是"窗口外前值"
    seq5 groups.list   = [...]  actor=session  ← seed（非 human/ai 归因）

随后取 `since_seq = seq4`（该键**在窗口内没有更早事件**）请求 `/history`：
**`before` 必须回 21.7 而不是 null**——这正是"只 fold 窗口内事件"那类错法的照妖镜
（错了则面板展开行显示 `∅ → 30.0`，静默错、只有人眼看面板才发现）。

## 隔离纪律

纯内存装配：`History()` 不带 sink（**不落盘**）、`V2Bridge` 手搓（不建 ARC cache）、
`_SchemaOnlyEngine` 只回 `schema()`（秒级）。**不碰仓内 `.mecha-v2`**，也不调用
`make_v2_bridge`（故不依赖 `EL_NO_AI_SERVER`，但在 run_checks 的隔离环境里同样安全）。

## 无 schema 分支：为什么第二个端点必须**后**起

本判据起**两个**端点（有 schema / 无 schema 两种 `/config` 分支）。当前实现把 bridge 绑在
**类属性**上（`_MonitorHandler.bridge`，`src/v2_bind.py:128`）⇒ 起第二个端点会**改写第一个
端点看到的 bridge**。故顺序固定：**夹具一的观测与 POST 只读验证全部做完，再起夹具二**。
（框架化后该隐患按契约 §9-D5 应消失；本判据**不对它做断言**，以免迁移时反向变红。）

## 致死性

C 组把 11 处**故意做坏的观测**喂给同一批探测器，必须逐条报出问题；干净观测必须零问题。
只绿不红的守卫等于没有（同 `docs_consistency_selfcheck` 纪律）。

用法：
    python checks\\cockpit_contract_selfcheck.py                              # 契约 + golden diff
    python checks\\cockpit_contract_selfcheck.py --url http://127.0.0.1:8767  # 拿活端点比对骨架
    python checks\\cockpit_contract_selfcheck.py --record                     # 重新生成 golden

退出码：0 全过；1 有问题。
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402
from mecha.authority import Authority, Mode                  # noqa: E402
from mecha.gate import Gate, accept_all                      # noqa: E402
from mecha.history import History                            # noqa: E402

from mecha_v2.bridge import V2Bridge                         # noqa: E402
from src.v2_bind import start_monitor_endpoint               # noqa: E402

C = Checker('cockpit_contract')
GOLDEN = os.path.join(_HERE, 'cockpit_golden.json')

#: 契约 §8 覆盖的断言点计数（每次审计执行都 +1，用于如实报告"覆盖了多少条"）。
_STATS = {'rules': 0}

#: 可观测头（其余 Server/Date/Content-Length 每次都不同：只记录、不参与比对）。
HEAD_KEEP = ('content-type', 'access-control-allow-origin',
             'access-control-allow-methods', 'access-control-allow-headers')

BODY_UNKNOWN = '{"ok": false, "error": "unknown endpoint"}'
BODY_POST = '{"ok": false, "error": "read-only monitor endpoint"}'

STATUS_KEYS = {'ok', 'mode'}
STATUS_OPTIONAL = {'pid', 'uptime_s', 'mcp_port', 'monitor_port'}
CONFIG_KEYS = {'ok', 'has_schema', 'groups', 'orphans', 'n_keys'}
CONFIG_OPTIONAL = {'n_snapshot_keys'}
ACTIVITY_EVENT_KEYS = {'kind', 'actor', 'target', 'value', 'seq', 'call_id'}
HISTORY_EVENT_KEYS = {'seq', 'kind', 'actor', 'target', 'value', 'before',
                      'after', 'reason', 'call_id'}
#: 面板实际消费的行字段（契约 §3.4）：必须都在；`prefix/type/desc` 是允许的冗余。
ROW_REQUIRED = {'name', 'short', 'value', 'unit', 'set'}
ROW_ALLOWED = ROW_REQUIRED | {'prefix', 'type', 'desc'}
ATTR_KEYS = {'reason', 'actor', 'seq'}
MODES = {'human', 'ai', 'locked'}

#: 夹具用的 schema（仿 EL 的三族：config / solve / engine），共 7 行。
#: engine 族键是**裸名**（EL registry 约定，见 `mecha_v2/bridge.py:352-355`）⇒ 树里落 prefix=""。
SCHEMA = {
    'config': {
        'dynamic.t_max': {'type': 'float', 'unit': 'ns', 'lo': 0.0, 'hi': 1e5,
                          'desc': '演化时长', 'default': 100.0},
        'field.B_gauss': {'type': 'float', 'unit': 'G', 'lo': 0.0, 'hi': 1e4,
                          'desc': '磁场', 'default': 0.0},
        'field.E_vpcm': {'type': 'float', 'unit': 'V/cm', 'lo': -1e6, 'hi': 1e6,
                         'desc': '电场', 'default': 0.0},
        'groups.list': {'type': 'list', 'unit': '', 'lo': None, 'hi': None,
                        'desc': '能级组', 'default': []},
        'static.columns': {'type': 'str', 'enum': ['off', 'basic', 'ac'], 'unit': '',
                           'lo': None, 'hi': None, 'desc': '静态列', 'default': 'off'},
    },
    'solve': {
        'solve.MAX_STEP': {'type': 'float', 'unit': '', 'lo': 0.0, 'hi': 1.0,
                           'desc': '最大步长', 'default': 1e-6},
    },
    'engine': {
        'R_CRIT': {'type': 'float', 'unit': '', 'lo': None, 'hi': None,
                   'desc': '临界半径', 'readonly': True},
    },
}

_FIXTURE_WRITES = (
    ('field.B_gauss', 21.7, 'gui', '设磁场'),
    ('weird.orphan', 1, 'gui', '契约外键'),
    ('dynamic.t_max', 200.0, 'ai', '拉长演化'),
    ('field.B_gauss', 30.0, 'ai', '再改磁场'),
)
_FIXTURE_SEED = ('groups.list', [{'n': 60, 'l': 0, 'j': 0.5,
                                  'label': '60S_{1/2}', 'color': '#1f77b4'}],
                 'session', '启动种子')
B_GAUSS_SECOND_SEQ = 4                                     # 同键第二次写（见写序列）
FIXTURE_LAST_SEQ = len(_FIXTURE_WRITES) + 1                # 4 写 + 1 种子 = 5
FIXTURE_TARGETS = [w[0] for w in _FIXTURE_WRITES] + [_FIXTURE_SEED[0]]
FIXTURE_ACTORS = [w[2] for w in _FIXTURE_WRITES] + [_FIXTURE_SEED[2]]
SCHEMA_N_ROWS = sum(len(v) for v in SCHEMA.values())       # 7
NOSCHEMA_ORPHANS = ['a.one', 'fallback.key']


class _SchemaOnlyEngine:
    """只回 `schema()` 的哑引擎：够 `/config` 走"有 schema"分支，不建 ARC cache。"""

    _cache = None

    def schema(self):
        return SCHEMA


def _bad(bad, violation, msg):
    """登记一个断言点：`violation` 为真即违规。顺带统计"已执行的断言条数"。"""
    _STATS['rules'] += 1
    if violation:
        bad.append(msg)


# ---------------------------------------------------------------- 夹具
def build_fixture():
    """夹具一（有 schema）→ (bridge, history)。纯内存：不落盘、不建 cache、不碰数据根。"""
    auth = Authority(mode=Mode.HUMAN)
    hist = History()
    gate = Gate(auth, hist, validate=accept_all)
    br = V2Bridge(gate, hist, auth, engine=_SchemaOnlyEngine())
    switched = False
    for key, value, actor, reason in _FIXTURE_WRITES:
        if actor == 'ai' and not switched:
            br.switch_mode('ai', actor='human')             # 人开闸（看门人语义）
            switched = True
        br.write(key, value, reason=reason, actor=actor)
    key, value, actor, reason = _FIXTURE_SEED
    br.session.set(key, value, actor=actor, reason=reason)   # 启动种子（非 human/ai）
    return br, hist


def build_minimal_fixture():
    """夹具二（**无 schema** 分支）→ (bridge, history)。

    刻意用**独立的最小装配**（只两个键，不套用夹具一的写序列）：这样"快照全部键进 orphans"
    这条断言读起来就是字面事实，不掺夹具一的键。
    """
    auth = Authority(mode=Mode.HUMAN)
    hist = History()
    gate = Gate(auth, hist, validate=accept_all)
    br = V2Bridge(gate, hist, auth, engine=None)             # 无引擎 ⇒ 无 schema 行
    br.write('a.one', 1, reason='无 schema 夹具', actor='gui')
    br.session.set('fallback.key', 'v', actor='session', reason='种子')
    return br, hist


# ---------------------------------------------------------------- HTTP
def http_call(method, url, payload=None):
    """→ (status, headers_lower, text)。4xx/5xx 也如实返回（不抛）。"""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, \
                r.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, \
            e.read().decode('utf-8')


def _route(method, base, path, payload=None):
    status, headers, text = http_call(method, base + path, payload)
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    return {'status': status,
            'headers': {k: headers[k] for k in HEAD_KEEP if k in headers},
            'text': text, 'json': body}


# ---------------------------------------------------------------- 结构骨架 / diff
_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def structure_of(obj):
    """JSON 的**数据无关骨架**：键名 + 类型 + 嵌套；**无数值、无数组长度**。

    数组 → `{"$list": <元素骨架>}`；元素骨架不一致 → `{"$list": {"$variants": [...]}}`
    （元素形状漂移必须抓得住）。
    """
    if isinstance(obj, dict):
        return {str(k): structure_of(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return {'$list': None}
        seen = []
        for item in obj:
            s = structure_of(item)
            if s not in seen:
                seen.append(s)
        return {'$list': seen[0] if len(seen) == 1 else {'$variants': seen}}
    if isinstance(obj, bool):
        return 'bool'
    if obj is None:
        return 'null'
    if isinstance(obj, (int, float)):
        return 'num'
    return 'str'


def _join(path, key):
    key = str(key)
    if _IDENT.match(key):
        return (path + '.' + key) if path else key
    return '%s[%r]' % (path, key)


def _walk_diff(path, gold, act, out, exemptions):
    if _exempt(path, exemptions):
        return
    # 数组骨架的"多形态"容错：golden 记了 `{"$list":{"$variants":[...]}}`（同一数组里
    # 元素形状不止一种），而实际只落在其中一种 ⇒ **任一匹配即算一致**；都不匹配才报，
    # 且与最接近的那种比、给出可读差异（否则每个键都会报"多出/缺失"，吵到看不出重点）。
    if (isinstance(gold, dict) and set(gold) == {'$list'}
            and isinstance(act, dict) and set(act) == {'$list'}):
        g, a = gold['$list'], act['$list']
        gv = g.get('$variants') if isinstance(g, dict) and set(g) == {'$variants'} else None
        av = a.get('$variants') if isinstance(a, dict) and set(a) == {'$variants'} else None
        if gv is not None or av is not None:
            gset = gv if gv is not None else [g]
            aset = av if av is not None else [a]
            if any(x == y for x in gset for y in aset):
                return
            best = min(((len(diff_skeleton(x, y, ())), x, y)
                        for x in gset for y in aset), key=lambda t: t[0])
            _walk_diff(path + ".$list", best[1], best[2], out, exemptions)
            return
    if isinstance(gold, dict) and isinstance(act, dict):
        for k in gold:
            sub = _join(path, k)
            if _exempt(sub, exemptions):
                continue
            if k not in act:
                out.append('%s: golden 有、实际缺' % sub)
            else:
                _walk_diff(sub, gold[k], act[k], out, exemptions)
        for k in act:
            sub = _join(path, k)
            if k not in gold and not _exempt(sub, exemptions):
                out.append('%s: 实际多出（golden 无；如属有意差异请登记 exemptions）' % sub)
        return
    if isinstance(gold, list) and isinstance(act, list):
        if len(gold) != len(act):
            out.append('%s: 元素数 %d → %d' % (path, len(gold), len(act)))
            return
        for i, (g, a) in enumerate(zip(gold, act)):
            _walk_diff('%s[%d]' % (path, i), g, a, out, exemptions)
        return
    if gold != act:
        out.append('%s: %r → %r' % (path, gold, act))


def _exempt(path, exemptions):
    """豁免：精确路径或前缀（`a.b` / `a.b.*`）。每条豁免必须带 why（见 golden）。"""
    for ex in exemptions or ():
        p = str(ex.get('path', ''))
        if not p:
            continue
        if path == p or path.startswith(p + '.') or path.startswith(p + '['):
            return True
        if p.endswith('.*') and path.startswith(p[:-1]):
            return True
    return False


def diff_skeleton(gold, act, exemptions=()):
    """逐字段比对骨架 → 差异列表（空 = 一致）。"""
    out = []
    _walk_diff('', gold, act, out, exemptions)
    return out


# ---------------------------------------------------------------- 观测
def observe(base, *, fixture_mode, last_seq=None):
    """抓一次完整观测：7 条严格路由 + 2 条"已知差异"探针（后者只记录、不严判）。"""
    routes = {
        'OPTIONS /status': _route('OPTIONS', base, '/status'),
        'GET /status': _route('GET', base, '/status'),
        'GET /activity': _route('GET', base, '/activity?since_seq=0'),
        'GET /history': _route('GET', base, '/history?since_seq=0'),
        'GET /config': _route('GET', base, '/config'),
        'GET /nope': _route('GET', base, '/nope'),
        'POST /status': _route('POST', base, '/status', {'mode': 'ai'}),
    }
    # 「已知差异」探针（契约 §9-D3）：迁移后 `/historyXYZ` 200→404、`/config?x=1` 404→200。
    # 只要求"有响应"（路由不该崩），不参与严格比对；golden 里单独一段 + 说明。
    probes = {
        'GET /historyXYZ': _route('GET', base, '/historyXYZ'),
        'GET /config?x=1': _route('GET', base, '/config?x=1'),
    }
    obs = {'base': base, 'routes': routes, 'probes': probes,
           'incremental': None, 'fixture': {}}
    if fixture_mode:
        evs = (routes['GET /history']['json'] or {}).get('events') or []
        seqs = [e.get('seq') for e in evs
                if isinstance(e, dict) and e.get('target') == 'field.B_gauss']
        if seqs:
            since = max(seqs)
            obs['incremental'] = {
                'since_seq': since,
                'history': _route('GET', base, '/history?since_seq=%d' % since),
                'activity': _route('GET', base, '/activity?since_seq=%d' % since),
            }
        obs['fixture'] = {'last_seq': last_seq, 'n_events': len(evs)}
    return obs


def skeleton_of(obs):
    """golden 可比对的部分：状态码 + 可观测头 + 结构骨架（数据无关）。"""
    out = {'routes': {}, 'probes': {}}
    for name, r in obs['routes'].items():
        out['routes'][name] = {'status': r['status'], 'headers': r['headers'],
                               'shape': structure_of(r['json'])}
    for name, r in obs['probes'].items():
        out['probes'][name] = {'responded': r['status'] in (200, 404),
                               'observed_status': r['status']}
    return out


# ---------------------------------------------------------------- 契约断言
def audit(obs, *, fixture_mode):
    """返回问题列表（空 = 合规）。纯函数：同一批断言可喂活端点或合成坏观测。"""
    bad = []
    R = obs.get('routes') or {}
    need = ('OPTIONS /status', 'GET /status', 'GET /activity', 'GET /history',
            'GET /config', 'GET /nope', 'POST /status')
    for name in need:
        _bad(bad, name not in R, 'A 缺路由观测：%s' % name)
    if bad:
        return bad

    def _acao(r, label):
        _bad(bad, r['headers'].get('access-control-allow-origin') != '*',
             '%s: 缺/错 Access-Control-Allow-Origin（CORS 须覆盖所有响应）' % label)

    # ---- B OPTIONS：204 + 恰好三个 CORS 头 + 无 body ----
    opt = R['OPTIONS /status']
    h = opt['headers']
    _bad(bad, opt['status'] != 204, 'B OPTIONS 状态码 %s ≠ 204' % opt['status'])
    _bad(bad, h.get('access-control-allow-origin') != '*', 'B OPTIONS 缺 ACAO:*')
    _bad(bad, h.get('access-control-allow-methods') != 'GET, OPTIONS',
         'B OPTIONS ACAM %r ≠ GET, OPTIONS' % h.get('access-control-allow-methods'))
    _bad(bad, h.get('access-control-allow-headers') != 'Content-Type',
         'B OPTIONS ACAH %r ≠ Content-Type' % h.get('access-control-allow-headers'))
    _bad(bad, bool(opt['text']),
         'B OPTIONS 不该有 body（实际 %d 字节）' % len(opt['text']))

    # ---- C /status ----
    st = R['GET /status']
    _bad(bad, st['status'] != 200, 'C /status 状态码 %s ≠ 200' % st['status'])
    _acao(st, 'C /status')
    _bad(bad, not str(st['headers'].get('content-type', '')).startswith('application/json'),
         'C /status Content-Type %r 不是 application/json' % st['headers'].get('content-type'))
    body = st['json'] if isinstance(st['json'], dict) else None
    _bad(bad, body is None, 'C /status body 不是 JSON 对象')
    if body is not None:
        ks = set(body)
        _bad(bad, not STATUS_KEYS <= ks,
             'C /status 缺必需键 %s' % sorted(STATUS_KEYS - ks))
        _bad(bad, bool(ks - STATUS_KEYS - STATUS_OPTIONAL),
             'C /status 出现契约外键 %s' % sorted(ks - STATUS_KEYS - STATUS_OPTIONAL))
        _bad(bad, body.get('ok') is not True, 'C /status ok 不是 true')
        _bad(bad, body.get('mode') not in MODES,
             'C /status mode=%r 不在 %s' % (body.get('mode'), sorted(MODES)))

    # ---- D /activity（6 键，无 before/after/reason）----
    act = R['GET /activity']
    _bad(bad, act['status'] != 200, 'D /activity 状态码 %s ≠ 200' % act['status'])
    _acao(act, 'D /activity')
    ab = act['json'] if isinstance(act['json'], dict) else {}
    _bad(bad, set(ab) != {'ok', 'events', 'mode'},
         'D /activity 顶层键 %s ≠ {ok,events,mode}' % sorted(ab))
    evs = ab.get('events') if isinstance(ab.get('events'), list) else []
    _bad(bad, not evs, 'D /activity events 不是非空列表')
    for e in evs:
        if not isinstance(e, dict) or set(e) != ACTIVITY_EVENT_KEYS:
            _bad(bad, True, 'D /activity 事件键 %s ≠ %s'
                 % (sorted(e) if isinstance(e, dict) else type(e).__name__,
                    sorted(ACTIVITY_EVENT_KEYS)))
            break
    for e in evs:
        if isinstance(e, dict) and e.get('kind') != 'set':
            _bad(bad, True, 'D /activity kind=%r ≠ set' % e.get('kind'))
            break
    seqs = [e.get('seq') for e in evs if isinstance(e, dict)]
    _bad(bad, seqs != sorted(seqs), 'D /activity 事件未按 seq 升序')

    # ---- E /history（9 键；value==after；必须带 mode）----
    his = R['GET /history']
    _bad(bad, his['status'] != 200, 'E /history 状态码 %s ≠ 200' % his['status'])
    _acao(his, 'E /history')
    hb = his['json'] if isinstance(his['json'], dict) else {}
    _bad(bad, set(hb) != {'ok', 'events', 'mode'},
         'E /history 顶层键 %s ≠ {ok,events,mode}' % sorted(hb))
    # 面板写权徽标读的是 /history 的 mode（不是 /status）——漏了它徽标恒"连接中…"
    _bad(bad, 'mode' not in hb, 'E /history 缺 mode（面板徽标依赖它）')
    _bad(bad, 'mode' in hb and hb.get('mode') not in MODES,
         'E /history mode=%r 不在 %s' % (hb.get('mode'), sorted(MODES)))
    hev = hb.get('events') if isinstance(hb.get('events'), list) else []
    _bad(bad, not hev, 'E /history events 不是非空列表')
    for e in hev:
        if not isinstance(e, dict) or set(e) != HISTORY_EVENT_KEYS:
            _bad(bad, True, 'E /history 事件键 %s ≠ %s'
                 % (sorted(e) if isinstance(e, dict) else type(e).__name__,
                    sorted(HISTORY_EVENT_KEYS)))
            break
    for e in hev:
        if isinstance(e, dict) and e.get('value') != e.get('after'):
            _bad(bad, True, 'E /history value != after（同一事件的两处投影必须同值）')
            break
    hseq = [e.get('seq') for e in hev if isinstance(e, dict)]
    _bad(bad, hseq != sorted(hseq), 'E /history 事件未按 seq 升序')
    _bad(bad, bool(hseq) and hseq != list(range(hseq[0], hseq[0] + len(hseq))),
         'E /history seq 不连续（%s…）' % hseq[:5])

    # ---- F /config ----
    cfg = R['GET /config']
    _bad(bad, cfg['status'] != 200, 'F /config 状态码 %s ≠ 200' % cfg['status'])
    _acao(cfg, 'F /config')
    cb = cfg['json'] if isinstance(cfg['json'], dict) else {}
    cks = set(cb)
    _bad(bad, not CONFIG_KEYS <= cks, 'F /config 缺必需键 %s' % sorted(CONFIG_KEYS - cks))
    _bad(bad, bool(cks - CONFIG_KEYS - CONFIG_OPTIONAL),
         'F /config 出现契约外键 %s' % sorted(cks - CONFIG_KEYS - CONFIG_OPTIONAL))
    _bad(bad, not isinstance(cb.get('has_schema'), bool), 'F /config has_schema 不是 bool')
    _bad(bad, not isinstance(cb.get('groups'), dict), 'F /config groups 不是对象')
    _bad(bad, not isinstance(cb.get('orphans'), list), 'F /config orphans 不是数组')
    _bad(bad, not isinstance(cb.get('n_keys'), int) or isinstance(cb.get('n_keys'), bool),
         'F /config n_keys 不是整数')
    groups = cb.get('groups') if isinstance(cb.get('groups'), dict) else {}
    if cb.get('has_schema'):
        _bad(bad, not groups, 'F /config has_schema=true 但 groups 为空（凭空降级？）')
        fams = list(groups)
        _bad(bad, fams != sorted(fams), 'F /config 族未按字典序：%s' % fams)
        for fam, subs in groups.items():
            if not isinstance(subs, list):
                _bad(bad, True, 'F /config groups[%s] 不是数组' % fam)
                continue
            prefixes = [s.get('prefix') for s in subs if isinstance(s, dict)]
            _bad(bad, prefixes != sorted(prefixes),
                 'F /config groups[%s] 前缀未按字典序：%s' % (fam, prefixes))
            for s in subs:
                if not isinstance(s, dict) or set(s) != {'prefix', 'rows'}:
                    _bad(bad, True, 'F /config groups[%s] 子节点键 %s ≠ {prefix,rows}'
                         % (fam, sorted(s) if isinstance(s, dict) else '?'))
                    continue
                rows = s.get('rows')
                if not isinstance(rows, list) or not rows:
                    _bad(bad, True, 'F /config groups[%s]/%s rows 不是非空数组'
                         % (fam, s.get('prefix')))
                    continue
                names = [r.get('name') for r in rows if isinstance(r, dict)]
                _bad(bad, names != sorted(names),
                     'F /config rows 未按键名字典序：%s' % names)
                for r in rows:
                    if not isinstance(r, dict):
                        _bad(bad, True, 'F /config row 不是对象')
                        break
                    miss = ROW_REQUIRED - set(r)
                    if miss:
                        _bad(bad, True, 'F /config row 缺面板消费字段 %s' % sorted(miss))
                        break
                    extra = set(r) - ROW_ALLOWED
                    if extra:
                        _bad(bad, True, 'F /config row 出现契约外字段 %s' % sorted(extra))
                        break
                    attr = r.get('set')
                    if attr is not None and (not isinstance(attr, dict)
                                             or set(attr) != ATTR_KEYS):
                        _bad(bad, True, 'F /config row.set 形状 %s ≠ {reason,actor,seq}'
                             % (sorted(attr) if isinstance(attr, dict)
                                else type(attr).__name__))
                        break
    else:
        _bad(bad, bool(groups),
             'F /config has_schema=false 但 groups 非空（不许凭空造树撑满面板）')
        _bad(bad, cb.get('n_keys') != 0,
             'F /config has_schema=false 但 n_keys=%r ≠ 0' % cb.get('n_keys'))
    orphans = cb.get('orphans') if isinstance(cb.get('orphans'), list) else []
    onames = [o.get('name') for o in orphans if isinstance(o, dict)]
    _bad(bad, onames != sorted(onames),
         'F /config orphans 未按 name 升序：%s' % onames)
    for o in orphans:
        if not isinstance(o, dict) or set(o) != {'name', 'value', 'set'}:
            _bad(bad, True, 'F /config orphan 键 %s ≠ {name,value,set}'
                 % (sorted(o) if isinstance(o, dict) else type(o).__name__))
            break

    # ---- G/H 404 两条（body 逐字）+ CORS 覆盖 ----
    nope = R['GET /nope']
    _bad(bad, nope['status'] != 404, 'G 未知路径状态码 %s ≠ 404' % nope['status'])
    _bad(bad, nope['text'] != BODY_UNKNOWN,
         'G 未知路径 body 非逐字：%r' % nope['text'][:80])
    _acao(nope, 'G 未知路径(404)')
    post = R['POST /status']
    _bad(bad, post['status'] != 404,
         'H POST 状态码 %s ≠ 404（只读：一切 POST 恒 404）' % post['status'])
    _bad(bad, post['text'] != BODY_POST, 'H POST body 非逐字：%r' % post['text'][:80])
    _acao(post, 'H POST(404)')

    # ---- P 已知差异探针：只要求"有响应" ----
    for name, r in (obs.get('probes') or {}).items():
        _bad(bad, r['status'] not in (200, 404),
             'P 探针 %s 状态码 %s（应为 200/404，路由不该崩）' % (name, r['status']))

    # ---- X 夹具相关（`--url` 活端点模式不适用）----
    if fixture_mode:
        _audit_fixture(bad, obs)
    return bad


def _audit_fixture(bad, obs):
    """只在进程内夹具模式跑：条数/不截断/`before` fold/树与归因口径。"""
    R = obs['routes']
    hev = ((R['GET /history']['json'] or {}).get('events') or [])
    exp_last = (obs.get('fixture') or {}).get('last_seq')

    # 夹具自证：写序列确定，若它漂了后面的断言就是假的
    _bad(bad, exp_last != FIXTURE_LAST_SEQ,
         'X 夹具史长 %r ≠ %d（夹具漂了，后续断言失真）' % (exp_last, FIXTURE_LAST_SEQ))
    _bad(bad, [e.get('target') for e in hev] != FIXTURE_TARGETS,
         'X 夹具事件序 %s ≠ %s' % ([e.get('target') for e in hev], FIXTURE_TARGETS))
    _bad(bad, [e.get('actor') for e in hev] != FIXTURE_ACTORS,
         'X 夹具归因序 %s ≠ %s' % ([e.get('actor') for e in hev], FIXTURE_ACTORS))
    # 不截断（也不倒序）：全量条数 == 史长
    _bad(bad, len(hev) != FIXTURE_LAST_SEQ,
         'X /history?since_seq=0 条数 %d ≠ 史长 %d（不许截断/倒序）'
         % (len(hev), FIXTURE_LAST_SEQ))
    aev = ((R['GET /activity']['json'] or {}).get('events') or [])
    _bad(bad, len(aev) != len(hev),
         'X /activity 条数 %d ≠ /history 条数 %d' % (len(aev), len(hev)))

    # `before` 的 fold 语义：窗口内无前值、全史有前值 ⇒ 必须回 21.7
    inc = obs.get('incremental')
    _bad(bad, not inc, 'X 增量观测缺失（since_seq 探针没跑？）')
    if inc:
        iev = ((inc['history'] or {}).get('json') or {}).get('events') or []
        first = iev[0] if iev else {}
        _bad(bad, inc['since_seq'] != B_GAUSS_SECOND_SEQ,
             'X 增量探针 since_seq=%r ≠ %d（夹具漂了）'
             % (inc['since_seq'], B_GAUSS_SECOND_SEQ))
        _bad(bad, not iev, 'X 增量 /history 空（since_seq 语义坏了）')
        if iev:
            _bad(bad, first.get('target') != 'field.B_gauss' or first.get('after') != 30.0,
                 'X 增量首事件不是 seq4（%r=%r）' % (first.get('target'),
                                                first.get('after')))
            _bad(bad, first.get('before') is None,
                 'X `before` 没 fold 全史：窗口内无前值时回了 null'
                 '（正确值应为 21.7；面板会显示 ∅ → 30.0，静默错）')
            _bad(bad, first.get('before') is not None and first.get('before') != 21.7,
                 'X `before`=%r ≠ 21.7' % first.get('before'))
        same = [e for e in hev if e.get('seq') == inc['since_seq']]
        _bad(bad, bool(same) and bool(iev)
             and same[0].get('before') != first.get('before'),
             'X 全量与增量对同一 seq 的 before 不一致（fold 与过滤必须分离）')
        b_first = [e for e in hev if e.get('target') == 'field.B_gauss']
        _bad(bad, bool(b_first) and b_first[0].get('before') is not None,
             'X 首写的 before=%r ≠ null' % (b_first[0].get('before') if b_first else None))

    # /config 有 schema 分支：行数 + 族/前缀 + 归因 + 契约外键
    cb = R['GET /config']['json'] or {}
    _bad(bad, cb.get('has_schema') is not True,
         'X /config has_schema=%r ≠ true（夹具带了 schema 引擎）' % cb.get('has_schema'))
    _bad(bad, cb.get('n_keys') != SCHEMA_N_ROWS,
         'X /config n_keys=%r ≠ %d（口径 = schema 行数）'
         % (cb.get('n_keys'), SCHEMA_N_ROWS))
    groups = cb.get('groups') or {}
    _bad(bad, sorted(groups) != ['config', 'engine', 'solve'],
         'X /config 族 %s ≠ [config,engine,solve]' % sorted(groups))
    cfg_pre = sorted(s.get('prefix') for s in (groups.get('config') or []))
    _bad(bad, cfg_pre != ['dynamic', 'field', 'groups', 'static'],
         'X /config config 族前缀 %s' % cfg_pre)
    eng_pre = sorted(s.get('prefix') for s in (groups.get('engine') or []))
    _bad(bad, eng_pre != [''],
         'X /config engine 族前缀 %s ≠ [""]（裸名键落 (top)）' % eng_pre)

    flat = {}
    for _fam, subs in groups.items():
        for s in subs or []:
            for r in s.get('rows') or []:
                flat[r.get('name')] = r
    _bad(bad, len(flat) != SCHEMA_N_ROWS,
         'X /config 树里行数 %d ≠ %d' % (len(flat), SCHEMA_N_ROWS))
    row = flat.get('field.B_gauss') or {}
    _bad(bad, row.get('value') != 30.0,
         'X rows[field.B_gauss].value=%r ≠ 30.0（应取当前快照）' % row.get('value'))
    _bad(bad, (row.get('set') or {}).get('actor') != 'ai',
         'X rows[field.B_gauss].set.actor=%r ≠ ai（应取全史最后一次写）'
         % (row.get('set') or {}).get('actor'))
    _bad(bad, (row.get('set') or {}).get('reason') != '再改磁场',
         'X rows[field.B_gauss].set.reason=%r' % (row.get('set') or {}).get('reason'))
    _bad(bad, row.get('short') != 'B_gauss' or row.get('prefix') != 'field',
         'X rows 前缀切分错：short=%r prefix=%r' % (row.get('short'), row.get('prefix')))
    _bad(bad, row.get('unit') != 'G' or row.get('type') != 'float',
         'X rows 元数据错：unit=%r type=%r' % (row.get('unit'), row.get('type')))
    never = flat.get('field.E_vpcm') or {}
    _bad(bad, never.get('value') is not None or never.get('set') is not None,
         'X 从未写过的键应 value=null 且 set=null（实际 %r/%r）'
         % (never.get('value'), never.get('set')))
    onames = [o.get('name') for o in (cb.get('orphans') or [])]
    _bad(bad, onames != ['weird.orphan'], 'X /config orphans=%s ≠ [weird.orphan]' % onames)
    if onames == ['weird.orphan']:
        o = cb['orphans'][0]
        _bad(bad, o.get('value') != 1 or (o.get('set') or {}).get('actor') != 'gui',
             'X orphan 归因/值错：%r / %r' % (o.get('value'), o.get('set')))
        _bad(bad, (o.get('set') or {}).get('reason') != '契约外键',
             'X orphan set.reason=%r' % (o.get('set') or {}).get('reason'))


def audit_noschema(bad, obs):
    """无 schema 分支：如实 `has_schema=false`、`groups={}`、快照全进 orphans、`n_keys=0`。"""
    cb = ((obs.get('routes') or {}).get('GET /config') or {}).get('json') or {}
    _bad(bad, cb.get('has_schema') is not False,
         'N has_schema=%r ≠ false（无 schema 引擎时应如实降级）' % cb.get('has_schema'))
    _bad(bad, cb.get('groups') != {},
         'N groups=%r ≠ {}（不许凭空造树撑满面板）' % cb.get('groups'))
    _bad(bad, cb.get('n_keys') != 0, 'N n_keys=%r ≠ 0' % cb.get('n_keys'))
    names = sorted(o.get('name') for o in (cb.get('orphans') or []))
    _bad(bad, names != NOSCHEMA_ORPHANS,
         'N orphans=%s ≠ 快照全部键 %s' % (names, NOSCHEMA_ORPHANS))


# ---------------------------------------------------------------- 致死性自测
def _good_obs():
    """供自测的最小合规观测（不参与 golden，只喂探测器）。"""
    ev = {'seq': 1, 'kind': 'set', 'actor': 'gui', 'target': 'k', 'value': 1,
          'before': None, 'after': 1, 'reason': 'r', 'call_id': ''}
    ae = {'kind': 'set', 'actor': 'gui', 'target': 'k', 'value': 1, 'seq': 1,
          'call_id': ''}
    ok_hdr = {'access-control-allow-origin': '*'}
    return {
        'routes': {
            'OPTIONS /status': {'status': 204, 'headers': {
                'access-control-allow-origin': '*',
                'access-control-allow-methods': 'GET, OPTIONS',
                'access-control-allow-headers': 'Content-Type'},
                'text': '', 'json': None},
            'GET /status': {'status': 200, 'headers': dict(
                ok_hdr, **{'content-type': 'application/json'}), 'text': '{}',
                'json': {'ok': True, 'mode': 'human'}},
            'GET /activity': {'status': 200, 'headers': ok_hdr, 'text': '{}',
                              'json': {'ok': True, 'events': [ae], 'mode': 'human'}},
            'GET /history': {'status': 200, 'headers': ok_hdr, 'text': '{}',
                             'json': {'ok': True, 'events': [ev], 'mode': 'human'}},
            'GET /config': {'status': 200, 'headers': ok_hdr, 'text': '{}',
                            'json': {'ok': True, 'has_schema': True, 'n_keys': 1,
                                     'groups': {'config': [{'prefix': 'x', 'rows': [{
                                         'name': 'x.y', 'short': 'y', 'prefix': 'x',
                                         'value': 1, 'unit': '', 'type': 'float',
                                         'desc': '', 'set': {'reason': 'r',
                                                             'actor': 'gui',
                                                             'seq': 1}}]}]},
                                     'orphans': [{'name': 'o.n', 'value': 2,
                                                  'set': None}]}},
            'GET /nope': {'status': 404, 'headers': ok_hdr, 'text': BODY_UNKNOWN,
                          'json': {'ok': False, 'error': 'unknown endpoint'}},
            'POST /status': {'status': 404, 'headers': ok_hdr, 'text': BODY_POST,
                             'json': {'ok': False,
                                      'error': 'read-only monitor endpoint'}},
        },
        'probes': {'GET /historyXYZ': {'status': 200, 'headers': {}, 'text': '{}',
                                       'json': {'ok': True}}},
        'incremental': {'since_seq': 1,
                        'history': {'status': 200, 'headers': {}, 'text': '{}',
                                    'json': {'ok': True, 'mode': 'human',
                                             'events': [ev]}},
                        'activity': {'status': 200, 'headers': {}, 'text': '{}',
                                     'json': {'ok': True, 'mode': 'human',
                                              'events': [ae]}}},
        'fixture': {'last_seq': 1, 'n_events': 1},
    }


def _path_parts(path):
    parts = []
    for seg in path.split('.'):
        m = re.match(r'^([^\[\]]+)((?:\[\d+\])*)$', seg)
        if not m:
            raise ValueError('坏路径：%r' % path)
        parts.append(m.group(1))
        parts.extend(int(i) for i in re.findall(r'\[(\d+)\]', m.group(2)))
    return parts


def _mutate(base, path, value):
    obj = json.loads(json.dumps(base))
    parts = _path_parts(path)
    node = obj
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value
    return obj


def _mutate_del(base, path):
    obj = json.loads(json.dumps(base))
    parts = _path_parts(path)
    node = obj
    for p in parts[:-1]:
        node = node[p]
    del node[parts[-1]]
    return obj


def selftest(obs_good):
    """把故意做坏的观测喂给同一批探测器 ⇒ 逐条必须报出问题；干净的必须零问题。"""
    subs = []
    clean = audit(obs_good, fixture_mode=False)
    subs.append(('干净观测 → 零问题（实际 %d 条）' % len(clean), not clean))

    cases = [
        ('/history 丢 mode → 报出', _mutate(
            obs_good, 'routes.GET /history.json.mode', 'gone'), 'mode'),
        ('/history 事件少一个键 → 报出键集不符', _mutate_del(
            obs_good, 'routes.GET /history.json.events[0].reason'), '事件键'),
        ('/activity 事件混入 before → 报出键集不符', _mutate(
            obs_good, 'routes.GET /activity.json.events[0].before', None), '事件键'),
        ('事件倒序 → 报出', _mutate(obs_good, 'routes.GET /history.json.events', [
            {'seq': 2, 'kind': 'set', 'actor': 'gui', 'target': 'k', 'value': 1,
             'before': 1, 'after': 1, 'reason': 'r', 'call_id': ''},
            {'seq': 1, 'kind': 'set', 'actor': 'gui', 'target': 'k', 'value': 1,
             'before': None, 'after': 1, 'reason': 'r', 'call_id': ''}]), '升序'),
        ('POST body 改字 → 报出', _mutate(
            obs_good, 'routes.POST /status.text',
            '{"ok": false, "error": "nope"}'), 'POST'),
        ('未知路径 body 改字 → 报出', _mutate(
            obs_good, 'routes.GET /nope.text',
            '{"ok": false, "error": "unknown"}'), '未知路径'),
        ('OPTIONS 丢 ACAM → 报出', _mutate_del(
            obs_good, 'routes.OPTIONS /status.headers.access-control-allow-methods'),
         'OPTIONS ACAM'),
        ('404 丢 CORS → 报出', _mutate(
            obs_good, 'routes.GET /nope.headers', {}), 'Access-Control-Allow-Origin'),
        ('/config 多出契约外键 → 报出', _mutate(
            obs_good, 'routes.GET /config.json.oops', 1), '契约外键'),
        ('row.set 形状错 → 报出', _mutate(
            obs_good, 'routes.GET /config.json.groups.config[0].rows[0].set',
            {'reason': 'r'}), 'row.set'),
        ('has_schema 假但 groups 非空 → 报出', _mutate(
            obs_good, 'routes.GET /config.json.has_schema', False), '凭空造树'),
    ]
    for label, broken, expect in cases:
        probs = audit(broken, fixture_mode=False)
        if not probs:
            subs.append(('%s（实际：探测器没反应 ✗）' % label, False))
        elif expect and not any(expect in p for p in probs):
            subs.append(('%s（实际报的是 %s ✗）' % (label, probs[0][:56]), False))
        else:
            subs.append(('%s（%s）' % (label, probs[0][:64]), True))

    # 夹具相关探测器的致死性：史长/条数不符 + before=null
    fx = json.loads(json.dumps(obs_good))
    fx['fixture'] = {'last_seq': FIXTURE_LAST_SEQ, 'n_events': 1}
    probs = audit(fx, fixture_mode=True)
    subs.append(('夹具探测器：史长/条数/事件序不符即报（%d 条问题）' % len(probs), bool(probs)))
    subs.append(('夹具探测器：`before` 为 null 必须报（fold 全史）',
                 any('fold 全史' in p for p in probs)))

    # 无 schema 分支的致死性
    ns = json.loads(json.dumps(obs_good))
    ns['routes']['GET /config']['json'] = {'ok': True, 'has_schema': False,
                                          'groups': {}, 'orphans': [], 'n_keys': 0}
    nsb = []
    audit_noschema(nsb, ns)
    subs.append(('无 schema 分支：orphans 不对即报（%d 条）' % len(nsb), bool(nsb)))

    # golden diff 的致死性
    gold = {'routes': {'GET /config': {'status': 200, 'shape': {
        'ok': 'bool',
        'groups': {'$list': {'prefix': 'str',
                             'rows': {'$list': {'name': 'str', 'value': 'num'}}}}}}}}
    same = json.loads(json.dumps(gold))
    subs.append(('golden 骨架：自身比对零差异', not diff_skeleton(gold, same, ())))
    renamed = json.loads(json.dumps(gold))
    node = renamed['routes']['GET /config']['shape']['groups']['$list']['rows']['$list']
    node['nameX'] = node.pop('name')
    d1 = diff_skeleton(gold, renamed, ())
    subs.append(('golden 骨架：改键名 → 报出（%s）' % (d1[0][:52] if d1 else '无'), bool(d1)))
    typed = json.loads(json.dumps(gold))
    typed['routes']['GET /config']['shape']['groups']['$list']['rows']['$list']['value'] = 'str'
    subs.append(('golden 骨架：改类型 → 报出', bool(diff_skeleton(gold, typed, ()))))
    added = json.loads(json.dumps(gold))
    added['routes']['GET /config']['shape']['n_snapshot_keys'] = 'num'
    subs.append(('golden 骨架：多出键 → 报出（未豁免时）',
                 bool(diff_skeleton(gold, added, ()))))
    subs.append(('golden 骨架：多出键 + 显式豁免 → 放行（迁移"有意差异"机制）',
                 not diff_skeleton(gold, added, (
                     {'path': "routes['GET /config'].shape.n_snapshot_keys",
                      'why': '框架版新增'},))))
    return subs


# ---------------------------------------------------------------- 主流程
def _last_seq_of(base):
    """活端点史长（只读 /history 反推）；拿不到返回 None。"""
    st, _h, text = http_call('GET', base + '/history?since_seq=0')
    if st != 200:
        return None
    try:
        evs = json.loads(text).get('events') or []
    except ValueError:
        return None
    return max([e.get('seq', 0) for e in evs] or [0])


def _default_exemptions():
    """迁移的**有意差异**（主代理裁决）——逐条带 why，diff 时显式放行。"""
    return [
        {'path': "routes['GET /config'].shape.n_snapshot_keys",
         'why': '框架版 /config 新增可选字段 n_snapshot_keys（主代理裁决：进核心恒回；'
                '面板按键名读，加法安全）'},
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description='监控端点契约门（迁移前置）')
    ap.add_argument('--url', default='', help='拿外部端点比对（跳过夹具断言）')
    ap.add_argument('--record', action='store_true', help='重新生成 golden（改契约时才用）')
    ap.add_argument('--no-selftest', action='store_true', help='跳过致死性自测')
    a = ap.parse_args(argv)

    subs_a, all_problems = [], []
    diffs, n_cmp = [], 0
    gold = None
    obs1 = None

    if a.url:
        base = a.url.rstrip('/')
        print('[cockpit_contract] 外部端点模式：%s（夹具相关断言不适用）' % base)
        obs = observe(base, fixture_mode=False)
        probs = audit(obs, fixture_mode=False)
        all_problems += ['外部端点: %s' % p for p in probs]
        subs_a.append(('外部端点：%d 条契约问题%s'
                       % (len(probs), '（首条：%s）' % probs[0][:60] if probs else ''),
                       not probs))
    else:
        # ---- 夹具一（有 schema）：观测 + POST 只读验证必须**全部做完**，再起夹具二 ----
        br, hist = build_fixture()
        base = 'http://127.0.0.1:%d' % start_monitor_endpoint(br, port=0)
        print('[cockpit_contract] 夹具一（有 schema）%s，史长 %d' % (base, hist.last_seq))
        obs1 = observe(base, fixture_mode=True, last_seq=hist.last_seq)
        probs = audit(obs1, fixture_mode=True)
        all_problems += ['夹具一: %s' % p for p in probs]
        subs_a.append(('夹具一（有 schema）：%d 条契约问题%s'
                       % (len(probs), '（首条：%s）' % probs[0][:60] if probs else ''),
                       not probs))
        # 只读红线：POST 不得产生任何事件（此刻类属性仍指向夹具一，可安全反推史长）
        http_call('POST', base + '/status', {'mode': 'ai'})
        after = _last_seq_of(base)
        subs_a.append(('只读红线：POST 之后史长 %s → %s（须不变）'
                       % (hist.last_seq, after), after == hist.last_seq))

        # ---- 夹具二（无 schema）：后起（见 docstring 的类属性隐患）----
        br2, _h2 = build_minimal_fixture()
        base2 = 'http://127.0.0.1:%d' % start_monitor_endpoint(br2, port=0)
        print('[cockpit_contract] 夹具二（无 schema）%s' % base2)
        obs2 = observe(base2, fixture_mode=False)
        probs2 = audit(obs2, fixture_mode=False)
        nsb = []
        audit_noschema(nsb, obs2)
        all_problems += ['夹具二: %s' % p for p in probs2 + nsb]
        subs_a.append(('夹具二（无 schema 分支）：%d 条契约问题%s'
                       % (len(probs2 + nsb),
                          '（首条：%s）' % (probs2 + nsb)[0][:60] if (probs2 + nsb) else ''),
                       not (probs2 + nsb)))

    C.group('A 契约断言（OPTIONS / 4 路由逐字段 / 404 逐字 / POST 只读 / CORS 全覆盖；'
            '共执行 %d 个断言点）' % _STATS['rules'], subs_a)

    # ---- B golden 骨架 + 逐字段 diff ----
    if a.record and obs1 is not None:
        gold_new = {
            'format': 1,
            'contract': '_mecha-extraction/cockpit-contract.md §8（不可变行为清单）',
            'note': 'skeleton = 键名 + 类型 + 嵌套（不含数值、不含数组长度）；'
                    '夹具条数见 counts；差异豁免见 exemptions',
            'skeleton': skeleton_of(obs1),
            'counts': {'history_events': FIXTURE_LAST_SEQ,
                       'config_n_keys': SCHEMA_N_ROWS,
                       'config_orphans': 1},
            'exemptions': _default_exemptions(),
            'variance_probes': {
                'GET /historyXYZ': {
                    'before_migration': 200, 'after_migration': 404,
                    'why': '路由匹配改「纯路径 + 精确匹配」（原 startswith 把 /historyXYZ 误判为 /history）'},
                'GET /config?x=1': {
                    'before_migration': 404, 'after_migration': 200,
                    'why': '同上：带查询串是合法请求，原先 404 是误判'},
            },
        }
        with open(GOLDEN, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(gold_new, f, ensure_ascii=False, indent=1)
            f.write('\n')
        print('[cockpit_contract] golden 已写入 %s' % GOLDEN)

    if os.path.isfile(GOLDEN):
        with open(GOLDEN, encoding='utf-8') as f:
            gold = json.load(f)
        ex = gold.get('exemptions') or []
        sources = ([('夹具一', obs1)] if obs1 is not None
                   else [('外部端点', observe(a.url.rstrip('/'), fixture_mode=False))])
        for label, obs in sources:
            act = skeleton_of(obs)
            act.pop('probes', None)                     # 探针段单独看，不严判
            n_cmp += 1
            diffs += ['%s: %s' % (label, d) for d in
                      diff_skeleton({'routes': gold['skeleton']['routes']}, act, ex)]
        if obs1 is not None:                            # 夹具条数（强断言，另见 X 组）
            for k, v in (gold.get('counts') or {}).items():
                cur = {'history_events': FIXTURE_LAST_SEQ,
                       'config_n_keys': SCHEMA_N_ROWS,
                       'config_orphans': 1}.get(k)
                if cur != v:
                    diffs.append('counts.%s: golden %r → 夹具 %r' % (k, v, cur))
        C.group('B golden 骨架逐字段 diff（golden %d 路由 × %d 来源；豁免 %d 条）'
                % (len(gold['skeleton']['routes']), n_cmp, len(ex)), [
                    ('零差异（%s）' % (diffs[0][:70] if diffs else '无差异'), not diffs)])
    else:
        C.check('B golden 骨架存在（%s）' % GOLDEN, False,
                '缺 golden：跑 --record 生成并入库（golden 入库才可当迁移放行门）')

    # ---- C 致死性自测 ----
    if a.no_selftest:
        C.skip('C 致死性自测', '--no-selftest')
    else:
        C.group('C 致死性：故意做坏的观测必须被抓（非哑守卫）', selftest(_good_obs()))

    ok = C.summary()
    if all_problems:
        print('[cockpit_contract] 契约问题明细：')
        for p in all_problems[:30]:
            print('   - %s' % p)
    if diffs:
        print('[cockpit_contract] golden 差异明细：')
        for d in diffs[:30]:
            print('   - %s' % d)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
