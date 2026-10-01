# -*- coding: utf-8 -*-
"""只读监控端点判据（两模式收拢后：端点从"人工控制面"降级为"只读监控面"）。

2026-09-24 两模式收拢：人的控制权切换改由**看门人**（launcher.py）编排进程交接，
本端点不再承担夺权/交权/写——它只暴露"当前写权模式 + AI 最近做了什么"给 dsh
右侧栏面板轮询。本判据守两件事：

  A. 装配门控是活的（**与端点无关的库层护栏，仍要守**）：human 模式下 AI 通道写
     被拒（GateDenied + 可教学）、快照不变；human 通道可写。
     ——"AI 不能自授权"靠 Authority 的 side 判定，不靠端点鉴权。
  B. GET /status：回 {ok, mode}，不回任何域键值（只读协调态）。
  C. GET /activity?since_seq=N：返回 seq>=N 的写事件流，actor 归因正确
     （gui 写→actor=gui、ai 写→actor=ai）、seq 单调；since 增量生效。
  D. **只读**：一切 POST（含旧的 /switch_mode /write /write_many）一律 404——
     写与切模式的路由已退役，控制权不经这个端点。

轻量装配：纯内存 V2Bridge（不建 EL cache）——测**端点机制与只读边界**；
"装配点挂没挂 authority"由 authority_selfcheck 与 mecha 仓
phase5_bridge_selfcheck（真 assemble 路径）守，不在此重复起 cache。

**端点实现已归框架**（`mecha.cockpit`，2026-09-25 第 1 段）：本判据只断言**可观测
契约**（路由/形状/归因/只读边界），不碰实现细节；调用的 `src.v2_bind.
start_monitor_endpoint` 现为薄壳，首参仍是 V2Bridge、返回值改成 `MonitorEndpoint`。
"""
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402
from mecha.authority import Authority, Mode                # noqa: E402
from mecha.errors import GateDenied                        # noqa: E402
from mecha.gate import Gate, accept_all                    # noqa: E402
from mecha.history import History                          # noqa: E402

from mecha_v2.bridge import V2Bridge          # noqa: E402
from src.v2_bind import start_monitor_endpoint             # noqa: E402

C = Checker('human_control')


def _get(path):
    try:
        with urllib.request.urlopen(path, timeout=10) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))


def _post(url, obj):
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))


def main():
    tempfile.mkdtemp(prefix='mon_')
    auth = Authority(mode=Mode.HUMAN)
    hist = History()
    br = V2Bridge(Gate(auth, hist, validate=accept_all), hist, auth)

    # ---- A. 装配门控是活的（库层护栏，与端点退役无关，仍守） ----
    try:
        br.write('field.B_gauss', 5.0, reason='AI 越权（应被拒）', actor='ai')
        ai_denied = False
    except GateDenied as e:
        ai_denied = e
    C.group('A 门控是活的：human 模式下 AI 通道写被拒（可教学、不静默）', [
        ('AI 写被拒（不是静默通过）', ai_denied is not False),
        ('拒绝可教学（kind=authority_mode_mismatch + hint）',
         getattr(ai_denied, 'kind', '') == 'authority_mode_mismatch'
         and bool(getattr(ai_denied, 'hint', ''))),
        ('被拒后快照未变', 'field.B_gauss' not in br.effective()),
        ('human 通道写放行（对照：门不是全拒的哑门）',
         br.write('field.B_gauss', 5.0, reason='人写', actor='gui')['ok'] is True),
    ])

    # 监控端点：实现 = `mecha.cockpit`（框架，2026-09-25 第 1 段迁移）；
    # 返回的是 `MonitorEndpoint`（可 .port / .url / .stop()），**不是裸端口 int**。
    endpoint = start_monitor_endpoint(br, port=0)
    base = 'http://127.0.0.1:%d' % endpoint.port

    # ---- B. GET /status：只回协调态 ----
    st, body = _get(base + '/status')
    C.group('B GET /status：回 mode、不回域键值', [
        ('200 且 ok', st == 200 and body.get('ok') is True),
        ('带 mode（当前 human）', body.get('mode') == 'human'),
        ('不回任何域数据（无 config/snapshot/effective 键）',
         not ({'config', 'snapshot', 'effective'} & set(body))),
    ])

    # ---- C. GET /activity?since_seq：增量事件流 + actor 归因 ----
    # 先制造两条写：一条 human（A 里写过 field.B_gauss），再来一条 ai 模式下的写
    br.switch_mode(Mode.AI, actor='human')          # 人开闸（看门人语义）
    br.write('dynamic.npts', 30.0, reason='AI 改点数', actor='ai')
    st, body = _get(base + '/activity?since_seq=0')
    evs = body.get('events', [])
    C.group('C GET /activity?since_seq=0：全量事件流 + 归因正确', [
        ('200 且带 events 列表', st == 200 and isinstance(evs, list) and evs),
        ('human 侧写事件归因到细标签 actor=gui（通道钉死，不被端点抹平）',
         any(e['actor'] == 'gui' and e['target'] == 'field.B_gauss' for e in evs)),
        ('ai 写事件 actor=ai（归因不被端点抹平）',
         any(e['actor'] == 'ai' and e['target'] == 'dynamic.npts' for e in evs)),
        ('事件带 seq/call_id/target/value（面板渲染所需）',
         all({'seq', 'target', 'value', 'actor'} <= set(e) for e in evs)),
        ('seq 单调不减', [e['seq'] for e in evs] == sorted(e['seq'] for e in evs)),
    ])
    # since 增量：取最后一条 seq，再拉 since=它 ⇒ 只应剩它自己或更少
    last_seq = evs[-1]['seq']
    st, body2 = _get(base + '/activity?since_seq=%d' % last_seq)
    evs2 = body2.get('events', [])
    C.check('C since_seq 增量生效（只回 seq>=N，不回更早）',
            st == 200 and evs2 and all(e['seq'] >= last_seq for e in evs2)
            and len(evs2) < len(evs),
            'N=%d 全量=%d 增量=%d' % (last_seq, len(evs), len(evs2)))

    # ---- D. 只读边界：一切 POST 退役（404） ----
    C.group('D 只读端点：写/切模式 POST 一律退役（控制权归看门人）', [
        ('POST /switch_mode → 404（切模式不经端点）',
         _post(base + '/switch_mode', {'mode': 'human'})[0] == 404),
        ('POST /write → 404（写不经端点）',
         _post(base + '/write', {'key': 'x', 'value': 1})[0] == 404),
        ('POST /write_many → 404',
         _post(base + '/write_many', {'patch': {}})[0] == 404),
        ('POST 后快照未被动（真拒，不是回 404 却偷偷写）',
         br.effective().get('x') is None),
    ])

    # ---- E. v1 驾驶舱形状投影（用户裁决：呈现整回来，数据源换活桥） ----
    # 再写一次 field.B_gauss（制造 before→after 链：5.0 → 7.5）
    br.switch_mode(Mode.HUMAN, actor='human')
    br.write('field.B_gauss', 7.5, reason='人再调磁场', actor='gui')
    st, hb = _get(base + '/history?since_seq=0')
    hevs = hb.get('events', [])
    bg = [e for e in hevs if e['target'] == 'field.B_gauss']
    C.group('E GET /history：v1 飞行记录仪形状（before→after 史重建）', [
        ('200 且带 events/mode', st == 200 and hevs and 'mode' in hb),
        ('事件带 v1 全字段（seq/kind/actor/target/before/after/reason）',
         all({'seq', 'kind', 'actor', 'target', 'before', 'after', 'reason'}
             <= set(e) for e in hevs)),
        ('同键二次写：后一条 before=前一条 after（史重建对账）',
         len(bg) == 2 and bg[0]['before'] is None and bg[1]['before'] == 5.0
         and bg[1]['after'] == 7.5),
        ('since_seq 增量生效',
         _get(base + '/history?since_seq=%d' % hevs[-1]['seq'])[1]['events']
         == hevs[-1:]),
    ])
    st, cb = _get(base + '/config')
    C.group('E GET /config：v1 配置态树形状（纯内存桥无 schema → 如实降级）', [
        ('200 且带 groups/orphans/has_schema 键',
         st == 200 and {'groups', 'orphans', 'has_schema'} <= set(cb)),
        ('无引擎桥如实 has_schema=False（不凭空造树）', cb.get('has_schema') is False),
        ('快照键全进 orphans（契约外如实投影，含现值）',
         any(o['name'] == 'field.B_gauss' and o['value'] == 7.5
             for o in cb.get('orphans', []))),
    ])

    # ---- F. `/config` 的**真 schema** 分支 ----
    # 原 `cockpit_contract_selfcheck.py`（迁移前置夹具，1074 行 + golden 354 行）已删：
    # 端点的**传输契约**（四路由字段 / CORS / OPTIONS / POST 404 / 精确匹配 / 无 schema 分支）
    # 归框架 `mecha.cockpit`，由框架 `checks/phase11_cockpit_selfcheck.py` 守；
    # EL 侧只剩**这一条**：**EL 自己的 schema 真的接进了树**（`_el_schema_rows` + 引擎族裸名约定）。
    # ⇒ 承诺改写成行为级断言（R13），不是"这东西不存在"。
    from src.core import protocol as P            # noqa: PLC0415

    class _SchemaOnly:
        """只回 schema 的引擎替身（不建 ARC cache，秒级）——口径同 `ai_surface` 的先例。"""
        _cache = None          # V2Bridge 构造期会读 engine._cache（本组不碰 cache）

        def schema(self):
            return {"config": P.config_schema(), "solve": P.solve_schema(),
                    "engine": P.engine_params(), "settable_keys": P.settable_keys()}

    h2, a2 = History(), Authority(mode=Mode.HUMAN)
    br2 = V2Bridge(Gate(a2, h2, validate=accept_all), h2, a2, engine=_SchemaOnly())
    br2.write('field.B_gauss', 12.0, reason='F 组夹具（真 schema 桥）', actor='gui')
    ep2 = start_monitor_endpoint(br2, port=0)
    st, cb2 = _get('http://127.0.0.1:%d/config' % ep2.port)
    groups = cb2.get('groups', {})
    eng_names = [r['name'] for g in groups.get('engine', []) for r in g.get('rows', [])]
    C.group('F GET /config：EL 的真 schema 接进树（族/裸名约定/孤儿边界）', [
        ('200 且 has_schema=True（有引擎 ⇒ 不是降级分支）',
         st == 200 and cb2.get('has_schema') is True),
        ('三族都在（config / solve / engine），n_keys > 0',
         {'config', 'solve', 'engine'} <= set(groups) and cb2.get('n_keys', 0) > 0),
        ('engine 族键是**裸名**（不带 engine. 前缀）——bridge.params 的既有约定',
         bool(eng_names) and not any(n.startswith('engine.') for n in eng_names)),
        ('写了 1 个 schema 内键 ⇒ n_snapshot_keys=1 且不进 orphans',
         cb2.get('n_snapshot_keys') == 1 and cb2.get('orphans') == []),
    ])
    ep2.stop()

    ok = C.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
