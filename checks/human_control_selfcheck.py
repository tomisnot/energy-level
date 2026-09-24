# -*- coding: utf-8 -*-
"""人工控制端点判据（v2）：token 鉴权 HTTP + actor 服务端钉死 + 模式切换。

守"AI 不能自授权"的护栏（总纲 §5.2⑤）：人的夺权/交权/写走人工控制端点，
actor 由**服务端固定为 human**——请求体里塞 actor 改不了归因；错 token 一律 403。

  A. 装配门控是活的：human 模式下 AI 通道写**被拒**（GateDenied + 可教学）、
     快照不变；human 通道可写。
  B. /switch_mode：错 token 拒（403）；正确 token 切 ai/human 成功且回执带新 mode。
  C. /write：回执 {ok,resolved,seq}；史里 actor=human（不是 ai）；请求体塞
     actor='ai' **无效**（服务端固定）；错 token 写被拒（403）。
  D. /write_many：两键同时落地，史里各一条 actor=human。
  E. 端点面退役：v2 无 /undo（History append-only，撤销=反向 set，ADR
     `EL能力面取舍`）；v1 的 create_server actor 显式化检查移入库仓
     （phase5_hub_selfcheck 守 MCP 面）。

轻量装配：纯内存 V2Bridge（不建 EL cache）——本判据测**端点机制与归因钉死**；
"装配点挂没挂 authority"由 authority_selfcheck 与 mecha 仓
phase5_bridge_selfcheck（真 assemble 路径）守，不在此重复起 cache。
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

from examples.energy_level.bridge import V2Bridge          # noqa: E402
from src.v2_bind import start_human_control                # noqa: E402

C = Checker('human_control')


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
    tmp = tempfile.mkdtemp(prefix='human_ctl_')
    auth = Authority(mode=Mode.HUMAN)
    hist = History()
    br = V2Bridge(Gate(auth, hist, validate=accept_all), hist, auth)

    # ---- A. 装配门控是活的（修复 v1 时代"单元被测、装配没被测"的那一环） ----
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

    port, token = start_human_control(br, port=0,
                                      token_file=os.path.join(tmp, '.human-token'))
    url = 'http://127.0.0.1:%d/switch_mode' % port

    # ---- B. /switch_mode：token 鉴权 ----
    st, _ = _post(url, {'token': 'wrong', 'mode': 'ai', 'reason': 'x'})
    C.check('B 错 token 拒（403）', st == 403, 'got %s' % st)
    st, body = _post(url, {'token': token, 'mode': 'ai', 'reason': '判据夺权'})
    C.check('B 正确 token 切 ai 成功（回执带新 mode）',
            st == 200 and body.get('ok') and br.mode == 'ai',
            'st=%s body=%s mode=%s' % (st, body, br.mode))
    st, body = _post(url, {'token': token, 'mode': 'human', 'reason': '判据还权'})
    C.check('B 切回 human 成功', st == 200 and br.mode == 'human', 'st=%s' % st)

    # ---- C. /write：actor 服务端钉死 ----
    wurl = 'http://127.0.0.1:%d/write' % port
    st, wb = _post(wurl, {'token': token, 'key': 'field.E_vpcm', 'value': 12.5,
                          'reason': '人在端点调场强'})
    set_evs = [e for e in br.history.events() if e.key == 'field.E_vpcm']
    C.group('C 人工写端点：回执形状 + 史里 actor=human（含致死性）', [
        ('/write 成功（回执 {ok,resolved,seq}）',
         st == 200 and wb.get('ok') is True and wb.get('resolved') == 12.5
         and wb.get('seq', 0) >= 1),
        ('快照真变了', br.effective().get('field.E_vpcm') == 12.5),
        ('史里 actor 是 human（不是 ai）',
         bool(set_evs) and set_evs[-1].actor == 'human'),
        ('**致死性**：若服务端漏钉 actor（回退默认 ai），本断言必红',
         set_evs[-1].actor != 'ai'),
        ('请求体塞 actor=ai 无效（服务端固定 human，防伪造）',
         _post(wurl, {'token': token, 'key': 'field.E_vpcm', 'value': 1.0,
                      'reason': '试伪造', 'actor': 'ai'})[0] == 200
         and br.history.events()[-1].actor == 'human'),
        ('错 token 写被拒（403）',
         _post(wurl, {'token': 'wrong', 'key': 'field.B_gauss', 'value': 9.9,
                      'reason': 'x'})[0] == 403),
    ])

    # ---- D. /write_many ----
    st, mb = _post('http://127.0.0.1:%d/write_many' % port,
                   {'token': token, 'reason': '人批量写',
                    'patch': {'field.B_gauss': 1.0, 'dynamic.npts': 20}})
    tail = br.history.events()[-2:]
    C.group('D /write_many 原子批量（actor 固定 human）', [
        ('回 200 ok', st == 200 and mb.get('ok') is True),
        ('两键都落地', br.effective().get('field.B_gauss') == 1.0
         and br.effective().get('dynamic.npts') == 20),
        ('史里两条 actor=human',
         [e.actor for e in tail] == ['human', 'human']),
    ])

    # ---- E. 退役面如实检查 ----
    st, _ = _post('http://127.0.0.1:%d/undo' % port,
                  {'token': token, 'reason': '人撤销'})
    C.check('E v2 端点无 /undo（404；撤销=反向 set 记新事件，ADR EL能力面取舍）',
            st == 404, 'got %s' % st)

    ok = C.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
