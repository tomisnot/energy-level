# -*- coding: utf-8 -*-
"""人工控制端点判据（Phase 1 出口）：switch_mode 经 **token 鉴权 HTTP**（非 MCP，AI 不能
自授权）；错 token 拒（403）；正确 token 切换成功且 `authority_switch` 事件落总线（actor=human）。
"""
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checklib import Checker                              # noqa: E402
from src.mecha.authority import Authority                # noqa: E402
from src.mecha.bus import Bus                            # noqa: E402
from src.mecha.session import Session                    # noqa: E402
from src.adapter.serve import start_human_control        # noqa: E402

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
    bus = Bus(root=None)
    auth = Authority(bus=bus, mode='human')
    sess = Session(bus=bus, authority=auth)
    port, token = start_human_control(sess, port=0)
    url = 'http://127.0.0.1:%d/switch_mode' % port

    st, _ = _post(url, {'token': 'wrong', 'mode': 'ai', 'reason': 'x'})
    C.check('错 token 拒（403）', st == 403, 'got %s' % st)

    st, body = _post(url, {'token': token, 'mode': 'ai', 'reason': '判据夺权'})
    C.check('正确 token 切 ai 成功',
            st == 200 and body.get('ok') and sess.authority.mode == 'ai',
            'st=%s body=%s mode=%s' % (st, body, sess.authority.mode))

    evs = [e for e in bus.since(0) if e.get('kind') == 'authority_switch']
    C.check('authority_switch 落总线且 actor=human',
            len(evs) >= 1 and evs[-1].get('actor') == 'human',
            'n=%d' % len(evs))

    st, _ = _post(url, {'token': token, 'mode': 'human', 'reason': '判据还权'})
    C.check('切回 human 成功', st == 200 and sess.authority.mode == 'human',
            'st=%s' % st)

    ok = C.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
