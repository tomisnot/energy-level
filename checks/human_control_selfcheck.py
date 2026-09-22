# -*- coding: utf-8 -*-
"""人工控制端点判据（Phase 1 出口）：switch_mode 经 **token 鉴权 HTTP**（非 MCP，AI 不能
自授权）；错 token 拒（403）；正确 token 切换成功且 `authority_switch` 事件落总线（actor=human）。

★ **本判据 2026-09-22 的重要修正：改为测"装配"而不只是"机制"。**
原先它**自己造 session**（`Session(bus=bus, authority=auth)`），只验端点处理器；
于是**从没验过启动器 `serve.main` 的装配**——而 main 里当时传的是
`Session(root=root)`（**没有 authority**）⇒ **Hub 进程里"两模式写权"这条不变量根本不存在**，
本判据却 7/0 全绿。**单元被测了，装配没被测。**
现改为调 `serve.build_session(...)`（启动器用的**同一个**装配函数）⇒ 装配点可被测。
"""
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checklib import Checker                              # noqa: E402
from mecha import runtime                            # noqa: E402
from src.adapter import EngineAdapter                # noqa: E402
from src.adapter.serve import build_session, start_human_control   # noqa: E402

# 人工**写**端点走 `session.set` ⇒ 过 registry 校验 ⇒ 需要注入引擎
# （P2-4 之前本判据只测 switch_mode，不碰写路径，故一直没注入过）。
runtime.set_engine(EngineAdapter())

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
    # 数据根**显式**（P0-1：mecha 不再从 __file__ 倒推项目根）。判据是独立进程、
    # cwd 未必是仓根，故用临时目录而非相对路径。
    tmp = tempfile.mkdtemp(prefix='human_ctl_')
    # ★★ 关键：用**启动器的装配函数**（`serve.build_session`），不再自己造 session。
    sess = build_session(tmp, mode='human')
    auth = sess.authority
    bus = sess.bus

    # ---- ★ 装配门控是**活的**（这是修复前真正缺失的那一环） ----
    _ai_denied = None
    try:
        sess.set('field.B_gauss', 5.0, actor='ai', reason='AI 越权（应被拒）')
        _ai_denied = False
    except Exception as e:                                    # noqa: BLE001
        _ai_denied = e
    C.group('★ 装配门控是活的：human 模式下 AI 写**被拒**（可教学、不静默）', [
        ('`build_session` 返回的 Session **挂着 authority**（不是 None）',
         auth is not None),
        ('human 模式下 AI 写入**被拒**（不是静默通过）', _ai_denied is not False),
        ('拒绝可教学（kind=authority_denied + hint）',
         getattr(_ai_denied, 'kind', None) == 'authority_denied'
         and bool(getattr(_ai_denied, 'hint', None))),
        ('被拒后快照**未变**（没写进去）', 'field.B_gauss' not in sess.effective()),
        ('**致死性**：若装配漏了 authority（= 修复前 `main` 的状态），'
         'AI 写会**静默通过** ⇒ 上面那条必红', auth is not None),
    ])

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

    # ---------------- P2-4：人工**写**端点（人的面，actor 由服务端固定 human） ----------------
    wurl = 'http://127.0.0.1:%d/write' % port
    st, wb = _post(wurl, {'token': token, 'key': 'field.B_gauss', 'value': 12.5,
                          'reason': '人在 GUI 调磁场'})
    set_ev = [e for e in bus.since(0) if e.get('kind') == 'set']
    C.group('P2-4 人工写端点：human 模式可写 + journal 记 actor=human（含致死性）', [
        ('human 模式下经 /write 写入成功（回执与 set 同形）',
         st == 200 and wb.get('ok') is True and wb.get('resolved') == 12.5),
        ('快照真变了', sess.effective().get('field.B_gauss') == 12.5),
        ('**journal 里 actor 是 `human`（不是 `ai`）**',
         bool(set_ev) and set_ev[-1].get('actor') == 'human'),
        ('**致死性**：若 actor 被写成 `ai`（旧默认值那个 bug），本断言必红',
         set_ev[-1].get('actor') != 'ai'),
        ('请求体里塞 actor 无效（服务端固定 human，防伪造 M1）',
         _post(wurl, {'token': token, 'key': 'field.E_vpcm', 'value': 1.0,
                      'reason': '试伪造', 'actor': 'ai'})[0] == 200
         and [e for e in bus.since(0) if e.get('kind') == 'set'][-1]
         .get('actor') == 'human'),
        ('错 token 写被拒（403）',
         _post(wurl, {'token': 'wrong', 'key': 'field.B_gauss', 'value': 9.9,
                      'reason': 'x'})[0] == 403),
    ])

    # /write_many 原子批量 + /undo
    st, mb = _post('http://127.0.0.1:%d/write_many' % port,
                   {'token': token, 'reason': '人批量写',
                    'patch': {'field.B_gauss': 1.0, 'field.E_vpcm': 2.0}})
    # ⚠ 在 undo **之前**取快照（undo 会撤掉最后一次写 ⇒ 之后再查就看不到两键同时落地）
    _snap_after_many = sess.effective()
    st_u, ub = _post('http://127.0.0.1:%d/undo' % port,
                     {'token': token, 'reason': '人撤销'})
    C.group('P2-4 /write_many 原子批量 + /undo（actor 均固定 human）', [
        ('write_many 回 2 条 set 回执',
         st == 200 and len((mb or {}).get('recs') or []) == 2),
        ('两键**在 undo 前**都落地',
         _snap_after_many.get('field.B_gauss') == 1.0
         and _snap_after_many.get('field.E_vpcm') == 2.0),
        ('undo 成功且 actor=human',
         st_u == 200 and ub.get('ok') is True),
        ('undo 事件 actor=human',
         [e for e in bus.since(0) if e.get('kind') == 'undo'][-1]
         .get('actor') == 'human'),
    ])

    # ---------------- P0-5：create_server 的 actor 必须显式（不能有 'ai' 默认） ----------------
    from mecha.server import create_server as _cs
    from mecha.ops import Ops as _Ops
    try:
        _cs(sess, _Ops(sess))
        _no_actor_raises = False
    except Exception as e:                                    # noqa: BLE001
        _no_actor_raises = 'actor' in str(e)
    C.group('P0-5/P2-4 create_server 的 actor 必须显式给（防静默冒充 AI）', [
        ('**致死性**：零参 create_server → 明确报错（不再默认 ai）', _no_actor_raises),
        ('给了 actor 就能起（MCP 面给 ai）',
         _cs(sess, _Ops(sess), actor='ai')[0] is not None),
    ])

    ok = C.summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
