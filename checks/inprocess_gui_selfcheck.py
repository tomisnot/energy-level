# -*- coding: utf-8 -*-
"""A4 判据：**窗口复用已有 bridge**（三层架构 §3 裁决 1 / §6 义务 1）+ §4 的"人机同门"纪律。

四条（每条都带"怎么才会红"）：

| # | 判据 | 红证（本判据自带对照，不靠概率） |
|---|---|---|
| ① | ⭐ 同进程开窗**不再 `writer_lease_held`**（窗口复用**注入的**桥） | 同一进程里再 `MainWindow()`（不给桥 ⇒ 它自建）⇒ **预期**抛 `writer_lease_held`；那一条与"注入成功"互为对照 ⇒ 注入路径若退回自建，①立刻红 |
| ② | ⭐ 窗口开着时 **MCP 往返仍通**（真客户端，非"看起来起来了"） | 起子进程用**声明里的 argv**（`--mcp --with-gui`）：端点不通/子进程当场死 ⇒ 红 |
| ③ | ⭐ **GUI 的写与 AI 的写走同一道门、记同一本账**（§4：为可追溯，不是为安全） | `EL_3LAYER_FAULT=1` 时把 `V2Bridge.write` 换成**直改内部快照**（绕过账）⇒ 账里查不到 ⇒ ③红（实测已跑过） |
| ④ | 写权切换**立刻生效**（开窗形态下再验一次） | 切 human 后 AI 的写若被放行（或切 ai 后人侧仍能写）⇒ 红 |

跑：`python checks\\inprocess_gui_selfcheck.py`（约 30-60s：真起子进程 + 真 MCP 往返）
红证：`set EL_3LAYER_FAULT=1 && python checks\\inprocess_gui_selfcheck.py`（③ 应红、其余仍绿）
"""
import asyncio
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')          # 必须在 import PyQt5 之前

from checks.checklib import Checker                            # noqa: E402

C = Checker('inprocess_gui')
ROOT = pathlib.Path(__file__).resolve().parent.parent
FAULT = os.environ.get('EL_3LAYER_FAULT') == '1'               # ③ 的红证开关

from mecha.authority import Mode                               # noqa: E402
from mecha.errors import GateDenied, MechaError                # noqa: E402
from mecha_v2.bridge import V2Bridge                           # noqa: E402

# ================================================================ 第一部分：进程内
TMP = pathlib.Path(tempfile.mkdtemp(prefix='el_a4_'))
os.environ['RE0_V2_DATA_ROOT'] = str(TMP)                      # 隔离：不碰仓内 journal
bridge = V2Bridge.for_shared(el_root=ROOT, data_root=TMP, mode=Mode.HUMAN)

from PyQt5.QtWidgets import QApplication                        # noqa: E402
from app_entry import MainWindow                               # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)

_win, _win_err, _selfbuilt_err = None, None, None
try:
    _win = MainWindow(bridge=bridge)                           # ⭐ 注入：复用同一份桥
except BaseException as e:                                     # noqa: BLE001
    _win_err = '%s: %s' % (type(e).__name__, e)
try:
    MainWindow()                                               # 对照：自建 ⇒ 抢租约
except MechaError as e:
    _selfbuilt_err = 'MechaError kind=%s' % getattr(e, 'kind', '?')
except BaseException as e:                                     # noqa: BLE001
    _selfbuilt_err = '%s: %s' % (type(e).__name__, e)

C.group('① 同进程开窗复用**注入的**桥；不给桥时自建的那条路被写租约拦（互为对照）', [
    ('注入桥开窗成功（%s）' % (_win_err or 'ok'), _win is not None),
    ('窗口拿到的**就是**那份桥（同一对象，不是第二份）',
     _win is not None and _win.app_bridge is bridge),
    ('对照：自建桥那条路仍被拦（%s）' % _selfbuilt_err,
     str(_selfbuilt_err).startswith('MechaError kind=writer_lease_held')),
])

# ---- ③ "人机同门"：GUI 的写要过门、要进账（带它的 actor）
_gate_ok, _actor_ok, _snap_ok = False, False, False
try:
    if FAULT:                                                  # 红证：只绕过"记账"这一步
        def _bypass(self, key, value, **kw):                   # noqa: ANN001
            # 写权仍照查（否则④也会红，红证就不干净了）——只把"落账"换成直改快照。
            self._gate.check(self._channel_for(kw.get('actor', 'gui')))   # noqa: SLF001
            self._gate._snapshot[key] = value                  # noqa: SLF001  ← 绕过 History
            return {'ok': True}
        V2Bridge.write = _bypass                               # type: ignore[method-assign]
    from src.ui import _mecha_bind as MB
    sp = MB.spinbox(_win.binder, 'field.B_gauss', 0.0, 0.0, 10000.0)
    sp.setValue(12.5)                                          # ← 真控件事件 → binder._write
    app.processEvents()
    _snap_ok = bridge.effective().get('field.B_gauss') == 12.5
    evs = [e for e in bridge.history.events()
           if e.actor == 'gui' and e.target == 'field.B_gauss']
    _gate_ok, _actor_ok = bool(evs), bool(evs) and evs[-1].after == 12.5
except BaseException as e:                                     # noqa: BLE001
    print('[a4] ③ 抛了：%s: %s' % (type(e).__name__, e))

C.group('③ §4 纪律：GUI 的写过同一道门、记同一本账、带 actor（%s）'
        % ('红证模式：期望它红' if FAULT else '正常'), [
    ('值确实落进快照（%s）' % _snap_ok, _snap_ok),
    ('⭐ **账里**有这条写（没有 = 长了后门）（%s）' % _gate_ok, _gate_ok),
    ('且带它的 actor=gui 与 after（%s）' % _actor_ok, _actor_ok),
])

# ---- ④ 写权切换立刻生效（开窗形态下）
_h = _a = None
try:
    bridge.switch_mode('human', actor='human')
    try:
        bridge.write('dynamic.npts', 30, reason='A4 越权', actor='ai')
        _h = '竟然放行（缺陷）'
    except GateDenied as e:
        _h = getattr(e, 'kind', '?')
    bridge.switch_mode('ai', actor='human')
    try:
        bridge.write('dynamic.npts', 31, reason='A4 越权', actor='gui')
        _a = '竟然放行（缺陷）'
    except GateDenied as e:
        _a = getattr(e, 'kind', '?')
    bridge.switch_mode('human', actor='human')
except BaseException as e:                                     # noqa: BLE001
    _h = _a = '%s: %s' % (type(e).__name__, e)

C.group('④ 写权切换立刻生效（开窗形态；切完下一拍就用新值）', [
    ('切 human 后 AI 的写当场被拒（%s）' % _h, _h == 'authority_mode_mismatch'),
    ('切 ai 后人侧的写当场被拒（%s）' % _a, _a == 'authority_mode_mismatch'),
])

# ================================================================ 第二部分：真子进程
PF = ROOT / '.mcp-port'
_before = PF.read_text(encoding='utf-8').strip() if PF.exists() else ''
os.environ['MECHA_SITE_ROOTS'] = str(ROOT)
from mecha.launch import load_sites                            # noqa: E402
SVC = next(v for s in load_sites() for v in s.services if v.id == 'EL')

TMP2 = pathlib.Path(tempfile.mkdtemp(prefix='el_a4_svc_'))
env = dict(os.environ)
env.update({str(k): str(v) for k, v in SVC.env.items()})
env['RE0_V2_DATA_ROOT'] = str(TMP2)
env['QT_QPA_PLATFORM'] = 'offscreen'
log = open(os.path.join(TMP2, 'child.log'), 'w', encoding='utf-8', errors='replace')
proc = subprocess.Popen(list(SVC.argv), cwd=str(SVC.cwd), env=env,
                        stdout=log, stderr=subprocess.STDOUT)
port, rtt, is_err = None, None, None
try:
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(0.25)
        if proc.poll() is not None:
            break
        cur = PF.read_text(encoding='utf-8').strip() if PF.exists() else ''
        if cur and cur != _before and cur.isdigit():
            port = int(cur)
            break
    if port:
        async def _rt():
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            async with streamable_http_client('http://127.0.0.1:%d/mcp' % port) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    res = await s.call_tool('read_config', {})
                    return (getattr(res, 'is_error', None)
                            if getattr(res, 'is_error', None) is not None
                            else getattr(res, 'isError', None))
        rtt, is_err = 'ok', asyncio.run(_rt())
except BaseException as e:                                     # noqa: BLE001
    rtt = '%s: %s' % (type(e).__name__, e)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    tail = open(os.path.join(TMP2, 'child.log'), encoding='utf-8',
                errors='replace').read()
    for f in (PF, ROOT / '.monitor-port'):
        try:
            f.unlink()
        except OSError:
            pass

C.group('② 真子进程（声明里的 argv：`--mcp --with-gui`）+ 窗口开着时 MCP 往返', [
    ('子进程没当场死、端口文件**新鲜**（%r → %r）' % (_before, port), port is not None),
    ('日志里**没有** writer_lease_held（桥被复用，没抢第二份）',
     'writer_lease_held' not in tail),
    ('真 MCP 往返：call_tool(read_config) 拿到回执（%s, is_error=%s）' % (rtt, is_err),
     rtt == 'ok' and is_err is False),
])

ok = C.summary()
if not ok:
    print('[a4] 子进程日志尾部：\n%s' % tail[-800:])
try:
    if _win is not None:
        _win.app_bridge  # noqa: B018
        _win.close()
    bridge.close()
    app.quit()
except Exception:                                              # noqa: BLE001
    pass
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(TMP2, ignore_errors=True)
print('[a4] 临时根已清：%s / %s | FAULT=%s' % (not TMP.exists(), not TMP2.exists(), FAULT))
sys.exit(0 if ok else 1)
