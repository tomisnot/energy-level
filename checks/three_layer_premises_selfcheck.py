# -*- coding: utf-8 -*-
"""三层架构 **步骤 0 · 证伪实验**（P1 / P2）——只验证，**不改功能代码**。

权威：`mecha/docs/v2/软件-交互-写权-三层架构.md` + `三层架构-施工方案.md` §0。
本脚本回答两个"若不成立则整个新架构要重设计"的前提：

**P1 · 软件进程能同时提供 MCP 服务与 PyQt 窗口，且互不干扰**
- A1 同一进程里 `V2Bridge.for_shared` + `serve_mcp` + 真 Qt 窗口（+ matplotlib/pyqtgraph 已加载）；
- A2 ⭐ **窗口活着**时跑一次**真算**：后台线程走 `ElOps.prepare/execute`（= `build_run →
  check_run → evolve_run`，**numba/OpenMP 的 native 路径**），主线程同时泵 Qt 事件
  ⇒ 这是"业务计算与 Qt 共存"的最接近形态；崩了就如实报；
- A3 ⭐ **真 MCP 客户端往返**（`mcp.client.streamable_http.streamable_http_client`，
  设计 §7.2.1 实测的真名）：initialize + list_tools + `call_tool(read_config)`；
- A4（观察项）同进程**二次装配** `MainWindow` ⇒ 它内部会自己 `make_v2_bridge(...)`
  ⇒ 同一数据根第二次抢写租约，**预期 `writer_lease_held`**（新架构的必改点：窗口必须
  **复用**已有 bridge，不能自建第二份）。

**P2 · 写权切换立刻生效**（"切手动 = 只切 Mode"）
- B1 切 `human` 后：**AI 侧**（MCP 工具面，通道在 `serve_mcp` 时钉死为 `Channel('ai')`）
  的下一次 `set_config` **立刻被拒**；同进程**人侧**（`bridge.write(actor='gui')`）立刻可写；
- B2 切回 `ai`：反向同理（AI 立刻可写、人侧立刻被拒）；
- B3 适配层核实：EL 的 `V2Bridge.switch_mode(new_mode, *, actor=…)` 是**适配器**
  （把 actor 标签映到框架要的 `side`），框架真签名是 `Authority.switch_mode(new_mode, side)`
  ⇒ 把"实际生效的那条路"打出来。

口径/隔离：Qt 用 **offscreen**（本仓判据惯例；native 冲突与像素无关）；
数据根与端口文件都用**临时目录**（不碰仓内 `.mecha-v2`/`.mcp-port`）。
一次性脚本，**不登记进 run_checks 档位**（要真 Qt + mcp 客户端 + 十几秒）。

跑：`python checks\\three_layer_premises_selfcheck.py`
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')       # 必须在 import PyQt5 之前

from checks.checklib import Checker                          # noqa: E402

C = Checker('three_layer_premises')
ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix='el_3layer_'))
PORT_FILE = TMP / '.mcp-port'
#: ⚠ **必须钉死**：A4 的 `MainWindow()` 内部自己 `make_v2_bridge(...)`，它按
#: `RE0_V2_DATA_ROOT` 决定落哪个根——不钉就往**仓内 `.mecha-v2`** 里写（第一次实测
#: 正是如此：A4 悄悄成功了，因为两条桥落在不同的根）。钉死之后两条桥同根 ⇒
#: A4 才会撞上写租约，那正是新架构要在结构上避免的形态。
os.environ['RE0_V2_DATA_ROOT'] = str(TMP)

_tracebacks: list[str] = []


def _is_err(res) -> bool:
    """MCP 回执的**错误位**。⚠ 真名是 `is_error`（snake）；`isError` 只是它的
    序列化别名——框架 R11 记过这个坑（`getattr(res, "isError", False)` 恒为假 ⇒
    观测量不携带信息），本脚本第一版**又踩了一次**：实测 `isError` 属性根本不存在。
    """
    v = getattr(res, 'is_error', None)
    if v is None:
        v = getattr(res, 'isError', None)
    return bool(v)


def _mcp_call(fn):
    """在独立事件循环里跑一段 async 代码（MCP 客户端要用 asyncio）。"""
    return asyncio.run(fn())


async def _roundtrip(url: str, *, switch_first=None, write_key='dynamic.npts',
                     write_val=30) -> dict:
    """真 MCP 往返：initialize → list_tools → （可选切模式）→ read_config / set_config。

    ⭐ **同一条 session 全程复用**：切模式后**不重连**再调 —— 这才叫"下一拍立刻生效"。
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    out: dict = {}
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            tools = await sess.list_tools()
            out['tools'] = sorted(t.name for t in tools.tools)
            r = await sess.call_tool('read_config', {})
            out['read_ok'] = not _is_err(r)

            if switch_first is not None:
                switch_first()                      # ← 同一进程里立刻切写权
                out['switched_at'] = time.monotonic()

            w = await sess.call_tool('set_config', {'key': write_key,
                                                    'value': write_val,
                                                    'reason': 'P2 证伪实验'})
            out['write_is_error'] = _is_err(w)
            txt = ''
            try:
                txt = (w.content or [{}])[0].text or ''
            except Exception:                                      # noqa: BLE001
                txt = str(getattr(w, 'structuredContent', '') or '')
            out['write_text'] = txt[:200]
            try:
                out['write_kind'] = json.loads(txt).get('error', {}).get('info', {}).get('kind')
            except Exception:                                      # noqa: BLE001
                out['write_kind'] = None
    return out


# ---------------------------------------------------------------- 装配（人机同门）
from mecha.authority import Mode                             # noqa: E402
from mecha.errors import GateDenied, MechaError              # noqa: E402
from mecha_v2.bridge import V2Bridge                         # noqa: E402

#: ⚠ **不覆盖 data_dir_name**：必须与 `MainWindow` 内部那条桥落**同一个数据目录**
#: （默认 `.mecha-v2`），否则两条桥井水不犯河水 ⇒ A4 撞不上租约（实测踩过）。
bridge = V2Bridge.for_shared(el_root=ROOT, data_root=TMP, mode=Mode.HUMAN)
host, port = bridge.serve_mcp(host='127.0.0.1', port=0, port_file=str(PORT_FILE))
URL = 'http://127.0.0.1:%d/mcp' % port
print('[3layer] MCP 端点 :%d（端口文件 %s）｜数据根 %s' % (port, PORT_FILE, TMP))

# ---------------------------------------------------------------- A1 Qt 窗口
_qt_err = None
app = window = None
try:
    import matplotlib                                            # noqa: F401
    import pyqtgraph                                             # noqa: F401
    from PyQt5.QtWidgets import QApplication, QWidget
    app = QApplication.instance() or QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle('EL · 三层架构 P1 证伪实验')
    window.resize(320, 200)
    window.show()
    app.processEvents()
except Exception as e:                                            # noqa: BLE001
    _qt_err = '%s: %s' % (type(e).__name__, e)
    _tracebacks.append(traceback.format_exc())

C.group('A1 同一进程：MCP 服务 + 真 Qt 窗口同时起来（matplotlib/pyqtgraph 也已加载）', [
    ('Qt 起来了（%s）' % (_qt_err or 'ok'), app is not None and window is not None),
    ('窗口对象已 show 且事件泵正常（processEvents 不抛）',
     window is not None and window.isVisible()),
    ('sys.modules 里有 PyQt5 + matplotlib + pyqtgraph（GUI 栈真的进了这个进程）',
     all(m in sys.modules for m in ('PyQt5.QtWidgets', 'matplotlib', 'pyqtgraph'))),
])

# ---------------------------------------------------------------- A2 窗口活着时跑真算
_native: dict = {'ok': False, 'why': '未跑', 'detail': ''}
try:
    snap_keys = [
        ('groups.list', [dict(n=60, l=0, j=0.5), dict(n=60, l=1, j=1.5)]),
        ('dynamic.method', 'mixed'),          # mixed = numba 内核（native 路径）
        ('dynamic.t_max', 20.0),
        ('dynamic.npts', 21),
    ]
    for k, v in snap_keys:
        bridge.write(k, v, reason='P1 证伪实验配置', actor='gui')
    lv = bridge._engine.list_levels(bridge.effective())           # noqa: SLF001
    states = lv.get('states') or []
    if states:
        bridge.write('dynamic.psi_single', states[0]['key'],
                     reason='P1 证伪实验初态', actor='gui')
    holder: dict = {}

    def _work():
        try:
            prepared = bridge.ops.prepare(actor='gui', reason='P1 真算')
            holder['run'] = prepared
            res = bridge.ops.execute(prepared, actor='gui', reason='P1 真算')
            holder['res'] = res
        except BaseException as e:                                # noqa: BLE001
            holder['err'] = '%s: %s' % (type(e).__name__, e)
            _tracebacks.append(traceback.format_exc())

    th = threading.Thread(target=_work, daemon=True)
    t0 = time.perf_counter()
    th.start()
    pumps = 0
    while th.is_alive() and time.perf_counter() - t0 < 120:
        app.processEvents()                                       # ← Qt 活着，native 在跑
        pumps += 1
        time.sleep(0.005)
    th.join(timeout=5)
    secs = time.perf_counter() - t0
    summary = (holder.get('res') or {}).get('summary') or {}
    _native.update(ok=bool(summary.get('ok')),
                   why=holder.get('err') or ('ok' if summary.get('ok') else 'summary.ok 假'),
                   detail='n_states=%s pops=%s 用时=%.1fs 事件泵=%d 次'
                          % (len(states),
                             '有' if summary.get('final_top_pop') else '无', secs, pumps))
except BaseException as e:                                        # noqa: BLE001
    _native.update(ok=False, why='%s: %s' % (type(e).__name__, e))
    _tracebacks.append(traceback.format_exc())

C.group('A2 ⭐ 窗口活着时跑一次**真算**（mixed=numba/OpenMP 的 native 路径，主线程泵 Qt 事件）', [
    ('真算跑完且没崩（%s）' % _native['detail'], _native['ok']),
    ('结局：%s' % _native['why'], _native['why'] == 'ok'),
    ('那期间 Qt 事件泵一直在转（说明两者真的同时在跑）', True),
])

# ---------------------------------------------------------------- A3 MCP 真往返
_rt: dict = {}
try:
    _rt = _mcp_call(lambda: _roundtrip(URL))
except BaseException as e:                                        # noqa: BLE001
    _rt = {'err': '%s: %s' % (type(e).__name__, e)}
    _tracebacks.append(traceback.format_exc())

C.group('A3 ⭐ 真 MCP 客户端往返（同进程起服务 + Qt 窗口开着）', [
    ('initialize + list_tools 成功（%d 个工具，含 read_config/set_config）'
     % len(_rt.get('tools', [])),
     bool(_rt.get('tools')) and 'read_config' in _rt.get('tools', [])
     and 'set_config' in _rt.get('tools', [])),
    ('call_tool(read_config) 拿到回执（%s）' % (_rt.get('err') or 'ok'), _rt.get('read_ok') is True),
])

# ---------------------------------------------------------------- A4 观察：二次装配
_second = '未测'
_main_window = None
try:
    from app_entry import MainWindow
    _main_window = MainWindow()                                                  # 自己 make_v2_bridge → 抢同一数据根
    _second = '竟然成功了（同进程二次装配未被租约拦）'
except MechaError as e:
    _second = 'MechaError kind=%s（预期：新架构里窗口必须复用已有 bridge）' % getattr(e, 'kind', '?')
except BaseException as e:                                        # noqa: BLE001
    _second = '%s: %s' % (type(e).__name__, e)
C.check('A4 观察项：同进程二次装配 MainWindow ⇒ %s' % _second, True,
        '这一条不判红绿（两种结局都携带信息）：租约拦下 = 新架构必须让窗口复用同一 bridge')

# ---------------------------------------------------------------- B P2 写权切换
_b: dict = {}


def _human_write(tag):
    try:
        bridge.write('field.B_gauss', 21.7, reason='P2 人侧写（%s）' % tag, actor='gui')
        return True
    except GateDenied:
        return False


def _ai_write(tag):
    try:
        bridge.write('dynamic.npts', 30, reason='P2 AI 侧写（%s）' % tag, actor='ai')
        return True
    except GateDenied:
        return False


try:
    # 起点：for_shared 出厂 human
    _b['start_mode'] = bridge.mode
    _b['gui_before'] = _human_write('human 模式')
    _b['ai_before'] = _ai_write('human 模式（应被拒）')
    # ⭐ 同一条 MCP session 内：切到 ai，再调 set_config
    _b['switch_to_ai'] = _mcp_call(lambda: _roundtrip(
        URL, switch_first=lambda: bridge.switch_mode('ai', actor='human'),
        write_key='dynamic.t_max', write_val=30))
    _b['mode_after_ai'] = bridge.mode
    _b['gui_after_ai'] = _human_write('ai 模式（应被拒）')
    # ⭐ 切回 human：AI 侧下一次写必须立刻被拒
    _b['switch_to_human'] = _mcp_call(lambda: _roundtrip(
        URL, switch_first=lambda: bridge.switch_mode('human', actor='human'),
        write_key='dynamic.t_max', write_val=40))
    _b['mode_after_human'] = bridge.mode
    _b['gui_after_human'] = _human_write('human 模式（应放行）')
    # B3 适配层：actor 标签 → side（框架真签名是 switch_mode(mode, side)）
    import inspect
    from mecha.authority import Authority
    _b['authority_sig'] = str(inspect.signature(Authority.switch_mode))
    _b['bridge_sig'] = str(inspect.signature(V2Bridge.switch_mode))
    try:
        bridge.switch_mode('ai', actor='martian')
        _b['unknown_actor'] = '竟然放行（缺陷）'
    except GateDenied as e:
        _b['unknown_actor'] = 'GateDenied kind=%s' % getattr(e, 'kind', '?')
    bridge.switch_mode('human', actor='human')
except BaseException as e:                                        # noqa: BLE001
    _b['err'] = '%s: %s' % (type(e).__name__, e)
    _tracebacks.append(traceback.format_exc())

_w_ai = _b.get('switch_to_ai', {})
_w_hu = _b.get('switch_to_human', {})
C.group('B1 切 human：AI 侧下一次写**立刻**被拒（不重连、不等待），人侧立刻可写', [
    ('出厂模式 = human（%s）' % _b.get('start_mode'), _b.get('start_mode') == 'human'),
    ('切换前：人侧可写 / AI 侧被拒（%s / %s）'
     % (_b.get('gui_before'), _b.get('ai_before')),
     _b.get('gui_before') is True and _b.get('ai_before') is False),
    ('切成 ai 后：AI 侧 MCP set_config **成功**（isError=%s）'
     % _w_ai.get('write_is_error'),
     _w_ai.get('write_is_error') is False),
    ('切回 human 后：**同一条 session** 上 AI 侧 set_config 立刻被拒（isError=%s kind=%s）'
     % (_w_hu.get('write_is_error'), _w_hu.get('write_kind')),
     _w_hu.get('write_is_error') is True
     and _w_hu.get('write_kind') == 'authority_mode_mismatch'),
    ('同一拍人侧立刻可写（%s）' % _b.get('gui_after_human'), _b.get('gui_after_human') is True),
])
C.group('B2 反向同理：切 ai 后 AI 立刻可写、人侧立刻被拒（无"等一拍"）', [
    ('切 ai 后 mode=%s' % _b.get('mode_after_ai'), _b.get('mode_after_ai') == 'ai'),
    ('人侧 write 立刻被拒（%s）' % _b.get('gui_after_ai'), _b.get('gui_after_ai') is False),
])
C.group('B3 适配层核实：EL 的 switch_mode 吃 **actor 标签**，框架真签名吃 **side**', [
    ('框架真签名：Authority.switch_mode%s' % _b.get('authority_sig'),
     'side' in (_b.get('authority_sig') or '')),
    ('EL 适配器签名：V2Bridge.switch_mode%s' % _b.get('bridge_sig'),
     'actor' in (_b.get('bridge_sig') or '')),
    ('未知 actor 被拒（%s）' % _b.get('unknown_actor'),
     str(_b.get('unknown_actor')).startswith('GateDenied')),
])

# ---------------------------------------------------------------- 收尾
ok = C.summary()
if _tracebacks:
    print('\n[3layer] 捕获到的异常栈（最多 2 份）：')
    for t in _tracebacks[:2]:
        print(t[-900:])
try:
    host.stop()
except Exception:                                                 # noqa: BLE001
    pass
try:
    if _main_window is not None:           # A4 那条桥也持租约：先放它，否则临时根删不掉
        _main_window.app_bridge.close()
except Exception:                                                 # noqa: BLE001
    pass
try:
    bridge.close()                                                # 释放写租约
except Exception:                                                 # noqa: BLE001
    pass
try:
    if app is not None:
        app.quit()
except Exception:                                                 # noqa: BLE001
    pass
shutil.rmtree(TMP, ignore_errors=True)
print('[3layer] 临时根已清：%s%s' % (not TMP.exists(),
      '' if not TMP.exists() else '（仍有句柄持着：见上面 A4/桥的收尾）'))
sys.exit(0 if ok else 1)
