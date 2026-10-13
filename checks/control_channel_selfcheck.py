# -*- coding: utf-8 -*-
"""控制通道的真实契约：写权立即生效、开窗复用同一软件、查询纯净与同账审计。

含 MCP 与生产子进程链路，Qt offscreen。故障注入仅供显式诊断；
不扫描接线源码的函数名，也不把它升级为普遍开发义务。
"""
import asyncio
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from checks.checklib import Checker                            # noqa: E402

C = Checker('control_channel')


def _watchdog(sec: float) -> None:
    """看门狗：到点**强制退出**（判据宁可红也不能挂死——第一次跑就挂死过一次）。"""
    def _kill():
        time.sleep(sec)
        print('[ctl] ⚠ 看门狗超时（%.0fs）⇒ 强制退出，避免判据挂死' % sec, flush=True)
        os._exit(3)
    threading.Thread(target=_kill, daemon=True).start()


_watchdog(360.0)
ROOT = pathlib.Path(__file__).resolve().parent.parent
FAULTS = {x for x in os.environ.get('EL_CONTROL_FAULT', '').split(',') if x}

from mecha.authority import Mode                               # noqa: E402
from mecha.errors import MechaError                            # noqa: E402
from mecha_v2.bridge import V2Bridge                           # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp(prefix='el_ctl_'))
os.environ['RE0_V2_DATA_ROOT'] = str(TMP)
bridge = V2Bridge.for_shared(el_root=ROOT, data_root=TMP, mode=Mode.HUMAN)
if 'lazy-mode' in FAULTS:                                      # 红证②：切换不生效
    bridge.switch_mode = lambda *a, **k: None                  # type: ignore[method-assign]
if 'no-ledger' in FAULTS:                                      # 红证⑤：不记账
    bridge.record_audit = lambda *a, **k: None                 # type: ignore[method-assign]

from PyQt5.QtWidgets import QApplication, QWidget              # noqa: E402
from app_entry import open_manual_surface_now                  # noqa: E402
from mecha.software_control import start_control_endpoint      # noqa: E402
from mecha_v2.control import CONTROL_SERVER_NAME, wire_control  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
if 'leak' in FAULTS:
    # 红证⑦：把控制操作**也注册进能力面**（AI 那条端点就该看见它们 ⇒ ⑦ 应红）。
    # ⚠ 第一版这里引用了不存在的 `bridge._ability_registry` ⇒ 静默跳过、红证根本没生效
    #   （"守卫不生效"的典型）。现在**自己建**能力面注册表，真注册进去。
    from mecha.gate import Channel as _Ch
    from mecha.tools import define_tool as _define
    from mecha_v2.bridge import _build_gui_tools as _build
    from mecha_v2.hub import McpHost as _Host
    _abl = _build(bridge._engine, bridge._gate, bridge._authority, _Ch('ai'),  # noqa: SLF001
                  bridge.jobs, surface=bridge._surface)
    # 红证：把**改**写权那两个漏进能力面 ⇒ ⑦ 必须红（读本来就该在能力面）
    for _n in ('set_write_authority', 'open_manual_surface'):
        _abl.register(_define(name=_n, description='（红证：故意泄漏到能力面）',
                              parameters={},
                              output_schema={'type': 'object', 'required': ['ok']},
                              execute=lambda: {'ok': True}))
    ability_host = _Host(_abl, port_file=str(TMP / '.mcp-port'))
    ability_port = ability_host.start()
else:
    ability_host, ability_port = bridge.serve_mcp(port_file=str(TMP / '.mcp-port'))

import app_entry as _app_entry                                # noqa: E402

if 'selfbuild' in FAULTS:                                      # 红证③：自建桥（抢租约）
    def _selfbuild(_bridge):
        from app_entry import MainWindow
        w = MainWindow()                                       # ← 不给桥 ⇒ 自建
        w.showMaximized()
        return 'opened(self-build)'
    _app_entry.open_manual_surface_now = _selfbuild
elif 'no-idempotent' in FAULTS:                                # 红证④：每次都新建窗
    def _no_idem(_bridge):
        from app_entry import MainWindow
        w = MainWindow(bridge=_bridge)
        w.showMaximized()
        return 'opened(again)'
    _app_entry.open_manual_surface_now = _no_idem

# ⭐ 用**生产**那条路径：控制请求（MCP 线程）→ 队列 → Qt 主线程建窗。
_app_entry._MAIN_LOOP['running'] = True
_opener = lambda: _app_entry._open_on_qt_thread(bridge)         # noqa: E731

from app_entry import _drain_control_queue                      # noqa: E402

ctl_reg = wire_control(bridge, open_surface=_opener)
ctl_ep, ctl_port = start_control_endpoint(ctl_reg, server_name=CONTROL_SERVER_NAME,
                                          port_file=str(TMP / '.control-port'))


def _mcp_raw(url, tool=None, args=None, want_tools=False):
    """真 MCP 客户端：list_tools 或 call_tool。→ (工具名列表 | 回执, is_error, 正文)"""
    async def _run():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        async with streamable_http_client(url) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                if want_tools:
                    tl = await s.list_tools()
                    return [t.name for t in tl.tools], False, ''
                res = await s.call_tool(tool, args or {})
                err = getattr(res, 'is_error', None)
                if err is None:
                    err = getattr(res, 'isError', None)
                try:
                    txt = res.content[0].text or ''
                except Exception:                              # noqa: BLE001
                    txt = ''
                return None, bool(err), txt
    return asyncio.run(_run())


def mcp(url, tool=None, args=None, want_tools=False, timeout=90.0):
    """在工作线程里跑 MCP 调用，**主线程同时泵 Qt 事件 + 控制队列**。

    ⭐ 这不是测试脚手架：**它就是生产路径** —— 控制请求到达的是 MCP 服务器线程，
    而开窗必须在 Qt 主线程做（`app_entry._drain_control_queue` + `_open_on_qt_thread`）。
    ⚠ 若在服务器线程上直接建 Qt 窗口（第一版判据就是这么写的）⇒ **直接挂死**。
    """
    box: dict = {}

    def _work():
        """⭐ 在线程里跑，**并把异常带出来**。

        ⚠ 为什么必须 try（实测抓到的真缺陷）：原写法是
        ``box.update(raw=_mcp_raw(...))`` —— 一旦 `_mcp_raw` **抛**（例如 MCP 会话的
        `ExceptionGroup`），异常死在**线程里**、`box` 里**没有** `raw` 这个键
        ⇒ 主线程 `box['raw']` 抛 **`KeyError: 'raw'`** ⇒ ⚠ **真因（MCP 调用失败）被一个
        毫无信息量的 KeyError 盖掉**，判据表现成"崩溃"而不是"干净地红"
        （这正是本工程反复在防的「**崩溃冒充失败**」）。
        """
        try:
            box['raw'] = _mcp_raw(url, tool, args, want_tools)
        except BaseException as exc:                       # noqa: BLE001 - 带出去再抛
            box['exc'] = exc

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    t0 = time.time()
    while th.is_alive() and time.time() - t0 < timeout:
        _drain_control_queue(bridge)
        app.processEvents()
        time.sleep(0.01)
    if th.is_alive():
        raise TimeoutError('MCP 调用超时（%s）' % (tool or 'list_tools'))
    if 'exc' in box:
        raise box['exc']                                   # ⭐ 真因，不是 KeyError
    return box['raw']


CTL = 'http://127.0.0.1:%d/mcp' % ctl_port
ABL = 'http://127.0.0.1:%d/mcp' % ability_port

# ---- ① 端点起来 + 回读 + 能调
_pf_txt = (TMP / '.control-port').read_text(encoding='utf-8').strip()
_tools, _, _ = mcp(CTL, want_tools=True)
_r1 = mcp(CTL, 'set_write_authority', {'mode': 'ai'})
C.group('① 控制端点起来、`.control-port` 回读正确、能调 set_write_authority', [
    ('.control-port 内容 = %r，与回读端口一致（%d）' % (_pf_txt, ctl_port),
     _pf_txt == str(ctl_port)),
    ('控制端点工具表 = %s' % _tools,
     {'set_write_authority', 'open_manual_surface', 'read_write_authority'} <= set(_tools)),
    ('调 set_write_authority(ai) 成功（is_error=%s）' % _r1[1], _r1[1] is False),
    ('切完 mode 真的变了（%s）' % bridge.mode, str(getattr(bridge.mode, 'value', bridge.mode)) == 'ai'),
])

# ---- ② 立刻生效（跨端点：控制面切、能力面写）
mcp(CTL, 'set_write_authority', {'mode': 'human'})
_, e_human, t_human = mcp(ABL, 'set_config', {'key': 'dynamic.npts', 'value': 30,
                                              'reason': '控制通道判据'})
mcp(CTL, 'set_write_authority', {'mode': 'ai'})
_, e_ai, _t = mcp(ABL, 'set_config', {'key': 'dynamic.npts', 'value': 31,
                                      'reason': '控制通道判据'})
C.group('② ⭐ 切完**立刻生效**（同一个控制端点切、能力面立刻受影响）', [
    ('切 human 后 AI 侧写**当场被拒**（is_error=%s kind=%s）'
     % (e_human, 'authority_mode_mismatch' in t_human), e_human is True),
    ('拒因是 authority_mode_mismatch', 'authority_mode_mismatch' in t_human),
    ('切回 ai 后 AI 侧写成功（is_error=%s）' % e_ai, e_ai is False),
])

# ---- ③ 开窗（本进程内、复用桥）
_, e_open, t_open = mcp(CTL, 'open_manual_surface', {})
from app_entry import _MANUAL                                 # noqa: E402
_win = _MANUAL['window']
C.group('③ ⭐ open_manual_surface 在**本进程内**开窗、**复用同一份桥**', [
    ('调用成功（is_error=%s 正文=%s）' % (e_open, t_open[:60]), e_open is False),
    ('窗口真的开出来了（%s）' % type(_win).__name__, _win is not None),
    ('⭐ 窗口拿到的**就是**那份桥（同一对象）', _win is not None and _win.app_bridge is bridge),
    ('正文说 status=opened（%s）' % t_open[:60], '"opened"' in t_open.replace(' ', '')),
])

# ---- ④ 幂等
_, e_again, t_again = mcp(CTL, 'open_manual_surface', {})
C.group('④ ⭐ 幂等：再调一次不报错、不开第二个窗', [
    ('第二次调用成功（is_error=%s）' % e_again, e_again is False),
    ('返回 already-open（%s）' % t_again[:60], 'already-open' in t_again),
    ('窗口仍是同一个对象（没换）', _MANUAL['window'] is _win),
])

# ---- ⑤ 两个操作都在账里
_evs = list(bridge.history.events())
# ⚠ 审计键落在 **`Event.op`** 上（不是 `target`：`Gate.record(op, target, …)`，
# `record_audit` 把键当 op 传、target 传空串）——实测确认，别照观感写。
_auth_ev = [e for e in _evs if getattr(e, 'op', None) == 'control.write_authority']
_open_ev = [e for e in _evs if getattr(e, 'op', None) == 'control.open_manual_surface']
C.group('⑤ ⭐ 两个操作**都在账里**（同门同账；带 actor=human）', [
    ('账里有 control.write_authority（%d 条）' % len(_auth_ev), bool(_auth_ev)),
    ('账里有 control.open_manual_surface（%d 条）' % len(_open_ev), bool(_open_ev)),
    ('两条都带 actor=human（%s）'
     % {getattr(e, 'actor', None) for e in _auth_ev + _open_ev},
     all(getattr(e, 'actor', None) == 'human' for e in _auth_ev + _open_ev)),
])

# ---- ⑥ 响亮失败
_, e_bad, t_bad = mcp(CTL, 'set_write_authority', {'mode': 'Mode.AI'})
# 没有手动面 ⇒ 响亮失败：**真起一条 open_surface=None 的控制端点**再调（走完整协议栈）
_noface_ep, _noface_port = start_control_endpoint(
    wire_control(bridge, open_surface=None), server_name=CONTROL_SERVER_NAME,
    port_file=str(TMP / '.control-port-noface'))
_, _e_noface, _t_noface = mcp('http://127.0.0.1:%d/mcp' % _noface_port,
                             'open_manual_surface', {})
C.group('⑥ 响亮失败：未知 mode / 没有手动面', [
    ('未知 mode（"%s" 这种 str(枚举) 形状）被拒（is_error=%s）' % ('Mode.AI', e_bad),
     e_bad is True),
    ('拒因点名 bad_mode（%s）' % t_bad[:70], 'bad_mode' in t_bad),
    ('没有手动面 ⇒ 响亮失败（is_error=%s kind 在正文里=%s）'
     % (_e_noface, 'no_manual_surface' in _t_noface),
     _e_noface is True and 'no_manual_surface' in _t_noface),
])

# ---- ⑦ AI 那条端点看不到控制操作
_abl_tools, _, _ = mcp(ABL, want_tools=True)
C.group('⑦ ⚠ 能力面：**有** read_write_authority（AI 被拒后自查），**没有** set/open', [
    ('能力面工具表 %d 个，⭐**含** read_write_authority' % len(_abl_tools),
     'read_write_authority' in _abl_tools),
    ('⭐ 且**不含** set_write_authority（"不许 AI 给自己解锁"的可判形式）',
     'set_write_authority' not in _abl_tools),
    ('⭐ 且**不含** open_manual_surface', 'open_manual_surface' not in _abl_tools),
    ('能力面仍有它自己的工具（%s…）' % _abl_tools[:3], 'set_config' in _abl_tools),
])

# ---- ⑧ 控制端点上**有** read_write_authority（框架唯一实现），四字段齐
import json as _json                                               # noqa: E402
from mecha_v2.control import HOW_TO_OPEN                           # noqa: E402

_ctl_tools2, _, _ = mcp(CTL, want_tools=True)
_, e_rw, t_rw = mcp(CTL, 'read_write_authority', {})
try:
    _rw = _json.loads(t_rw)
except Exception:                                                  # noqa: BLE001
    _rw = {}
_, _e_rw_abl, _t_rw_abl = mcp(ABL, 'read_write_authority', {})
try:
    _rw_abl = _json.loads(_t_rw_abl)
except Exception:                                                  # noqa: BLE001
    _rw_abl = {}
C.group('⑧ 控制端点上**有** read_write_authority，且四字段齐', [
    ('控制端点工具表含 read_write_authority（%s）' % sorted(_ctl_tools2),
     'read_write_authority' in _ctl_tools2),
    ('调用成功（is_error=%s）' % e_rw, e_rw is False),
    ('四字段齐（%s）' % sorted(_rw), {'mode', 'ai_can_write', 'human_holds',
                                    'how_to_open'} <= set(_rw)),
])

# ---- ⑨ ⭐ 四档逐档对（**不许只断言"返回了 dict"**：恒 True 也能骗过那种断言）
_EXPECT = {'locked': (False, False), 'human': (False, True),
           'ai': (True, False), 'open': (True, True)}
_table = {}
for _m in ('locked', 'human', 'ai', 'open'):
    mcp(CTL, 'set_write_authority', {'mode': _m})
    _, _e2, _t2 = mcp(CTL, 'read_write_authority', {})
    try:
        _table[_m] = _json.loads(_t2)
    except Exception:                                              # noqa: BLE001
        _table[_m] = {}
_rows = []
for _m, (_ai, _hu) in _EXPECT.items():
    _got = _table.get(_m, {})
    _rows.append(('%s ⇒ ai_can_write=%s / human_holds=%s（期望 %s/%s）'
                  % (_m, _got.get('ai_can_write'), _got.get('human_holds'), _ai, _hu),
                  _got.get('ai_can_write') is _ai and _got.get('human_holds') is _hu
                  and _got.get('mode') == _m))
# 红证：把"恒 True"这种错误规则代进去，必须与实测**不符** ⇒ 说明逐档断言有分辨力
_wrong = [{'mode': m, 'ai_can_write': True, 'human_holds': True} for m in _EXPECT]
_discriminating = any((_table.get(r['mode'], {}).get('ai_can_write')
                       != r['ai_can_write']) for r in _wrong)
C.group('⑨ ⭐ 四档逐档对（locked/human/ai/open）', _rows + [
    ('红证：把"恒 True"的错规则代进来 ⇒ 与实测不符（说明逐档断言有分辨力）',
     _discriminating),
])
mcp(CTL, 'set_write_authority', {'mode': 'human'})       # 复原成人侧（本判据的起点）

# ---- ⑩ ⭐ 纯查询：mode 不变、账不增长
_n_before = len(bridge.history.events())
_m_before = str(getattr(bridge.mode, 'value', bridge.mode))
_, e_pure, _t_pure = mcp(CTL, 'read_write_authority', {})
_n_after = len(bridge.history.events())
_m_after = str(getattr(bridge.mode, 'value', bridge.mode))
# 红证（探测灵敏度）：故意落一笔账，探测必须能看见"账长了"
bridge.record_audit('control.__judge_probe', 'x', actor='human', reason='判据红证')
_n_probe = len(bridge.history.events())
C.group('⑩ ⭐ 纯查询无副作用（mode 不变、账不增长）', [
    ('调用成功（is_error=%s）' % e_pure, e_pure is False),
    ('mode 不变（%s → %s）' % (_m_before, _m_after), _m_before == _m_after),
    ('账不增长（%d → %d）' % (_n_before, _n_after), _n_before == _n_after),
    ('红证：探测能看见账增长（%d → %d，故意落那笔）' % (_n_after, _n_probe),
     _n_probe == _n_after + 1),
])

# ---- ⑪ ⭐ how_to_open 是**本项目**给的那句；且这个读操作**只在控制端点**
C.group('⑪ ⭐ how_to_open 由项目给（不是框架缺省）', [
    ('回执里的 how_to_open == 本项目声明的 HOW_TO_OPEN（%s…）' % _rw.get('how_to_open', '')[:40],
     _rw.get('how_to_open') == HOW_TO_OPEN),
    ('它非空、且点到了人侧入口', bool(_rw.get('how_to_open')) and '手动' in str(_rw.get('how_to_open'))),
    ('⭐ **两个面透出同一句**：控制面 == 能力面 == HOW_TO_OPEN（能力面那句 %s…）'
     % str(_rw_abl.get('how_to_open', ''))[:36],
     _rw_abl.get('how_to_open') == HOW_TO_OPEN == _rw.get('how_to_open')),
])

# ---- ③b 真部署形态（子进程 + 生产那条搬运路径）
# ⚠ 子进程的输出**必须落文件**、不能用 `stdout=PIPE`：子进程很啰嗦（uvicorn + Qt 字体告警），
# 而判据这时正阻塞在 MCP 调用上、没人读管道 ⇒ 管道缓冲写满 ⇒ **子进程卡在 write**
# ⇒ 请求永远不回 ⇒ 判据 90s 超时（实测踩过：那看起来像"开窗挂死"，其实是管道死锁）。
_svc_log = TMP / 'svc.log'
_logf = open(_svc_log, 'w', encoding='utf-8', errors='replace')
_svc_pf = ROOT / '.control-port'
_svc_before = _svc_pf.read_text(encoding='utf-8').strip() if _svc_pf.exists() else ''
sub = subprocess.Popen([sys.executable, 'app_entry.py', '--mcp'], cwd=str(ROOT),
                       env={**os.environ, 'RE0_V2_DATA_ROOT': str(TMP / 'svc'),
                            'QT_QPA_PLATFORM': 'offscreen'},
                       stdout=_logf, stderr=subprocess.STDOUT, text=True,
                       encoding='utf-8', errors='replace')
_svc_open, _svc_alive, _sub_tail = False, False, ''
try:
    _port_svc, _deadline = None, time.time() + 120
    while time.time() < _deadline and sub.poll() is None:
        time.sleep(0.25)
        txt = _svc_pf.read_text(encoding='utf-8').strip() if _svc_pf.exists() else ''
        if txt.isdigit() and txt != _svc_before:
            _port_svc = int(txt)
            break
    if _port_svc:
        _svc_tools, _, _ = mcp('http://127.0.0.1:%d/mcp' % _port_svc, want_tools=True)
        _, _e, _t = mcp('http://127.0.0.1:%d/mcp' % _port_svc, 'open_manual_surface', {})
        _svc_open = (_e is False and 'opened' in _t)
        _svc_alive = sub.poll() is None
finally:
    sub.terminate()
    try:
        sub.wait(timeout=20)
    except subprocess.TimeoutExpired:
        sub.kill()
        sub.wait(timeout=10)
    _logf.close()
    try:
        _sub_tail = _svc_log.read_text(encoding='utf-8', errors='replace')[-400:]
    except OSError:
        _sub_tail = ''
    for f in (_svc_pf, ROOT / '.mcp-port', ROOT / '.monitor-port'):
        try:
            f.unlink()
        except OSError:
            pass
C.group('③b ⭐ 真部署形态：子进程 `--mcp` + 生产那条"搬回 Qt 主线程"的路径', [
    ('子进程的控制端点起来了（端口文件 %r）' % _port_svc, _port_svc is not None),
    ('真 MCP 调用 open_manual_surface ⇒ opened（%s）', _svc_open),
    ('开完窗子进程仍活着（Qt 事件在主线程被泵着）', _svc_alive),
    ('日志里没有 writer_lease_held（没抢第二份桥）', 'writer_lease_held' not in _sub_tail),
])

ok = C.summary()
if not ok:
    print('[ctl] 子进程日志尾部：\n%s' % _sub_tail)
try:
    _noface_ep.stop()
    ctl_ep.stop()
    ability_host.stop()
    bridge.close()
    app.quit()
except Exception:                                              # noqa: BLE001
    pass
shutil.rmtree(TMP, ignore_errors=True)
print('[ctl] 临时根已清：%s | FAULTS=%s' % (not TMP.exists(), sorted(FAULTS) or '无'))
sys.exit(0 if ok else 1)
