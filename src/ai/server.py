# -*- coding: utf-8 -*-
"""进程内 MCP server：把 Energy Level 暴露为 AI harness 的 skill。

运行模式（G3）：软件启动时 start_server() 拉起 streamable HTTP 端点
（127.0.0.1，端口写项目根 .mcp-port 供 harness 配置发现）；harness 随后
启动、连 URL 即用。人不动 AI 动、AI 不动人动，两边经 SimState 同一状态。

线程模型：uvicorn 在专用线程跑 asyncio 循环；工具 handler 内所有 Qt 状态
读写经 dispatch.run_on_main 投递主线程执行（asyncio.to_thread 包裹避免
卡住事件循环）。

v1 策略：AI 写操作先斩后奏（草案 §开放问题，用户拍板）——直接生效，
归因与记账由参数注册表的 op 事件统一驱动（src/ai/params.py +
param_framework）：人经 touch→commit 自动 op(source='user')，AI 经
reg.set/mutate(source='ai', reason, 变更前快照)——本模块只把 op 桥接到
journal，undo / get_activity 基于 journal 兜底可追溯。
"""
import asyncio
import json
import os
import socket
import threading
import time
from typing import Any

from .dispatch import run_on_main
from .journal import get_journal
from .sim_state import SimState, json_safe

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PORT_FILE = os.path.join(_ROOT, '.mcp-port')
_SHOT_DIR = os.path.join(_ROOT, 'ui_design', 'screenshots', '_ai')

_server_thread = None
_server = None            # uvicorn.Server
_mcp = None               # MCPServer 实例
_state = None             # SimState 门面
_jobs = {}                # job_id -> {'started': ts, 'seq': int}
_job_seq = [0]


# ============================================================
# 注册表 op → journal 桥（人机统一落账）
# ============================================================
def _brief(v):
    """op 值 → journal 友好（大列表只记长度/摘要，防流水爆量）。"""
    if isinstance(v, list):
        return {'n': len(v), 'head': v[:2]}
    if isinstance(v, dict):
        return {k: _brief(x) for k, x in list(v.items())[:8]}
    return v


def _journal_bridge(op):
    """param_framework op 事件 → journal（source 即 actor）。"""
    j = get_journal(_ROOT)
    detail = {'path': op['full_name'], 'old': _brief(op['old']),
              'new': _brief(op['new'])}
    if not op['ok']:
        detail['error'] = op['error']
    j.record(op['source'], 'set_param', detail, op['reason'],
             op.get('snapshot'))


# ============================================================
# 工具实现（全部在主线程执行；返回 JSON 友好 dict）
# ============================================================
def _win():
    from PyQt5.QtWidgets import QApplication
    for w in QApplication.topLevelWidgets():
        if type(w).__name__ == 'MainWindow' and w.isVisible():
            return w
    return None


def _t_get_state(scope):
    return _state.get_state(scope)


def _t_summary():
    return _state.summary()


def _t_set_param(path, value, reason):
    # journal 由 reg.set 的 op 事件自动落账（含快照/归因/高亮）
    return _state.set_param(path, value, reason=reason)


def _t_add_group(n, l, j, reason):
    return _state.add_level_group(n, l, j, reason=reason)


def _t_remove_group(idx, reason):
    return _state.remove_level_group(idx, reason=reason)


def _t_set_waves(atom, waves, reason):
    return _state.set_waves(waves, atom, reason=reason)


def _t_activity(since_seq):
    return {'ops': get_journal(_ROOT).since(since_seq)}


def _t_undo(reason):
    j = get_journal(_ROOT)
    pub, snap = j.undo_last('ai')
    if snap is None:
        return {'ok': False, 'msg': '没有可撤销的 AI 操作'}
    _state.restore_snapshot(snap)
    j.record('ai', 'undo', {'undid': pub['seq'], 'kind': pub['kind'],
                            'path': pub['detail'].get('path')},
             reason, None)
    return {'ok': True, 'msg': f"已撤销 op#{pub['seq']} "
                               f"{pub['detail'].get('path', pub['kind'])}",
            'undone': pub}


def ui_undo_last_ai():
    """GUI"撤销AI"按钮入口（已在主线程，直接调用）。"""
    try:
        return _t_undo('人点撤销按钮')
    except Exception as e:          # noqa: BLE001
        return {'ok': False, 'msg': f'撤销失败: {e}'}


def _t_screenshot():
    w = _win()
    if w is None:
        return {'ok': False, 'msg': '主窗口不可见'}
    os.makedirs(_SHOT_DIR, exist_ok=True)
    path = os.path.join(_SHOT_DIR, f'shot_{time.strftime("%Y%m%d_%H%M%S")}.png')
    w.grab().save(path)
    return {'ok': True, 'path': path}


def _t_save_archive(path):
    w = _win()
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    snap = _state.to_dict()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    get_journal(_ROOT).record('ai', 'save_archive', {'path': path}, '', None)
    return {'ok': True, 'path': path}


def _t_load_archive(path):
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    with open(path, encoding='utf-8') as f:
        snap = json.load(f)
    before = _state.to_dict()
    # 先斩后奏兜底：破坏性恢复前自动快照到磁盘（undo 失效时的最后防线）
    try:
        ad = os.path.join(_ROOT, 'save_state', '_autosnap')
        os.makedirs(ad, exist_ok=True)
        with open(os.path.join(ad, time.strftime('%Y%m%d_%H%M%S') + '.json'),
                  'w', encoding='utf-8') as f:
            json.dump(before, f, ensure_ascii=False)
        old = sorted(os.listdir(ad))[:-20]      # 保留最近 20 份
        for nm in old:
            try:
                os.remove(os.path.join(ad, nm))
            except OSError:
                pass
    except OSError:
        pass
    _state.restore_snapshot(snap)
    get_journal(_ROOT).record('ai', 'load_archive', {'path': path}, '', before)
    return {'ok': True, 'path': path, 'summary': _state.summary()}


# ---------- 计算通道（P4） ----------
class _QuietModals:
    """上下文：把 QMessageBox 信息/警告框降级为状态栏日志（AI 静默模式）。

    防模态框卡住主线程 → AI 通道死锁；GUI 画布/日志不受影响。
    """

    def __init__(self, win):
        self.win = win
        self._saved = {}

    def __enter__(self):
        from PyQt5.QtWidgets import QMessageBox
        self._saved = {}
        for name in ('information', 'warning', 'critical'):
            fn = getattr(QMessageBox, name)
            self._saved[name] = fn

            def _log(*a, _n=name):
                try:
                    self.win.log(f'[AI 静默] {_n}: {a[2] if len(a) > 2 else a}')
                except Exception:      # noqa: BLE001
                    pass
                return QMessageBox.Ok

            setattr(QMessageBox, name, staticmethod(_log))
        return self

    def __exit__(self, *exc):
        from PyQt5.QtWidgets import QMessageBox
        for name, fn in self._saved.items():
            setattr(QMessageBox, name, fn)
        return False


def _t_run_compute(jid):
    """触发与"人点运行"完全同一条 _run 路径；立即返回 job_id。

    完成回收：_run 内部 `done.connect(self._on_evolve_done)` 在连接时解析
    实例属性 → 调用前先在实例上打补丁即可拦截（无需改 ui_dynamic）。
    """
    w = _win()
    dyn = w.dynamic
    if getattr(dyn, '_run_active', False):
        _jobs.pop(jid, None)
        return {'ok': False, 'msg': '已有演化在运行（人或 AI 发起），先等待或取消'}
    _jobs[jid] = {'started': time.time(), 'done': False, 'error': None,
                  'res': None}

    orig_s, orig_p = dyn._on_evolve_done, dyn._on_pair_evolve_done
    orig_show = dyn._show_progress

    def _restore():
        dyn._on_evolve_done = orig_s
        dyn._on_pair_evolve_done = orig_p
        dyn._show_progress = orig_show

    def _watch(orig):
        def _f(res, error):
            job = _jobs.get(jid)
            if job is not None:
                job['done'] = True
                job['error'] = None if error in (None, 'cancelled') \
                    else str(error)
                job['res'] = res
            _restore()              # 一次性：done 后立即还原，人后续运行不受扰
            with _QuietModals(w):
                orig(res, error)
        return _f

    dyn._on_evolve_done = _watch(orig_s)
    dyn._on_pair_evolve_done = _watch(orig_p)
    dyn._show_progress = lambda: None   # AI 静默：不弹进度框（轮询定时器无害）
    try:
        with _QuietModals(w):
            dyn._run()
    except Exception as e:          # noqa: BLE001
        _restore()
        _jobs.pop(jid, None)
        return {'ok': False, 'msg': f'启动失败: {e}'}
    worker = getattr(dyn, '_worker', None)
    if worker is None or not worker.isRunning():
        # _run 未启动 worker（校验提示/无波形等即时失败路径）
        _restore()
        job = _jobs.pop(jid, None)
        msg = (job or {}).get('error') or '未启动（校验未过或无波形，见 GUI 日志）'
        return {'ok': False, 'msg': msg}
    get_journal(_ROOT).record('ai', 'run_compute', {'job': jid}, '', None)
    return {'ok': True, 'job_id': jid,
            'msg': '已启动（GUI 同步显示进度）；用 job_status 轮询'}


def _t_job_status(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return {'ok': False, 'msg': f'未知 job {job_id}'}
    dyn = _win().dynamic
    pa = getattr(dyn, 'progress_arr', None)
    pct = None
    if not job['done'] and pa is not None:
        stage = int(pa[2]) if len(pa) > 2 else 0
        p0 = min(max(float(pa[0]), 0.0), 1.0)
        pct = 95 + 5 * p0 if stage == 2 else (5 + 90 * p0 if stage == 1 else 5 * p0)
    return {'ok': True, 'job_id': job_id, 'done': job['done'],
            'progress_pct': None if job['done'] else round(pct or 0.0, 1),
            'elapsed_s': round(time.time() - job['started'], 1),
            'error': job['error']}


def _res_summary(res, save_npz=False, tag=''):
    """演化结果 dict → 摘要（top/终态布居 + 可选 npz 落盘）。"""
    import numpy as np
    out = {}
    if res is None:
        return {'msg': '无结果（可能被取消）'}
    t = np.asarray(res.get('t', []))
    pops = res.get('pops')
    pops2 = res.get('pops_2d')
    states = getattr(_win().dynamic, '_run_states', None) or []
    labels = [s.get('label', '?') for s in states]
    summary = {'t_ns': [float(t[0]), float(t[-1])] if len(t) else None,
               'n_steps': int(len(t)), 'states': labels[:40],
               'n_states': len(labels)}
    arr = pops if pops is not None else pops2
    if arr is not None:
        arr = np.asarray(arr)
        avg = arr.mean(axis=0)
        order = np.argsort(-avg)[:8]
        summary['top_states_avg_pop'] = {
            (labels[i] if i < len(labels) else str(i)): round(float(avg[i]), 5)
            for i in order}
        summary['final_pop'] = {
            (labels[i] if i < len(labels) else str(i)): round(float(arr[-1][i]), 5)
            for i in order}
        if save_npz:
            path = os.path.join(_ROOT, 'tempsimu',
                                f'ai_{tag or "res"}_{time.strftime("%H%M%S")}.npz')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, **{k: np.asarray(v)
                                         for k, v in res.items()
                                         if isinstance(v, (np.ndarray, list, tuple))
                                         or hasattr(v, 'dtype')})
            summary['npz'] = path
    return summary


def _t_job_result(job_id, save_npz):
    job = _jobs.get(job_id)
    if job is None:
        return {'ok': False, 'msg': f'未知 job {job_id}'}
    if not job['done']:
        return {'ok': False, 'msg': '计算未完成，先 job_status 轮询'}
    if job['error']:
        return {'ok': False, 'msg': f"失败: {job['error']}"}
    out = {'ok': True, 'job_id': job_id}
    out['summary'] = _res_summary(job['res'], save_npz, tag=job_id)
    return out


def _t_cancel_job(job_id):
    job = _jobs.get(job_id)
    if job is None or job['done']:
        return {'ok': False, 'msg': 'job 不存在或已结束'}
    dyn = _win().dynamic
    run_on_main(dyn._cancel_run)
    return {'ok': True, 'msg': '取消标志已置位'}


# ============================================================
# MCP server 装配
# ============================================================
_INSTRUCTIONS = """Energy Level：Rb-87 里德堡原子仿真软件的 AI 接口。
单位约定：B=Gauss，E=V/cm，光频率=GHz，光强=mW/cm²，微波频率=MHz，
时间轴 t_max=ns，衰变率 Γ=MHz。
两种驱动形态：单步探索用细粒度工具（summary → set_param → run_compute →
job_status/job_result）；重复/组合动作（参数扫描、批量对比、结果后处理）
用 run_python 一次跑完，省回合且大数组不离境。
写操作先斩后奏：直接生效、GUI 实时反映；不满意用 undo（只撤 AI 自己的
最近一步；run_python 整段算一步）。get_activity 可查所有操作流水
（人/AI 混合）。"""


def _register_tools(mcp):
    d = run_on_main   # 简写

    @mcp.tool(description='读当前仿真状态（scope: 空=全部 / groups / field / dynamic）')
    async def get_state(scope: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: _t_get_state(scope or None))

    @mcp.tool(description='轻量摘要：能级组列表、场、方法、波形数（先调这个建立认知）')
    async def summary() -> dict:
        return await asyncio.to_thread(d, _t_summary)

    @mcp.tool(description='改一个参数。path 见 list_params；value 按类型；reason 简述动机')
    async def set_param(path: str, value: Any = None, reason: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: _t_set_param(path, value, reason))

    @mcp.tool(description='列出所有可改参数的 schema（type/unit/desc/range/options/value）')
    async def list_params(prefix: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: {'params': _state.schema(prefix)})

    @mcp.tool(description='添加能级组（n 主量子数, l 轨道角动量, j 总角动量），自动校验+配色')
    async def add_level_group(n: int, l: int, j: float, reason: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: _t_add_group(n, l, j, reason))

    @mcp.tool(description='按索引删除能级组（索引见 get_state groups）')
    async def remove_level_group(index: int, reason: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: _t_remove_group(index, reason))

    @mcp.tool(description='整表替换波形序列。atom: single 单原子 / A·B 双原子'
                         '（三套数据独立）。waves 为 serialize_waves 同构 list：'
                         '普通波 {name,freq_mhz,intensity,pol,offset_ns,env:[gauss,c,s,peak]|[square,start,edge,height]}')
    async def set_waves(waves: list, atom: str = 'A', reason: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: _t_set_waves(atom, waves, reason))

    @mcp.tool(description='启动演化计算（与 GUI 点"运行"同一路径）。立即返回 job_id，'
                         'GUI 进度照常显示；用 job_status/job_result 跟进')
    async def run_compute() -> dict:
        _job_seq[0] += 1
        jid = f'job{_job_seq[0]}'
        return await asyncio.to_thread(d, lambda: _t_run_compute(jid), 300.0)

    @mcp.tool(description='查询计算进度')
    async def job_status(job_id: str) -> dict:
        return await asyncio.to_thread(d, lambda: _t_job_status(job_id))

    @mcp.tool(description='取计算结果摘要（top 布居/终态布居）；save_npz=true 另存全量数组给路径')
    async def job_result(job_id: str, save_npz: bool = False) -> dict:
        return await asyncio.to_thread(d, lambda: _t_job_result(job_id, save_npz))

    @mcp.tool(description='取消运行中的计算')
    async def cancel_job(job_id: str) -> dict:
        return await asyncio.to_thread(d, lambda: _t_cancel_job(job_id))

    @mcp.tool(description='操作流水（journal）：since_seq 之后的全部变更，含人/AI 来源与原因')
    async def get_activity(since_seq: int = 0) -> dict:
        return await asyncio.to_thread(d, lambda: _t_activity(since_seq))

    @mcp.tool(description='撤销最近一个 AI 操作（回滚到其变更前快照）')
    async def undo(reason: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: _t_undo(reason))

    @mcp.tool(description='保存当前状态为 JSON 存档（与菜单"保存状态"同源）')
    async def save_archive(path: str = 'save_state/ai_state.json') -> dict:
        return await asyncio.to_thread(d, lambda: _t_save_archive(path))

    @mcp.tool(description='从 JSON 存档恢复（与菜单"恢复状态"同源，可 undo）')
    async def load_archive(path: str) -> dict:
        return await asyncio.to_thread(d, lambda: _t_load_archive(path))

    @mcp.tool(description='主窗口截图 PNG → 返回文件路径（了解人眼所见）')
    async def screenshot() -> dict:
        return await asyncio.to_thread(d, _t_screenshot)

    @mcp.tool(description='CodeAct：一次调用跑一段 Python，完成循环/扫描/后处理等'
                         '多步编排（省回合）。可用名字：summary/get_state/set_param/'
                         'add_level_group/remove_level_group/set_waves/load_archive/'
                         'save_archive/run_compute_and_wait(启动并等完成,返回摘要)/'
                         'activity/np/json/time/ROOT。把结论赋给变量 result。'
                         '大数组用 print 出统计量、别整体回传。整段算一个 undo 单元。')
    async def run_python(code: str, timeout: float = 1800.0) -> dict:
        from .codeact import run_code
        return await asyncio.to_thread(
            d, lambda: run_code(
                code, _state, _ROOT, {'seq': _job_seq},
                _t_run_compute, _t_job_status, _t_job_result,
                timeout=timeout, t_load=_t_load_archive,
                t_save=_t_save_archive), timeout + 60)


# ============================================================
# 生命周期
# ============================================================
# 端口解析：显式参数 > 环境变量 EL_MCP_PORT > 默认 8765（harness 配置可写死）
# > 被占则随机回退（GUI 日志提示实际端口，.mcp-port 文件始终为准）。
# 环境变量在 start_server 时读取（非模块导入时），保证测试/运行期可覆盖。


def _default_port():
    return int(os.environ.get('EL_MCP_PORT', '8765'))


def _port_free(port):
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def _pick_port():
    want = _default_port()
    if _port_free(want):
        return want, False
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1], True


def _serve_loop(port):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    import uvicorn
    cfg = uvicorn.Config(_server_app, host='127.0.0.1', port=port,
                         log_level='warning', lifespan='on')
    srv = uvicorn.Server(cfg)
    _server_state['srv'] = srv
    with open(_PORT_FILE, 'w', encoding='utf-8') as f:
        f.write(str(port))
    loop.run_until_complete(srv.serve())


_server_app = None
_server_state = {}


def start_server(win, port=None):
    """MainWindow 构造完成后调用：安装派发器 + 起 MCP server 线程。"""
    global _server_app, _state, _mcp, _server_thread
    if _server_thread is not None:
        return _server_state.get('port')
    from .dispatch import init
    init()
    _state = SimState(win)
    get_journal(_ROOT)
    _state.reg.on_op(_journal_bridge)      # 人机写操作统一落账（op→journal）
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        win.log('[AI] mcp SDK 未安装（pip install mcp），AI 接口未启用')
        return None
    _mcp = MCPServer(name='energy-level', instructions=_INSTRUCTIONS)
    _register_tools(_mcp)
    _server_app = _mcp.streamable_http_app(
        host='127.0.0.1', json_response=True, stateless_http=True)
    if port is not None:
        want = port
        if not _port_free(want):
            want, _ = _pick_port()
            win.log(f'[AI] 指定端口 {port} 被占，改用 {want}')
    else:
        want, _ = _pick_port()
    port = want
    _server_state['port'] = port
    _server_thread = threading.Thread(target=_serve_loop, args=(port,),
                                      daemon=True, name='mcp-server')
    _server_thread.start()
    # 等就绪（最多 5s）
    t0 = time.time()
    while time.time() - t0 < 5:
        if _server_state.get('srv') and _server_state['srv'].started:
            break
        time.sleep(0.05)
    win.log(f'[AI] MCP 端点就绪 http://127.0.0.1:{port}/mcp（端口见 .mcp-port）')
    return port


def stop_server():
    srv = _server_state.get('srv')
    if srv is not None:
        srv.should_exit = True
    if os.path.exists(_PORT_FILE):
        try:
            os.remove(_PORT_FILE)
        except OSError:
            pass
