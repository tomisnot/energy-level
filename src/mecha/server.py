# -*- coding: utf-8 -*-
"""MCP 写通道（localhost）+ 通用工具 + 人运行的 launcher。

拓扑（纲领 §10 / T4）：**写通道永不出本机**；由**人运行的 launcher** 起
（`python -m src.mecha.server`），harness 靠 `.mcp-port` 端口文件发现并 attach
——**harness 绝不拉起权威**（尤其硬件档：不能让 agent 去 spawn 控制激光的进程）。

工具是 `session/ops` 的**薄封装**（不套层检查：不含业务逻辑副本）：
    list_params   分页/前缀/tier 过滤（渐进披露，省 context）
    set_param     经 session.set（回执带 resolved + choke point 产出的 readback，M2）
    get_state     当前配置快照 + 定位包（只读、幂等）
    get_activity  diff-since-seq（读通道头号原语，R1）
    undo          撤销最近一次写入（L2 可逆）
    estimate_cost dry-run 成本预估（D7 契约先行，不跑演化）
    run_compute   复用 runspec 唯一分派（I5 人机同一路径）

三条纪律：
- **I4 同一把体积闸**：每个回程都过 `bus.gate_value`；截断必带"截了多少+原文在哪"。
- **D2 可教学错误**：MechaError → `{ok:False, error:{kind,hint,suggest}}`，第一次
  失败就能改对。
- **人机同路径（I5）**：run_compute 与"人点运行"走同一段 runspec。

工具函数带**真实签名**（MCPServer 从签名推输入 schema）；体内用 `_safe(lambda:…)`
统一做"体积闸 + 错误体"，逻辑全在 session/ops（薄封装）。
"""
import asyncio
import json
import os
import socket

from .errors import MechaError

__all__ = ['TOOL_NAMES', 'create_server', 'find_free_port', 'write_port_file',
           'read_port_file', 'port_file_path', 'run_streamable_http']

TOOL_NAMES = ['list_params', 'set_param', 'get_state', 'list_levels',
              'get_activity', 'undo', 'estimate_cost', 'preview_behavior',
              'run_compute', 'submit_job', 'job_status', 'cancel_job',
              'list_jobs', 'set_drive', 'get_waves', 'mecha_script']

# run_compute 的 auto 档：预估耗时 > 本阈值 → 自动转 async job（不阻塞写通道线程）。
# 两体演化动辄分钟级——这正是旧项目“不适清单”里“长任务必须 async”的教训。
ASYNC_THRESHOLD_SEC = 30.0

_INSTRUCTIONS = (
    'Re0 物理引擎语义内核的写通道。list_params 看可调项（默认隐藏引擎内部参数），'
    'set_param 逐项写配置（记进事件总线、可 undo），estimate_cost 看代价，'
    'preview_behavior 提交前预检“这条脉冲驱动哪些跃迁/基组够不够”（不跑演化），'
    'run_compute 跑演化（超阈值自动转 async）。长任务用 submit_job/job_status/'
    'cancel_job/list_jobs（异步、断线可找回）。键名走 registry 契约；打错会给建议。')


# =====================================================================
# 端口文件发现（harness attach 用；T4/T6）
# =====================================================================
def port_file_path():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo, '.mcp-port')


def find_free_port(preferred=8765):
    """取可用端口：preferred 空闲则用之，否则让 OS 分配（ephemeral）。"""
    for port in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError('找不到可用端口')


def write_port_file(port, path=None):
    path = path or port_file_path()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(int(port)))
    return path


def read_port_file(path=None):
    path = path or port_file_path()
    try:
        with open(path, encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


# =====================================================================
# 建 server（工具 = session/ops 的薄封装）
# =====================================================================
def create_server(session, ops, jobs=None, fields=None, actor='ai', log_level='INFO'):
    """→ (MCPServer, tools_dict)。tools_dict[name] 是**已注册的**工具函数
    （返回 JSON 字符串），判据可直接调用而无需起 HTTP（同一份代码路径）。
    log_level：传给 MCPServer（判据用 'ERROR' 压低 uvicorn 噪声）。
    """
    from typing import Any, Optional
    from mcp.server.mcpserver import MCPServer
    from . import registry
    from .bus import gate_value
    from .jobs import JobRunner
    from . import runtime

    root = session.bus.root
    jobs = jobs if jobs is not None else JobRunner(ops)
    if fields is None:
        try:
            fields = runtime.engine().make_fields(session)   # 建场层经 engine（领域）
        except (NotImplementedError, RuntimeError):
            fields = None
    srv = MCPServer('re0-mecha', instructions=_INSTRUCTIONS, log_level=log_level)

    # 工具回程预算（比 journal 事件字段大）：分页后的正常回程不该被截，
    # 仅病态回程（如内联大数组）触发截断并告知。
    _RETURN_BUDGET = 65536

    def _emit(obj):
        return json.dumps(gate_value(root, obj, _RETURN_BUDGET),
                          ensure_ascii=False, default=str)

    def _safe(thunk):
        """执行 thunk → 过闸 → JSON；MechaError/异常 → 可教学错误体（不 500）。"""
        try:
            return _emit(thunk())
        except MechaError as e:
            return _emit({'ok': False, 'error': e.to_dict()})
        except Exception as e:                           # noqa: BLE001
            return _emit({'ok': False,
                          'error': {'kind': type(e).__name__,
                                    'message': str(e)}})

    @srv.tool(name='list_params',
              description='列出可调参数（分页/前缀/家族；默认隐藏引擎内部参数）。'
                          '返回每项 name/type/unit/range/desc/default。')
    def list_params(prefix: str = '', family: str = '',
                    include_internal: bool = False, page: int = -1) -> str:
        return _safe(lambda: registry.list_params(
            prefix=prefix or None, family=family or None,
            include_internal=bool(include_internal),
            page=None if int(page) < 0 else int(page)))

    @srv.tool(name='set_param',
              description='写一个配置键（经 choke point）。回执带 resolved 与 '
                          'choke point 读回的 readback；非法值报错且不落地。')
    def set_param(key: str, value: Any, reason: str) -> str:
        return _safe(lambda: session.set(key, value, actor=actor, reason=reason))

    @srv.tool(name='get_state',
              description='当前配置快照 + 定位包（只读、幂等）。')
    def get_state() -> str:
        return _safe(lambda: ops.state())

    @srv.tool(name='list_levels',
              description='列出当前基组的能级态（label + 身份键 + 能量）——选初态/跃迁'
                          '端点用，不猜索引。身份键给 set_param("dynamic.psi_single", key)。')
    def list_levels() -> str:
        return _safe(lambda: ops.list_levels())

    @srv.tool(name='get_activity',
              description='diff-since-seq：给 since_seq 之后的事件（只读）。')
    def get_activity(since_seq: int = 0) -> str:
        def _do():
            evs = ops.activity(int(since_seq or 0))
            return {'since_seq': int(since_seq or 0), 'count': len(evs),
                    'last_seq': session.bus.last_seq, 'events': evs}
        return _safe(_do)

    @srv.tool(name='undo', description='撤销最近一次 set（L2 可逆）。')
    def undo(reason: str = 'undo') -> str:
        return _safe(lambda: ops.undo(actor=actor, reason=reason))

    @srv.tool(name='estimate_cost',
              description='dry-run 成本预估（不跑演化，D7 契约先行）。')
    def estimate_cost() -> str:
        return _safe(lambda: ops.estimate())

    @srv.tool(name='preview_behavior',
              description='提交前**行为预览**（D7，不跑演化）：这条脉冲驱动哪些跃迁、'
                          '各多强、基组够不够（缺谁）。复用 collect_transitions。')
    def preview_behavior() -> str:
        return _safe(lambda: ops.preview())

    @srv.tool(name='run_compute',
              description='跑演化（复用 runspec 唯一分派，I5）。mode=auto（默认）：'
                          '预估耗时 > 阈值自动转 async job（回 job_id，不阻塞写通道）；'
                          'sync 强制同步、async 强制后台。')
    def run_compute(reason: str, diag: bool = False, mode: str = 'auto') -> str:
        def _do():
            m = (mode or 'auto').lower()
            if m == 'async':
                jid = jobs.submit(reason=reason, actor=actor)
                return {'ok': True, 'async': True, 'job_id': jid,
                        'note': '已转后台 job；用 job_status 轮询、断线可找回'}
            if m == 'auto':
                est_sec = ops.estimate().get('est_sec') or 0.0
                if est_sec > ASYNC_THRESHOLD_SEC:
                    jid = jobs.submit(reason=reason, actor=actor)
                    return {'ok': True, 'async': True, 'job_id': jid,
                            'est_sec': est_sec,
                            'note': '预估 %.0fs > 阈值 %.0fs → 自动转 async（长任务'
                                    '不阻塞写通道）；用 job_status 轮询'
                                    % (est_sec, ASYNC_THRESHOLD_SEC)}
            return ops.run(actor=actor, reason=reason, diag=bool(diag))
        return _safe(_do)

    @srv.tool(name='submit_job',
              description='把当前配置作为后台 job 发起（长任务；断线可按 job_id 找回）。')
    def submit_job(reason: str) -> str:
        return _safe(lambda: {'ok': True, 'async': True,
                              'job_id': jobs.submit(reason=reason, actor=actor)})

    @srv.tool(name='job_status',
              description='轮询 job 状态/进度/结果（进度=[阶段内进度,取消标志,阶段]）。')
    def job_status(job_id: str) -> str:
        return _safe(lambda: jobs.poll(job_id))

    @srv.tool(name='cancel_job',
              description='请求取消 job（写取消标志，引擎在步间复核后停）。')
    def cancel_job(job_id: str) -> str:
        return _safe(lambda: jobs.cancel(job_id))

    @srv.tool(name='list_jobs',
              description='列出在册 job（断线重连后按 id 找回结果）。')
    def list_jobs() -> str:
        return _safe(lambda: {'jobs': jobs.list_jobs()})

    @srv.tool(name='set_drive',
              description='L2 意图级建场：在某跃迁上建驱动。给 src/tgt（label 或身份键）'
                          '+ Rabi 或强度 + 脉冲类型(pi/pi2)或时长 → 自动算 freq/intensity/duration（§2）。')
    def set_drive(reason: str, src: Any, tgt: Any, atom: str = 'single',
                  pol: str = 'π', rabi_mhz: Optional[float] = None,
                  intensity_mwcm2: Optional[float] = None,
                  detuning_mhz: float = 0.0, pulse: str = '',
                  dur_ns: Optional[float] = None, edge_ns: float = 1.0,
                  name: str = '') -> str:
        return _safe(lambda: fields.set_drive(
            atom, src, tgt, pol=pol, rabi_mhz=rabi_mhz,
            intensity_mwcm2=intensity_mwcm2, detuning_mhz=detuning_mhz,
            pulse=(pulse or None), dur_ns=dur_ns, edge_ns=edge_ns,
            name=(name or None), actor=actor, reason=reason))

    @srv.tool(name='get_waves',
              description='读回某原子的波形摘要（不内联大数组，§4；要曲线走 CodeAct get_envelope）。')
    def get_waves(atom: str = 'single') -> str:
        return _safe(lambda: fields.get_waves(atom))

    @srv.tool(name='mecha_script',
              description='多步编排（表达自由拉满）：在受限命名空间跑一段脚本，只暴露 '
                          'registry/session/ops/fields/jobs 门面 + math（够不到裸引擎，改状态只能经 '
                          'session.set、自动记事件）。脚本把产物赋给 result；print 被捕获。'
                          '（名 mecha_script 而非 codeact，避与 harness 自带 codeact 重名）')
    def mecha_script(script: str) -> str:
        from .codeact import run_codeact
        return _safe(lambda: run_codeact(script, session, ops, jobs, fields))

    tools = {'list_params': list_params, 'set_param': set_param,
             'get_state': get_state, 'list_levels': list_levels,
             'get_activity': get_activity,
             'undo': undo, 'estimate_cost': estimate_cost,
             'preview_behavior': preview_behavior, 'run_compute': run_compute,
             'submit_job': submit_job, 'job_status': job_status,
             'cancel_job': cancel_job, 'list_jobs': list_jobs,
             'set_drive': set_drive, 'get_waves': get_waves,
             'mecha_script': mecha_script}
    return srv, tools


# =====================================================================
# 运行 + launcher
# =====================================================================
async def run_streamable_http(srv, host='127.0.0.1', port=8765):
    await srv.run_streamable_http_async(host=host, port=port,
                                        streamable_http_path='/mcp')


# 领域 launcher（构造引擎 + 起服务）已移到 `src/adapter/serve.py`（人运行
# `python -m src.adapter.serve`）——mecha 不认领域，不在此构造 EngineAdapter。
