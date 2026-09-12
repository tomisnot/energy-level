# -*- coding: utf-8 -*-
"""CodeAct 通道：给 AI 一个受限的 run_python 工具，一次调用完成任意编排。

为什么需要（讨论结论）：细粒度 MCP 工具每个动作占一个模型回合，参数扫描
这类重复劳动要 20+ 回合；而"文本→动作"的翻译若发生在模型写的代码里，
一次调用即可跑完整个循环，且大数组在进程内被 numpy 消化、只有文本回
context。

安全与等效边界：
  - 命名空间只注入语义入口（注册表方法 + job 助手 + numpy/json/time），
    不给 os/sys/subprocess——不是防模型（本机本就有权限），而是保证
    每次状态变更都走 ParamRegistry → 走控件同路 → op 归因（P2/P3 不因
    传输形态而失效）。
  - 代码在 Qt 主线程 exec（可直接碰控件）。整段执行期间 journal 逐 op
    照记但**抑制快照**（suppress_snapshot），结束后补记一条带执行前
    快照的 run_python op——undo 一步回到脚本开始前，流水仍完整。
  - 等待计算用泵事件循环（processEvents），不冻结 GUI，人可随时取消。

结果落盘（2026-09-10 加）：每次执行都在 `<root>/ai_results/` 留一份 JSON
记录（先 running、结束覆写 ok/error），见 run_store 模块。这条是旁路：
落盘失败不影响脚本本身，但**结果不再依赖调用方是否收到响应**——客户端
断线后可用 list_runs / run_result 取回。
"""
import io
import json
import time
import traceback
from contextlib import redirect_stdout

import numpy as np

from . import run_store
from .journal import get_journal


def _make_namespace(state_mod, root, jobs, t_run, t_status, t_result,
                    t_load=None, t_save=None, t_list_params=None, t_diag=None,
                    t_cancel=None):
    """构造受限命名空间。

    函数语义与细粒度 MCP 工具**保持一致**（同名、返回 dict、接受
    reason），模型在两个通道共用一套心智模型。
    """

    def run_compute_and_wait(timeout=1800, poll=1.0, save_npz=False):
        """启动演化并阻塞等待完成（期间泵 GUI 事件，人可看可取消）。

        返回结果摘要 dict（含 top_states_avg_pop / final_pop / npz 路径）。
        """
        from PyQt5.QtWidgets import QApplication
        seq = jobs['seq']
        seq[0] += 1
        jid = f'job{seq[0]}'
        r = t_run(jid)
        if not r.get('ok'):
            return {'ok': False, 'msg': r.get('msg')}
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = t_status(jid)
            if st.get('done'):
                break
            QApplication.processEvents()
            time.sleep(poll)
        else:
            return {'ok': False, 'job_id': jid, 'msg': '等待超时'}
        res = t_result(jid, save_npz)
        res['job_id'] = jid
        # A5：把 npz 路径提到顶层——单体 / 两体一致，脚本不必猜键名
        _sm = res.get('summary') or {}
        if isinstance(_sm, dict) and _sm.get('npz') and not res.get('npz'):
            res['npz'] = _sm['npz']
        return res

    ns = {
        'np': np, 'numpy': np,
        'json': json,
        'time': time,
        'ROOT': root,
        'get_state': state_mod.get_state,
        'summary': state_mod.summary,
        'set_param': state_mod.set_param,
        'add_level_group': state_mod.add_level_group,
        'remove_level_group': state_mod.remove_level_group,
        'set_waves': state_mod.set_waves,
        'load_archive': t_load or (lambda path: {'ok': False,
                                                 'msg': 'load_archive 不可用'}),
        'save_archive': t_save or (lambda path='save_state/ai_state.json':
                                   {'ok': False, 'msg': 'save_archive 不可用'}),
        'run_compute_and_wait': run_compute_and_wait,
        'activity': lambda since=0: get_journal(root).since(since),
        # 二期 A：语义状态操作（P1/P5）——存在才注入，兼容旧 state_mod
        **{k: getattr(state_mod, k) for k in
           ('set_groups', 'set_initial_state', 'list_levels', 'get_config')
           if callable(getattr(state_mod, k, None))},
        # A1：只读查询 / 取消类工具进白名单——脚本里"先查再改"不必跳出去
        # 发独立 MCP 调用（本会话曾因缺 list_params 白跑一整个脚本）
        'job_status': t_status,
        'job_result': t_result,
    }
    for _k, _v in (('list_params', t_list_params), ('get_diag', t_diag),
                   ('cancel_job', t_cancel)):
        if _v is not None:
            ns[_k] = _v
    return ns


def run_code(code, state_mod, root, jobs, t_run, t_status, t_result,
             timeout=1800.0, t_load=None, t_save=None, run_id=None,
             t_list_params=None, t_diag=None, t_cancel=None):
    """在主线程 exec 模型写的代码；返回 {ok, run_id, stdout, result, error}。

    调用方（server 的 run_python / run_python_async）负责经
    dispatch.run_on_main 投递到这里。run_id 由调用方给定或此处自动生成；
    无论传不传，本次执行都会在 ai_results/ 留一份记录。
    """
    rid = run_id or run_store.new_run_id()
    run_store.start(root, rid, code)     # 先落 running，断线也能看到在跑
    j = get_journal(root)
    snap = state_mod.to_dict()
    buf = io.StringIO()
    env = _make_namespace(state_mod, root, jobs, t_run, t_status, t_result,
                          t_load=t_load, t_save=t_save,
                          t_list_params=t_list_params, t_diag=t_diag,
                          t_cancel=t_cancel)
    env['result'] = None
    j.suppress_snapshot = True          # 脚本内逐 op 照记、快照统一挂到末尾
    t0 = time.time()
    try:
        compiled = compile(code, '<ai_code>', 'exec')
        with redirect_stdout(buf):
            exec(compiled, env)               # noqa: S102 - 设计决定（见模块 docstring）
        out = {'ok': True, 'run_id': rid,
               'stdout': buf.getvalue()[-8000:],
               'result': _jsonable(env.get('result')),
               'elapsed_s': round(time.time() - t0, 2)}
        run_store.finish(root, rid, out, code=code)
        j.record('ai', 'run_python',
                 {'lines': len(code.splitlines()), 'run_id': rid}, '', snap)
        return out
    except Exception:                          # noqa: BLE001
        tb = traceback.format_exc()
        out = {'ok': False, 'run_id': rid,
               'stdout': buf.getvalue()[-8000:],
               'error': tb[-2000:],
               'elapsed_s': round(time.time() - t0, 2)}
        run_store.finish(root, rid, out, code=code)
        j.record('ai', 'run_python_failed',
                 {'run_id': rid, 'error': tb[-500:]}, '', snap)
        return out
    finally:
        j.suppress_snapshot = False


def _jsonable(v):
    """result 变量 → JSON 友好（numpy/数组收敛，超大只给形状摘要）。"""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in list(v.items())[:50]}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in list(v)[:50]]
    if isinstance(v, np.ndarray):
        return {'__ndarray__': list(v.shape),
                'note': '数组过大未回传，请用 print 输出统计量或存 npz'}
    try:
        return str(v)[:500]
    except Exception:                          # noqa: BLE001
        return repr(type(v))
