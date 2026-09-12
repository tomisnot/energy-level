# -*- coding: utf-8 -*-
"""进程内 MCP server：把 Energy Level 暴露为 AI harness 的 skill。

运行模式（G3）：软件启动时 start_server() 拉起 streamable HTTP 端点
（127.0.0.1，端口写项目根 .mcp-port 供 harness 配置发现）；harness 随后
启动、连 URL 即用。人不动 AI 动、AI 不动人动，两边经 SimState 同一状态。

线程模型：uvicorn 在专用线程跑 asyncio 循环；工具 handler 内所有 Qt 状态
读写经 dispatch.run_on_main 投递主线程执行（asyncio.to_thread 包裹避免
卡住事件循环）。

v1 策略：AI 写操作先斩后奏（草案 §开放问题，用户拍板）——直接生效，
归因与记账由参数注册表的 op 事件统一驱动（src/pm_setup.py 登记 +
param_manager 框架）：人经 touch→commit 自动 op(source='user')，AI 经
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
from . import run_store

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PORT_FILE = os.path.join(_ROOT, '.mcp-port')
_SHOT_DIR = os.path.join(_ROOT, 'ui_design', 'screenshots', '_ai')
_NPZ_KEEP = 100                   # tempsimu 下 ai_*.npz 保留份数（防无限膨胀）

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


# ---------------------------------------------------------------
# 大数组瘦身：波形包络采样不走工具响应（另设 get_envelope 通道）
# ---------------------------------------------------------------
def _slim_env(env):
    """file 型包络的采样数组 → 摘要。

    2026-09-10：一个 12001 点的 FileEnv 让单次 list_params 响应到 1 MB。
    摘要不丢语义（形状/量程/步长都在），要看或复用完整曲线用 get_envelope。
    """
    if not isinstance(env, (list, tuple)):
        return env
    env = list(env)
    if len(env) < 3 or env[0] != 'file':
        return env
    t, a = env[1], env[2]
    if not (isinstance(t, (list, tuple)) and isinstance(a, (list, tuple))
            and t and a):
        return env
    dt = min(t[i + 1] - t[i] for i in range(len(t) - 1)) if len(t) > 1 else 0.0
    return ['file', {'n': len(t), 't_span_ns': [t[0], t[-1]], 'dt_min_ns': dt,
                     'amp_range': [min(a), max(a)],
                     'note': '采样数组已省略，用 get_envelope(path, index) 取'}]


def _slim_waves(v):
    if not isinstance(v, list):
        return v
    out = []
    for w in v:
        if isinstance(w, dict) and 'env' in w:
            w = dict(w)
            w['env'] = _slim_env(w['env'])
        out.append(w)
    return out


def _slim_params(params):
    """波形参数（dynamic.waves_*）的包络采样 → 摘要；其余原样。

    兼容两种返回形态：
      · get_state  → {参数名: 值}
      · list_params→ {参数名: {schema..., 'value': 值}}（值在 'value' 里）
    漏掉后者会让 list_params 依旧整表回传（2026-09-10 自检抓到）。
    """
    out = {}
    for k, v in (params or {}).items():
        if str(k).startswith('dynamic.waves_'):
            if isinstance(v, dict) and 'value' in v:
                v = dict(v)
                v['value'] = _slim_waves(v['value'])
            else:
                v = _slim_waves(v)
        out[k] = v
    return out


def _t_get_state(scope):
    d = _state.get_state(scope)
    if isinstance(d, dict) and isinstance(d.get('params'), dict):
        d = dict(d)
        d['params'] = _slim_params(d['params'])
    # A9：mode / pair / t_max / npts 与 summary() 同源，免得脚本从
    # get_state('dynamic') 拿不到这些字段、无法可靠自检"当前什么配置"。
    # 只**新增** cfg 键，不覆盖既有字段（保持向后兼容）。
    if isinstance(d, dict) and (not scope or scope == 'dynamic'):
        cfg = {}
        for _k in ('mode', 't_max', 'npts', 'method', 'near_cutoff'):
            try:
                cfg[_k] = _state.reg.read('dynamic.' + _k)
            except Exception:                      # noqa: BLE001 - 缺项忽略
                pass
        for _k in ('R_um', 'theta_deg', 'phi_deg'):
            try:
                cfg[_k] = _state.reg.read('dynamic.pair.' + _k)
            except Exception:                      # noqa: BLE001
                pass
        d = dict(d)
        d['cfg'] = cfg
    return d


def _t_get_envelope(path, index):
    """取一条波形的原始包络：file 型给采样数组，其他型给参数（即全量）。"""
    try:
        waves = _state.reg.read(path)
    except Exception as e:                      # noqa: BLE001
        return {'ok': False, 'msg': f'读 {path} 失败：{e}；可用 '
                                    'dynamic.waves_single / waves_A / waves_B'}
    if not isinstance(waves, list):
        return {'ok': False, 'msg': f'{path} 不是波形列表；可用 '
                                    'dynamic.waves_single / waves_A / waves_B'}
    i = int(index)
    if not (0 <= i < len(waves)):
        return {'ok': False, 'msg': f'index {i} 越界（{path} 共 {len(waves)} 条）'}
    w = waves[i]
    env = list(w.get('env') or [])
    base = {'ok': True, 'path': path, 'index': i, 'name': w.get('name'),
            'freq_mhz': w.get('freq_mhz'), 'intensity': w.get('intensity'),
            'pol': w.get('pol'), 'offset_ns': w.get('offset_ns')}
    if env and env[0] == 'file' and len(env) >= 3:
        base.update({'kind': 'file', 'n': len(env[1]),
                     't_ns': list(env[1]), 'amp': list(env[2])})
    else:
        base.update({'kind': env[0] if env else None, 'env': env,
                     'msg': '非 file 型包络，env 参数即全量'})
    return base


def _running_job():
    """当前仍在跑的演化作业（A3：AI 发起的运行也要**可见、可取消**）。

    本会话实测：AI 提交重活后用户既看不到弹窗、GUI 取消按钮也不覆盖它，
    AI 侧又拿不到 job_id → 双方一起卡在"已有演化在运行"。
    """
    best = None
    for _jid, _j in list(_jobs.items()):
        if _j.get('done') or _j.get('error') or _j.get('cancelled'):
            continue
        _el = time.time() - float(_j.get('started') or time.time())
        if best is None or _jid > best['job_id']:
            best = {'job_id': _jid, 'elapsed_s': round(_el, 1)}
    return best


def _t_list_params(prefix=''):
    """参数 schema（与 MCP 工具 list_params 同源）——A1：CodeAct 里也能查。

    A2（2026-09-12）：`dynamic.psi_*` 的 `options` 此前恒为 null，AI 只能猜能量键
    （实测猜错会静默选到别的能级）→ 这里注入 `list_levels()` 给出的**合法身份键**。
    无面板/无基组时静默跳过（不影响其它参数）。
    """
    params = _slim_params(_state.schema(prefix))
    try:
        if any(k.startswith('dynamic.psi') for k in params):
            lv = _state.list_levels() or {}
            opts = [x.get('key') for x in (lv.get('levels') or [])
                    if x.get('key')]
            if opts:
                for k, v in params.items():
                    if k.startswith('dynamic.psi') and isinstance(v, dict):
                        v['options'] = opts
                        v['options_count'] = len(opts)
    except Exception:                              # noqa: BLE001 - 无面板则跳过
        pass
    return {'params': params}


def _last_job_id():
    """最近一个已完成的作业（缺 job_id 时的默认目标）。"""
    done = [j for j, v in _jobs.items() if v.get('done') and not v.get('error')]
    return max(done) if done else ''


def _t_list_jobs(limit=20):
    """统一作业视图（B1）：AI 作业来自 _jobs，人正在跑的来自 dyn._run_active。

    本会话的痛点：AI 提交的重活用户既看不到也取消不了、AI 也拿不到 job_id
    → 双方一起卡在"已有演化在运行"。这里把两边放到同一张表里。
    """
    rows = []
    for jid, j in list(_jobs.items()):
        el = (time.time() - float(j.get('started') or time.time()))
        st = ('cancelled' if j.get('cancelled') else
              ('done' if j.get('done') else
               ('error' if j.get('error') else 'running')))
        rows.append({'job_id': jid, 'owner': j.get('owner', 'ai'),
                     'kind': j.get('kind'), 'status': st,
                     'elapsed_s': round(el, 1),
                     'error': j.get('error'),
                     'npz': ((j.get('res') or {}).get('npz')
                             if isinstance(j.get('res'), dict) else None)})
    try:
        dyn = _win().dynamic
        if getattr(dyn, '_run_active', False):
            rows.append({'job_id': '(gui)', 'owner': 'human', 'kind': 'gui',
                         'status': 'running', 'elapsed_s': None,
                         'error': None, 'npz': None,
                         'note': '人手点的运行（GUI 取消按钮即可）'})
    except Exception:                              # noqa: BLE001
        pass
    rows.sort(key=lambda r: (r['status'] != 'running', r['job_id']))
    return {'ok': True, 'n': len(rows), 'jobs': rows[:int(limit)]}


def _t_summary():
    out = _state.summary()
    if isinstance(out, dict):
        out = dict(out)
        # A7：结构化能级组 (index, n, l, j) —— 不要再按标签字符串判重/删除
        # （本会话两次踩坑：'60P' 前缀先匹配到 60P_3/2 而删错组；
        #   '58G_{' 让 58G_9/2 被误判"已存在"而漏加）
        import re as _re
        nlj = []
        for _i, _g in enumerate(out.get('groups') or []):
            _m = _re.search(r'\(n=(\d+),l=(\d+),j=([\d.]+)\)', str(_g))
            if _m:
                nlj.append([_i, int(_m.group(1)), int(_m.group(2)),
                            float(_m.group(3))])
        out['groups_nlj'] = nlj
        # A3：暴露当前占用引擎的作业
        out['running_job'] = _running_job()
    return out


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


def _t_run_compute(jid, diag=True):
    """触发与"人点运行"完全同一条 _run 路径；立即返回 job_id。

    完成回收：_run 内部 `done.connect(self._on_evolve_done)` 在连接时解析
    实例属性 → 调用前先在实例上打补丁即可拦截（无需改 ui_dynamic）。
    """
    w = _win()
    dyn = w.dynamic
    # AI 发起的运行默认采诊断（AI 的用途就是要能解释"为什么"）；
    # 人发起的运行仍跟界面模式（上帝模式才采），不让人无谓付成本。
    dyn._ai_wants_diag = bool(diag)
    # AI 发起的运行**不弹成本预检模态**（无人点会卡死主线程）：置旗标，
    # ui_dynamic._preflight_cost 见到它就走「完整物理」路径并只写日志。
    dyn._ai_initiated = True
    if getattr(dyn, '_run_active', False):
        _jobs.pop(jid, None)
        # A3：把"占用者"报出来，AI 侧才知道等谁 / 取消谁
        _rj = _running_job()
        _hint = ("占用作业 %s（已 %.0f s）；可 cancel_job('%s') 或等待"
                 % (_rj['job_id'], _rj['elapsed_s'], _rj['job_id'])
                 if _rj else '先等待或取消')
        return {'ok': False, 'running_job': _rj,
                'msg': '已有演化在运行（人或 AI 发起）：' + _hint}
    _jobs[jid] = {'started': time.time(), 'done': False, 'error': None,
                  'res': None, 'owner': 'ai',
                  'kind': ('pair' if int(getattr(dyn, '_is_two_body', 0) or
                                         getattr(dyn, 'mode', 0) or 0) else
                           'single'),
                  't_max': None}

    orig_s, orig_p = dyn._on_evolve_done, dyn._on_pair_evolve_done
    orig_show = dyn._show_progress

    def _restore():
        dyn._on_evolve_done = orig_s
        dyn._on_pair_evolve_done = orig_p
        dyn._show_progress = orig_show

    def _watch(orig):
        def _f(res, error):
            dyn._ai_initiated = False          # 运行结束 → 清旗标（人随后手动跑会正常弹窗）
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


def _last_diag(job_id=''):
    """→ (diag_dict 或 None, 来源描述)。

    优先级：指定 job → 最近一个带 diag 的 job → GUI 检视器当前装载的。
    """
    if job_id:
        job = _jobs.get(job_id) or {}
        d = (job.get('res') or {}).get('diag')
        return d, f'job {job_id}'
    for jid in reversed(list(_jobs)):
        d = (_jobs[jid].get('res') or {}).get('diag')
        if d:
            return d, f'最近运行 {jid}'
    insp = getattr(getattr(_win(), 'dynamic', None), '_inspector', None)
    d = getattr(insp, '_diag', None) if insp is not None else None
    return d, 'GUI 检视器当前内容'


def _arr(x, dtype=float):
    """None 安全的数组化（`x or []` 对 numpy 数组会歧义，禁止这么写）。"""
    import numpy as _np
    return _np.asarray([] if x is None else x, dtype)


def _np_stat(a):
    """数组 → (min, median, max)（空数组 → None）。"""
    import numpy as _np
    if a is None:
        return None
    a = _np.asarray(a, float).ravel()
    if a.size == 0:
        return None
    return [float(a.min()), float(_np.median(a)), float(a.max())]


def _diag_summary(d):
    """诊断摘要：全是标量/短列表，可直接进 context。"""
    import numpy as _np
    meta = d.get('meta') or {}
    ac = d.get('acize') or {}
    ph = d.get('photon') or {}
    pr = d.get('pairs') or {}
    sv = d.get('solver') or {}
    out = {
        'meta': {k: meta.get(k) for k in (
            'method', 'backend', 'N', 'P', 'dim', 't_end', 'tol', 'max_step',
            'r_crit', 'bw', 'mw_cap', 'mw_max', 'M', 'n_pairs')},
        'labels': (meta.get('labels') or [])[:8],
        'n_labels': len(meta.get('labels') or []),
        'acize': {
            'r_crit': ac.get('r_crit'),
            'fields': ac.get('fields'),
            'shift_range_mhz': _np_stat(ac.get('shift')),
            'n_shifted': (int(_np.count_nonzero(
                _np.asarray(ac.get('shift'), float)))
                if ac.get('shift') is not None else 0),
        },
        'photon': {'Ms': ph.get('Ms'), 'P': ph.get('P'), 'dim': ph.get('dim'),
                   'strides': ph.get('strides'),
                   'nvecs': (list(_np.asarray(ph['nvecs']).ravel())
                             if ph.get('nvecs') is not None else None)},
        'solver': {
            'nfev': sv.get('nfev'), 'n_acc': sv.get('n_acc'),
            'n_rej': sv.get('n_rej'), 'norm_dev': sv.get('norm_dev'),
            'n_steps': int(_arr(sv.get('dt')).size),
            'dt_ns': _np_stat(sv.get('dt'))},
    }
    if pr:
        is_mw = _arr(pr.get('is_mw'), int)
        v = _np.abs(_arr(pr.get('v'), complex))
        dD = _arr(pr.get('dD'), float)
        out['pairs'] = {
            'n': int(is_mw.size),
            'n_microwave': int((is_mw == 1).sum()),
            'n_optical_vdd': int((is_mw == 0).sum()),
            'max_abs_v_mhz': float(v.max()) if v.size else 0.0,
            'dD_mhz': _np_stat(dD),
            'note': ('微波/光场保留判据：成本窗 mw_cap + 精度 Θ≥PHASE_TOL；'
                     'V_dd 走 static_phase_keep（同判据族）'),
        }
    return out


def _downsample(arr, limit):
    import numpy as _np
    a = _np.asarray(arr, float).ravel()
    if a.size <= limit:
        return a.tolist()
    idx = _np.linspace(0, a.size - 1, limit).astype(int)
    return a[idx].tolist()


def _t_get_diag(scope='', job_id='', limit=200, save_npz=False):
    """只读：运行诊断。scope='' → 摘要；其余给分项明细（截断/落盘）。

    摘要默认、明细要显式要、大数组落盘给路径——与 job_result(save_npz) 同纪律。
    """
    import numpy as _np
    d, src = _last_diag(job_id)
    if not d:
        return {'ok': False,
                'msg': '没有诊断数据。诊断只在 mixed 引擎产出：AI 发起的 '
                       'run_compute 现默认采诊断（diag=true）；人发起的运行'
                       '需先切到上帝模式。'}
    scope = (scope or '').strip().lower()
    if scope in ('', 'summary', 'all'):
        return {'ok': True, 'source': src, 'scope': 'summary',
                'summary': _diag_summary(d)}
    meta = d.get('meta') or {}
    if scope == 'pairs':
        pr = d.get('pairs') or {}
        v = _np.abs(_arr(pr.get('v'), complex))
        order = _np.argsort(-v)[:max(1, int(limit))]
        lab = meta.get('labels') or []
        rows = []
        for i in order:
            a = int(pr['a'][i])
            b = int(pr['b'][i])
            rows.append({'a': a, 'b': b,
                         'label_a': lab[a] if a < len(lab) else '',
                         'label_b': lab[b] if b < len(lab) else '',
                         'abs_v_mhz': float(abs(pr['v'][i])),
                         'dD_mhz': float(pr['dD'][i]),
                         'is_mw': int(pr['is_mw'][i]),
                         'w_mhz': float(pr['w'][i])})
        return {'ok': True, 'source': src, 'scope': 'pairs',
                'n_total': int(v.size), 'n_returned': len(rows),
                'by_abs_v_desc': rows}
    if scope in ('solver', 'steps'):
        sv = d.get('solver') or {}
        return {'ok': True, 'source': src, 'scope': 'steps',
                'nfev': sv.get('nfev'), 'n_acc': sv.get('n_acc'),
                'n_rej': sv.get('n_rej'), 'norm_dev': sv.get('norm_dev'),
                'dt_ns_stats': _np_stat(sv.get('dt')),
                't_ns': _downsample(sv.get('t_steps'), int(limit)),
                'dt_ns': _downsample(sv.get('dt'), int(limit))}
    if scope in ('sectors', 'photon'):
        ph = d.get('photon') or {}
        D = ph.get('D_full')
        return {'ok': True, 'source': src, 'scope': 'sectors',
                'Ms': ph.get('Ms'), 'P': ph.get('P'), 'dim': ph.get('dim'),
                'strides': ph.get('strides'),
                'nvecs': (list(_np.asarray(ph['nvecs']).ravel())
                          if ph.get('nvecs') is not None else None),
                'D_full_stats': _np_stat(D),
                'D_full_head': (_downsample(D, min(int(limit), 64))
                                if D is not None else None)}
    if scope in ('traj', 'trajectory'):
        tr = d.get('trajectory') or {}
        amps = tr.get('sector_amps') or {}
        out = {'ok': True, 'source': src, 'scope': 'trajectory',
               't_ns': _downsample(tr.get('t'), int(limit)), 'sectors': {}}
        for k, v in amps.items():
            a = _np.abs(_np.asarray(v))
            out['sectors'][str(k)] = {
                'shape': list(a.shape),
                'max_abs': float(a.max()) if a.size else 0.0,
                'at_last': (_downsample(a[-1], min(int(limit), 32))
                            if a.ndim > 1 else [])}
        return out
    if scope in ('h', 'h_snapshots', 'hamiltonian'):
        snaps = d.get('H_snapshots') or {}
        shp = {str(k): list(_np.asarray(v).shape) for k, v in snaps.items()}
        if not save_npz:
            return {'ok': True, 'source': src, 'scope': 'H',
                    'times_ns': list(snaps), 'shapes': shp,
                    'hint': 'save_npz=true 落盘全量矩阵并返回路径'}
        import time as _time
        outdir = os.path.join(_ROOT, 'ai_results')
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(
            outdir, 'diag_H_%s.npz' % _time.strftime('%Y%m%d_%H%M%S'))
        _np.savez_compressed(
            path, **{('H_%.6g' % float(k)): _np.asarray(v)
                     for k, v in snaps.items()})
        return {'ok': True, 'source': src, 'scope': 'H', 'path': path,
                'times_ns': list(snaps), 'shapes': shp}
    return {'ok': False,
            'msg': "scope 取值：''(摘要) / pairs / steps / sectors / traj / H"}


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


def _res_labels(res, pops2):
    """结果数组的态标签。

    单体（pops）：用 GUI 的 _run_states（单原子态，长度 = 列数）。
    两体（pops_2d 是 484 = 22² 的乘积基）：**必须**用 res['states_2d'] 的
    成对标签。此前直接照搬 _run_states，于是 484 维两体布居被 22 个单原子
    标签标注、索引 ≥22 的退化成裸数字（如键名 "247"），读数被误导
    （2026-09-10 修）。拿不到成对标签时宁可不标，也不误标。
    """
    import numpy as np
    if pops2 is not None:
        s2 = res.get('states_2d')
        if s2 is None:
            return []
        labs = []
        for x in np.asarray(s2).ravel():
            labs.append(str(x.get('label', '?')) if isinstance(x, dict) else str(x))
        return labs
    states = getattr(_win().dynamic, '_run_states', None) or []
    return [s.get('label', '?') for s in states]


def _res_summary(res, save_npz=False, tag=''):
    """演化结果 dict → 摘要（top/终态布居 + 可选 npz 落盘）。

    两体标签取自 res['states_2d'] 的成对标签（见 _res_labels）。
    落盘的 npz 会按 _NPZ_KEEP 保留最近若干份，防 tempsimu 无限膨胀
    （2026-09-10：一次抖动研究就堆了 200+ 个、共 189 MB）。
    """
    import numpy as np
    if res is None:
        return {'msg': '无结果（可能被取消）'}
    t = np.asarray(res.get('t', []))
    pops = res.get('pops')
    pops2 = res.get('pops_2d')
    arr = pops if pops is not None else pops2
    labels = _res_labels(res, pops2 if pops is None else None)
    n_col = (int(np.asarray(arr).shape[1])
             if arr is not None and np.asarray(arr).ndim > 1 else len(labels))
    _card = None
    # D（2026-09-12）：模型卡——把"静默物理"变成自述。
    # 数据来自引擎 diag（未开诊断时给最小集），并做**可表达性判定**。
    try:
        _dg = res.get('diag') or {}
        _meta = (_dg.get('meta') or {}) if isinstance(_dg, dict) else {}
        _solv = (_dg.get('solver') or {}) if isinstance(_dg, dict) else {}
        _pr = (_dg.get('pairs') or {}) if isinstance(_dg, dict) else {}
        _ac = (_dg.get('acize') or {}) if isinstance(_dg, dict) else {}
        _ph = (_dg.get('photon') or {}) if isinstance(_dg, dict) else {}
        _warn = []
        try:
            _wl = []
            for _a in ('single', 'A', 'B'):
                try:
                    _wl += list(_state.reg.read('dynamic.waves_' + _a) or [])
                except Exception:                  # noqa: BLE001
                    pass
            _om = [float(_w.get('freq_mhz') or 0) for _w in _wl]
            _mx = max(_om) if _om else 0.0
            _vmax = float(_pr.get('max_abs_v_mhz') or 0.0)
            if _mx > 0 and _vmax / _mx > 1.0:
                _warn.append('maxΩ/ω=%.2f > 1（已出 RWA 适用区，'
                             '物理存疑，数值未必错）' % (_vmax / _mx))
        except Exception:                          # noqa: BLE001
            _mx = _vmax = 0.0
        if int(_pr.get('n_microwave') or 0) == 0 and int(_meta.get('dim') or 0):
            _warn.append('微波候选对为 0：**基组可能缺中间态**（多光子通道需要'
                         '真实中间态参与）或频率与能级间隔不匹配 —— '
                         '该波不会产生任何转移')
        _card = {
            'method': _meta.get('method'), 'dim': _meta.get('dim'),
            'n_pairs': _pr.get('n') or _meta.get('n_pairs'),
            'n_microwave': _pr.get('n_microwave'),
            'keep': (_meta.get('keep') if 'keep' in _meta else None),
            'acized_states': _ac.get('n_shifted'),
            'P': _ph.get('P') or _meta.get('P'), 'M': _ph.get('Ms'),
            'max_abs_v_mhz': _pr.get('max_abs_v_mhz'),
            'max_omega_over_omega': (round(_vmax / _mx, 4) if _mx else None),
            'tol': _meta.get('tol'), 'max_step': _meta.get('max_step'),
            't_end_ns': _meta.get('t_end'),
            'n_steps': _solv.get('n_steps'), 'nfev': _solv.get('nfev'),
            'norm_dev': _solv.get('norm_dev'),
            'basis_n': len(labels),
            'warnings': _warn,
        }
    except Exception as _e:                        # noqa: BLE001 - 卡片不阻塞结果
        _card = {'error': 'model_card 生成失败：%s' % _e}
    summary = {'t_ns': [float(t[0]), float(t[-1])] if len(t) else None,
               'n_steps': int(len(t)),
               'two_body': pops is None and pops2 is not None,
               'states': labels[:40], 'n_states': n_col}
    summary['model_card'] = _card
    if arr is not None:
        arr = np.asarray(arr)
        avg = arr.mean(axis=0)
        order = np.argsort(-avg)[:8]
        summary['top_states_avg_pop'] = {
            (labels[i] if i < len(labels) else f'#{i}'): round(float(avg[i]), 5)
            for i in order}
        summary['final_pop'] = {
            (labels[i] if i < len(labels) else f'#{i}'): round(float(arr[-1][i]), 5)
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
            summary['npz_pruned'] = _prune_npz()
    return summary


def _prune_npz(keep=_NPZ_KEEP):
    """只保留 tempsimu 下最近的 keep 份 ai_*.npz（按名倒序=时序）。返回删除数。"""
    d = os.path.join(_ROOT, 'tempsimu')
    if keep <= 0 or not os.path.isdir(d):
        return 0
    try:
        names = sorted(n for n in os.listdir(d)
                       if n.startswith('ai_') and n.endswith('.npz'))
    except OSError:
        return 0
    n = 0
    for nm in names[:-keep]:
        try:
            os.remove(os.path.join(d, nm))
            n += 1
        except OSError:
            pass
    return n


def _t_job_result(job_id, save_npz, states=None, series=False):
    job = _jobs.get(job_id)
    if job is None:
        return {'ok': False, 'msg': f'未知 job {job_id}'}
    if not job['done']:
        return {'ok': False, 'msg': '计算未完成，先 job_status 轮询'}
    if job['error']:
        return {'ok': False, 'msg': f"失败: {job['error']}"}
    out = {'ok': True, 'job_id': job_id}
    out['summary'] = _res_summary(job['res'], save_npz, tag=job_id)
    if states:
        # C：请求了指定态 → 摘要里直接带"按态名"的终态（与 result_states 同源）
        _q = _t_result_states(job_id, states, series=bool(series))
        if _q.get('ok'):
            out['summary']['requested'] = _q.get('final')
            if series:
                out['summary']['requested_series'] = _q
    # A5：npz 路径同时给顶层键（单体/两体一致，脚本别再猜键名）
    _p = (out['summary'] or {}).get('npz')
    if _p:
        out['npz'] = _p
    return out


def _indexed_result(job_id):
    """(res, arr[t,state], labels) —— 结果取数的统一入口（C）。"""
    import numpy as np
    job = _jobs.get(job_id)
    if job is None or not job.get('done') or job.get('error'):
        return None, None, None
    res = job.get('res') or {}
    pops, pops2 = res.get('pops'), res.get('pops_2d')
    arr = pops if pops is not None else pops2
    if arr is None:
        return res, None, None
    labels = _res_labels(res, pops2 if pops is None else None)
    return res, np.asarray(arr), labels


def _t_result_states(job_id, states=None, series=True, t0=None, t1=None,
                     max_points=200):
    """按**态名**取结果：终态布居 + （可选）时间序列。

    states：名称片段列表（子串匹配，大小写不敏感）；空 = 全部里取均值最大的
    前 8 个（与 final_pop 同序）。序列按时长**降采样到 ≤max_points**，
    避免把 MB 级数组灌回模型上下文（与 list_params 防膨胀同纪律）。
    """
    import numpy as np
    import re as _re
    res, arr, labels = _indexed_result(job_id)
    if arr is None:
        return {'ok': False, 'msg': '无结果或结果不可用（job=%s）' % job_id}
    t = np.asarray(res.get('t', []), float)
    want = [str(s) for s in (states or []) if str(s).strip()]
    idx, used = [], []
    if want:
        for w in want:
            hit = [i for i, L in enumerate(labels) if w.lower() in L.lower()]
            if not hit:
                m = _re.match(r'^#(\d+)$', w)
                hit = [int(m.group(1))] if m else []
            if hit:
                used.append(w)
                idx.extend(hit)
        miss = [w for w in want if w not in used]
    else:
        avg = arr.mean(axis=0)
        idx = list(np.argsort(-avg)[:8])
        miss = []
    idx = list(dict.fromkeys(idx))
    if t0 is not None or t1 is not None:
        lo = 0 if t0 is None else int(np.searchsorted(t, float(t0)))
        hi = len(t) if t1 is None else int(np.searchsorted(t, float(t1)))
    else:
        lo, hi = 0, len(t)
    step = max(1, int(np.ceil(max(1, hi - lo) / float(max_points))))
    out = {'ok': True, 'job_id': job_id, 'n_states_matched': len(idx),
           'matched': [labels[i] if i < len(labels) else '#%d' % i
                       for i in idx],
           'missing': miss,
           'final': {labels[i] if i < len(labels) else '#%d' % i:
                     round(float(arr[-1][i]), 6) for i in idx}}
    if series:
        out['t_ns'] = t[lo:hi:step].round(3).tolist()
        out['series'] = {labels[i] if i < len(labels) else '#%d' % i:
                         arr[lo:hi:step, i].round(6).tolist() for i in idx}
        out['downsample_step'] = step
    return out


def ui_job_status():
    """GUI 用（B3）：当前 AI 作业（供状态栏常驻指示器）。无 → None。"""
    return _running_job()


def ui_cancel_job(job_id=''):
    """GUI 用（B3）：取消 AI 作业——与 MCP 的 cancel_job **同一条路径**（P2）。"""
    jid = job_id or (ui_job_status() or {}).get('job_id', '')
    if not jid:
        return {'ok': False, 'msg': '当前没有正在跑的 AI 作业'}
    r = _t_cancel_job(jid)
    if isinstance(r, dict):
        r.setdefault('job_id', jid)
    return r


def _t_cancel_job(job_id):
    # B3：取消意图**立刻**落到登记上——GUI 常驻指示器据此即时消失
    #（真正的停算由引擎在下一个进度轮询点完成，二者不必同步）
    try:
        _jobs[job_id]['cancelled'] = True
    except Exception:                              # noqa: BLE001
        pass
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
**结果不会丢**：每次 run_python 都会把 result/stdout/代码存到项目根
ai_results/ 下的 JSON（返回值含 run_id）。超长任务用 run_python_async
立即拿 run_id 后台跑，再用 run_result / list_runs 取回——客户端断线也不丢。
写操作先斩后奏：直接生效、GUI 实时反映；不满意用 undo（只撤 AI 自己的
最近一步；run_python 整段算一步）。get_activity 可查所有操作流水
（人/AI 混合）。"""


def _register_tools(mcp):
    d = run_on_main   # 简写

    @mcp.tool(description='读当前仿真状态（scope: 空=全部 / groups / field / dynamic）。'
                         '波形参数的包络采样只给摘要，完整曲线用 get_envelope。')
    async def get_state(scope: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: _t_get_state(scope or None))

    @mcp.tool(description='轻量摘要：能级组列表、场、方法、波形数（先调这个建立认知）')
    async def summary() -> dict:
        return await asyncio.to_thread(d, _t_summary)

    @mcp.tool(description='改一个参数。path 见 list_params；value 按类型；reason 简述动机')
    async def set_param(path: str, value: Any = None, reason: str = '') -> dict:
        return await asyncio.to_thread(d, lambda: _t_set_param(path, value, reason))

    @mcp.tool(description='列出所有可改参数的 schema（type/unit/desc/range/options/value）。'
                         '波形参数（dynamic.waves_*）的包络采样**只给摘要**'
                         '（点数/dt/幅值范围/时间跨度）——完整曲线用 '
                         'get_envelope(path, index) 取。')
    async def list_params(prefix: str = '') -> dict:
        return await asyncio.to_thread(
            d, lambda: {'params': _slim_params(_state.schema(prefix))})

    @mcp.tool(description='取某个波形参数的**原始包络采样数组**（大数组专用通道）。'
                         'path 如 dynamic.waves_A（或 waves_single / waves_B），'
                         'index 为波形序号，顺序同 get_state 里该列表。'
                         'file 型（光强包络）返回 t_ns / amp 两个等长数组；'
                         'gauss/square 型 env 参数本身就是全量。')
    async def get_envelope(path: str, index: int = 0) -> dict:
        return await asyncio.to_thread(d, lambda: _t_get_envelope(path, index))

    @mcp.tool(description='按**态名**取结果（C）：states 为名称片段列表'
                          '（子串匹配），返回终态布居与（series=true 时）'
                          '降采样时间序列；序列 ≤200 点防上下文膨胀')
    async def result_states(job_id: str = '', states: list = None,
                            series: bool = True, t0: float = None,
                            t1: float = None) -> dict:
        return await asyncio.to_thread(
            d, lambda: _t_result_states(job_id or _last_job_id(),
                                        states, series, t0, t1))

    @mcp.tool(description='统一作业视图（B1）：AI 作业 + 人正在跑的，'
                          '含 owner/status/elapsed。取消用 cancel_job(job_id)')
    async def list_jobs(limit: int = 20) -> dict:
        return await asyncio.to_thread(d, lambda: _t_list_jobs(limit))

    @mcp.tool(description='事前成本（B2）：单体给引擎真值报告要点，两体按实测'
                          '标定 N⁴·t 粗估，并建议是否走异步/开短窗口')
    async def estimate_cost() -> dict:
        return await asyncio.to_thread(d, lambda: _state.estimate_cost())

    @mcp.tool(description='**声明式**设定能级组：让基组恰好包含给定 '
                          '[(n,l,j),…]（原子/幂等，回 {added,removed,kept}）。'
                          '替代按索引增删——本会话两次因按标签判重而删错/漏加组')
    async def set_groups(groups: list, reason: str = '') -> dict:
        return await asyncio.to_thread(
            d, lambda: _state.set_groups(groups, reason=reason))

    @mcp.tool(description='按语义身份设初态：(atom,n,l,j,m)。不在基组则报错'
                          '（不再静默选到别的能级）。list_levels() 看合法项')
    async def set_initial_state(atom: str, n: int, l: int, j: float, m: float,
                                reason: str = '') -> dict:
        return await asyncio.to_thread(
            d, lambda: _state.set_initial_state(atom, n, l, j, m,
                                                reason=reason))

    @mcp.tool(description='基组内子能级清单（含**合法身份键**）——设初态的唯一'
                          '正确入口（此前 options 为 null，只能猜能量）')
    async def list_levels(scope: str = 'basis') -> dict:
        return await asyncio.to_thread(
            d, lambda: _state.list_levels(scope))

    @mcp.tool(description='规范化运行配置（groups/mode/pair/psi/t_max/npts/'
                          '方法/波数）——唯一真值源快照')
    async def get_config() -> dict:
        return await asyncio.to_thread(d, lambda: _state.get_config())

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
    async def run_compute(diag: bool = True) -> dict:
        _job_seq[0] += 1
        jid = f'job{_job_seq[0]}'
        return await asyncio.to_thread(
            d, lambda: _t_run_compute(jid, diag=diag), 300.0)

    @mcp.tool(description='只读：运行诊断。scope 空=摘要（引擎/AC化/'
                         '光子块/耦合对/求解器）；scope=pairs/steps/'
                         'sectors/traj/H 给分项明细（截断；H 需 '
                         'save_npz=true 落盘给路径）。job_id 空=最近'
                         '一次或 GUI 当前。诊断是理解"为什么"的主通道')
    async def get_diag(scope: str = '', job_id: str = '',
                       limit: int = 200, save_npz: bool = False) -> dict:
        return await asyncio.to_thread(
            d, lambda: _t_get_diag(scope, job_id, limit, save_npz))

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
                         '多步编排（省回合）。'
                         '命名空间白名单——**仅**这些名字，无 os / sys / subprocess / open：'
                         'np(=numpy)、json、time、ROOT(项目根绝对路径)、'
                         'summary()、get_state(scope)、set_param(path,value,reason)、'
                         'add_level_group(n,l,j)、remove_level_group(i)、'
                         'set_waves(waves,atom)、load_archive(path)、save_archive(path)、'
                         'run_compute_and_wait(timeout=1800,poll=1.0,save_npz=False)、'
                         'activity(since=0)。'
                         '要把数据写文件请拼路径：np.savez(ROOT + "/ai_results/x.npz", ...)，'
                         '不要用 os.path.join（os 不存在）。'
                         '把结论赋给变量 result（自动回传 + 自动落盘）。'
                         '大数组用 print 出统计量、别整体回传。整段算一个 undo 单元。'
                         '本次执行已自动存为 ai_results/<run_id>.json，返回值含 run_id。')
    async def run_python(code: str, timeout: float = 1800.0) -> dict:
        from .codeact import run_code
        return await asyncio.to_thread(
            d, lambda: run_code(
                code, _state, _ROOT, {'seq': _job_seq},
                _t_run_compute, _t_job_status, _t_job_result,
                timeout=timeout, t_load=_t_load_archive,
                t_save=_t_save_archive, t_list_params=_t_list_params, t_diag=_t_get_diag, t_cancel=_t_cancel_job, t_jobs=_t_list_jobs, t_states=_t_result_states), timeout + 60)

    @mcp.tool(description='CodeAct 异步版：立即返回 run_id，脚本在后台执行，'
                         '不再占住本次调用。长扫描/大系综用它——即使客户端断线，'
                         '结果也已落盘 ai_results/<run_id>.json，重连后用 run_result '
                         '或 list_runs 取回。用 run_result(run_id) 轮询进度。')
    async def run_python_async(code: str, timeout: float = 1800.0) -> dict:
        from .codeact import run_code
        rid = run_store.new_run_id()
        # 先登记 running：保证 run_id 一返回就有效（消除工作线程启动前的
        # 空窗），run_code 内的第二次登记会保留这里的 started（见 run_store）。
        run_store.start(_ROOT, rid, code)

        def _worker():
            try:
                d(lambda: run_code(
                    code, _state, _ROOT, {'seq': _job_seq},
                    _t_run_compute, _t_job_status, _t_job_result,
                    timeout=timeout, t_load=_t_load_archive,
                    t_save=_t_save_archive, run_id=rid, t_list_params=_t_list_params, t_diag=_t_get_diag, t_cancel=_t_cancel_job, t_jobs=_t_list_jobs, t_states=_t_result_states), timeout + 60)
            except Exception:                  # noqa: BLE001 - 落盘失败原因
                import traceback as _tb
                run_store.finish(_ROOT, rid, {
                    'ok': False, 'run_id': rid, 'stdout': '',
                    'error': _tb.format_exc()[-2000:]})

        threading.Thread(target=_worker, daemon=True,
                         name=f'codeact-{rid}').start()
        return {'ok': True, 'run_id': rid,
                'msg': f'已启动（后台执行）。用 run_result({rid!r}) 轮询；'
                       f'记录在 ai_results/{rid}.json'}

    @mcp.tool(description='取 CodeAct 运行的状态与结果。run_id 来自 run_python / '
                         'run_python_async 的返回值，或 list_runs。'
                         '只读 ai_results/ 下的文件、**不经 Qt 主线程**，'
                         '所以长任务正在跑时也能查（这正是它存在的理由：'
                         'run_python 的脚本占用主线程，别的工具会被排队）。')
    async def run_result(run_id: str) -> dict:
        rec = run_store.read(_ROOT, run_id)
        if rec is None:
            return {'ok': False, 'msg': f'未知 run {run_id}；用 list_runs 看最近记录'}
        st = rec.get('status')
        out = {'ok': True, 'run_id': run_id, 'status': st,
               'started_iso': rec.get('started_iso'),
               'elapsed_s': rec.get('elapsed_s'),
               'lines': rec.get('lines')}
        if st == 'running':
            out['live_s'] = (round(time.time() - rec['started'], 1)
                             if rec.get('started') else None)
            out['msg'] = '仍在执行；稍后再查（live_s 是已跑墙钟秒数）'
            return out
        out['ok_run'] = rec.get('ok')
        out['stdout'] = (rec.get('stdout') or '')[-4000:]
        if rec.get('ok'):
            out['result'] = rec.get('result')
        else:
            out['error'] = (rec.get('error') or '')[-3000:]
        return out

    @mcp.tool(description='列出最近的 CodeAct 运行记录（run_id / 状态 / 耗时 / '
                         '是否含结果 / 错误头部）。用途：客户端断线重连后，'
                         '找回之前那次跑丢的 run_id，再用 run_result 取结果。')
    async def list_runs(limit: int = 20) -> dict:
        return {'runs': run_store.list_runs(_ROOT, limit),
                'dir': run_store.results_dir(_ROOT)}


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


def _pick_port(retry_s=10.0, poll_s=0.25):
    """→ (port, fell_back)。

    ⚠ 端口回退是**静默断链**的经典来源：DSH（以及任何 harness）的配置里
    URL 是**写死**的 `http://127.0.0.1:8765/mcp`，一旦这里回退到随机端口，
    harness 侧永远连不上、工具表里一个都不出现，而软件看起来"一切正常"。
    故先把期望端口重试一段时间（覆盖"上一个实例还在退出/端口刚释放"这种
    常见情形），确实占不到才回退，并让调用方把回退**吵出来**。
    """
    want = _default_port()
    t0 = time.time()
    while time.time() - t0 < max(0.0, float(retry_s)):
        if _port_free(want):
            return want, False
        time.sleep(poll_s)
    if _port_free(want):                 # 最后一次机会（退避后刚好释放）
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
        want, _fell_back = _pick_port()
    port = want
    if _fell_back:
        # 吵出来：harness 的配置写死 8765，回退 = 那条 AI 通道静默失效
        win.log(f'[AI] ⚠ 默认端口 {_default_port()} 被占用 → 已回退到 {port}。'
                f'DSH/harness 配置里写死的 http://127.0.0.1:'
                f'{_default_port()}/mcp **连不上**（工具会一个都不出现）。'
                f'请关掉占用该端口的进程后重启软件，或把配置改成 127.0.0.1:'
                f'{port}（也可设环境变量 EL_MCP_PORT={port} 固定下来）。', 30000)
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
