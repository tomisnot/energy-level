# -*- coding: utf-8 -*-
"""扫描作业登记（**提交侧**，供 GUI 与 MCP 共用）。

职责：把一份"配置快照 + 轴"变成一个**独立 runner 进程**，并提供
status / result / cancel / list 四个查询。GUI 与 AI 读**同一份**
`status.json`（P2：人机同源），所以悬浮进度窗与 `sweep_status` 永远一致。

与 runner 的边界只有两样东西：**进程** 与 **文件**。
不共享内存对象、不共享 Qt 对象 —— 这样 runner 崩了 GUI 不受影响。

目录布局（`<root>/runtime/sweeps/<sid>/`）：

    spec.json          提交时的完整规格（params 快照 + 点 + 选项）→ 可复现
    meta.json          pid / python / 创建时刻（取消与排障用）
    status.json        进度（runner 每秒原子覆写）→ GUI 轮询它
    points/pt_XXXX.json  每点摘要（worker 完成即写 → 崩了也不丢）
    points/pt_XXXX.npz   每点数组（record='final'/'pops'）
    logs/worker_<pid>.log, runner.log
    cancel             取消请求（存在即请求取消；runner 优雅退出的通道）
"""
import json
import os
import subprocess
import sys
import time

from . import spec as _spec

SWEEPS_DIRNAME = os.path.join('runtime', 'sweeps')
_TERMINAL = ('done', 'failed', 'cancelled')


# =====================================================================
# 路径与 IO
# =====================================================================
def sweeps_root(root):
    return os.path.join(root, SWEEPS_DIRNAME)


def sweep_dir(root, sid):
    return os.path.join(sweeps_root(root), sid)


def _read_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:                              # noqa: BLE001
        return default


def _write_json(path, obj):
    """提交侧原子写（同样带重试：Windows 下 replace 撞上读者会 WinError 5）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    data = json.dumps(obj, ensure_ascii=False, default=str)
    last = None
    for i in range(10):
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(data)
            os.replace(tmp, path)
            return
        except OSError as e:                       # noqa: PERF203
            last = e
            time.sleep(0.05 * (i + 1))
    del last                                       # 仅用于诊断，不参与控制流
    with open(path, 'w', encoding='utf-8') as f:   # 退路：原地直写
        f.write(data)


def _new_sid():
    return 'sw' + time.strftime('%Y%m%d_%H%M%S') + ('_%03d'
                                                    % (int(time.time() * 1000)
                                                       % 1000))


# =====================================================================
# 并行度：自动按 内存 + 核数 定档（用户裁决）
# =====================================================================
def _mem_helpers():
    """复用实测标定（`src/core/cost.py`，单一真值源；无 Qt 也能 import）。"""
    try:
        from ..core.cost import estimate_pair_peak_gb, avail_ram_gb
        return estimate_pair_peak_gb, avail_ram_gb
    except Exception:                              # noqa: BLE001
        return None, None


def plan_workers(basis_n=None, two_body=None, n_points=None, requested=None,
                 cpu=None, avail_gb=None, mem_fraction=0.4):
    """并行度定档 + **为什么**（返回 dict，`why` 是要回传给 AI/人的理由）。

    约束来源（取三者最小）：
      ① 核数：`cpu - 1`（留一个核给 GUI/系统，否则界面会卡）；
      ② 内存：两体按实测锚点 dim=4624→1.36 GB 做 dim² 外推，单进程峰值
         不超过**可用内存的 40%**（软建议口径，见 A14/#12；
         实测该外推**偏保守上界**，故只当护栏）；
      ③ 点数：没必要比点数还多。
    显式 `requested` 优先，只做 1..4×cpu 的合理性夹取。
    """
    cpu = int(cpu or os.cpu_count() or 4)
    lim_cpu = max(1, cpu - 1)
    lim_mem = None
    per_gb = None
    mem_note = '未按内存限档'
    if two_body and basis_n:
        est_gb, avail = _mem_helpers()
        if est_gb is not None:
            if avail_gb is None:
                try:
                    avail_gb = avail()
                except Exception:                  # noqa: BLE001
                    avail_gb = None
            if avail_gb:
                per_gb = float(est_gb(int(basis_n) ** 2))
                budget = float(avail_gb) * float(mem_fraction)
                lim_mem = max(1, int(budget // max(per_gb, 1e-9)))
                mem_note = ('两体 dim=%d 单进程峰值 ~%.2f GB、可用 ~%.2f GB'
                            '（预算 %.0f%% = %.2f GB）⇒ ≤ %d 进程'
                            % (int(basis_n) ** 2, per_gb, float(avail_gb),
                               100 * mem_fraction, budget, lim_mem))
    lim_pts = max(1, int(n_points)) if n_points else None
    if requested:
        w = max(1, min(int(requested), 4 * cpu))
        why = ('显式指定 %d（夹取到 1..%d）' % (w, 4 * cpu))
    else:
        cands = [lim_cpu] + ([lim_mem] if lim_mem else []) \
            + ([lim_pts] if lim_pts else [])
        w = max(1, min(cands))
        why = ('自动：核数上限 %d（cpu=%d 留一核）'
               '%s%s ⇒ 取 %d'
               % (lim_cpu, cpu,
                  ('；' + mem_note) if lim_mem else '；' + mem_note,
                  ('；点数 %d' % lim_pts) if lim_pts else '', w))
    return {'workers': int(w), 'why': why, 'cpu': cpu, 'lim_cpu': lim_cpu,
            'lim_mem': lim_mem, 'per_worker_gb': per_gb,
            'avail_gb': (None if avail_gb is None else round(float(avail_gb), 2))}


# =====================================================================
# 提交
# =====================================================================
def submit(root, params, *, axes=None, points=None, workers=None, tag=None,
           options=None, basis_n=None, two_body=None, python=None, log=None):
    """提交一次并行扫描，**立即返回**（不等结果）。

    params  = 登记册快照（`sim_state.to_dict()['params']` / 存档同构）
    axes    = {'登记册路径': 取值列表|{'linspace':…}}（见 spec.py）
    points  = [{路径: 取值}, …]（显式逐点；与 axes 同时给则 axes 在前）
    workers = None ⇒ 自动（见 plan_workers）
    options = {record:'final'|'pops'|'none', record_npts:200, diag:false,
               states:[标签片段…], top_n:8}
    basis_n/two_body = 供**并行度内存定档**用（提交侧知道就传）
    """
    deltas = _spec.build_points(axes, points)
    opts = dict(options or {})
    n_points = len(deltas)
    if two_body is None:
        two_body = bool(int((params or {}).get('dynamic.mode') or 0))
    # 并行度：能就地算（给了 basis_n）就现在定；否则交给 runner 用**实际**
    # 基组定档 —— submit 必须**立刻返回**，不能为了一个估算在主线程里
    # 建一次完整基组（大基组要几十秒）。
    auto = (workers is None and basis_n is None)
    plan = plan_workers(basis_n=basis_n, two_body=two_body,
                        n_points=n_points, requested=workers)
    sid = _new_sid()
    sdir = sweep_dir(root, sid)
    os.makedirs(os.path.join(sdir, 'points'), exist_ok=True)

    pyspec = {
        'sweep_id': sid, 'created': time.time(), 'tag': tag or '',
        'root': root, 'dir': sdir,
        'params': params, 'axes': axes or {}, 'points': points or [],
        'deltas': deltas, 'options': opts,
        'plan': plan,
        'workers': (None if auto else plan['workers']),
        'workers_auto': auto,
        # 允许 MCP 工具把它注册的 job_id 带进来：GUI 的「AI 作业中」指示器
        # 会把它算进去，从而**横向也看得见**并行扫描（B3 口径）
        'job_id': opts.get('job_id'),
    }
    spec_path = os.path.join(sdir, 'spec.json')
    _write_json(spec_path, pyspec)

    # 起 runner：**独立进程**，与 GUI 无共享对象（见模块 docstring）
    if python is None:
        if getattr(sys, 'frozen', False):
            return {'ok': False, 'sweep_id': sid, 'dir': sdir,
                    'msg': '打包(frozen)运行下没有可用的 runner 入口：'
                           '并行扫描需要脚本模式（python 能执行 .py）。'
                           '请用开发入口「python 能级.py」启动软件。'}
        python = sys.executable
    script = os.path.join(root, 'scripts', 'sweep_runner.py')
    if not os.path.isfile(script):
        return {'ok': False, 'sweep_id': sid, 'dir': sdir,
                'msg': '找不到 runner 入口：%s' % script}
    env = os.environ.copy()
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'NUMBA_NUM_THREADS'):
        env.setdefault(k, '1')
    # 把本进程的 sys.path 交给 runner：软件本体可能依赖项目外的包
    # （本项目的 param_manager 就是同级另一个仓库，靠 sys.path 才可见）。
    env['EL_SWEEP_SYS_PATH'] = os.pathsep.join(p for p in sys.path if p)
    log_path = os.path.join(sdir, 'runner_out.log')
    kwargs = {'cwd': root, 'env': env}
    if os.name == 'nt':
        kwargs['creationflags'] = 0x00000200        # CREATE_NEW_PROCESS_GROUP
    try:
        outf = open(log_path, 'ab')
        proc = subprocess.Popen([python, script, '--spec', spec_path],
                                stdout=outf, stderr=subprocess.STDOUT,
                                **kwargs)
    except Exception as e:                         # noqa: BLE001
        _write_json(os.path.join(sdir, 'status.json'), {
            'sweep_id': sid, 'state': 'failed', 'n_points': n_points,
            'n_done': 0, 'n_failed': 0, 'workers': plan['workers'],
            'dir': sdir, 'points': [],
            'note': '起 runner 失败：%s' % e})
        return {'ok': False, 'sweep_id': sid, 'dir': sdir,
                'msg': '起 runner 失败：%s' % e}
    _write_json(os.path.join(sdir, 'meta.json'),
                {'pid': proc.pid, 'python': python, 'script': script,
                 'created': time.time(), 'plan': plan})
    why = ('自动：由 runner 按**实际**基组（N）与可用内存定档 —— '
           '结论见 sweep_status 的 workers_why' if auto else plan['why'])
    if log:
        log('并行扫描 %s 已提交：%d 点 × %s 进程（%s）'
            % (sid, n_points, '?' if auto else plan['workers'], why))
    return {'ok': True, 'sweep_id': sid, 'dir': sdir, 'n_points': n_points,
            'workers': (None if auto else plan['workers']),
            'workers_auto': auto, 'workers_why': why,
            'pid': proc.pid, 'tag': tag or '',
            'record': str(opts.get('record') or 'final'),
            'basis_n': basis_n, 'two_body': bool(two_body),
            'msg': '已提交（立即返回）：用 sweep_status/sweep_result 跟进；'
                   '取消用 cancel_sweep'}


# =====================================================================
# 查询
# =====================================================================
def _latest_sid(root):
    r = sweeps_root(root)
    if not os.path.isdir(r):
        return None
    best, bt = None, None
    for nm in os.listdir(r):
        st = _read_json(os.path.join(r, nm, 'status.json'))
        meta = _read_json(os.path.join(r, nm, 'meta.json')) or {}
        t = float((st or {}).get('started') or meta.get('created') or 0)
        if st is None and not meta:
            continue
        if bt is None or t > bt:
            best, bt = nm, t
    return best


def status(root, sweep_id=None, points_limit=200):
    """扫描进度（GUI 悬浮窗与 MCP `sweep_status` 读的是同一份文件）。"""
    sid = sweep_id or _latest_sid(root)
    if not sid:
        return {'ok': False, 'msg': '还没有任何并行扫描'}
    sdir = sweep_dir(root, sid)
    st = _read_json(os.path.join(sdir, 'status.json'))
    if st is None:
        exists = os.path.exists(os.path.join(sdir, 'status.json'))
        return {'ok': False, 'sweep_id': sid,
                'msg': ('status.json 暂时读不到（正被 runner 重写，下一拍自愈）'
                        if exists else
                        'status.json 还没出现（runner 正在预热/导入）')}
    out = dict(st)
    out['ok'] = True
    pts = list(out.get('points') or [])
    out['points'] = pts[:int(points_limit)]
    out['points_truncated'] = max(0, len(pts) - int(points_limit))
    out['running'] = out.get('state') not in _TERMINAL
    return out


def result(root, sweep_id=None, states=None, with_points=True):
    """扫描结果：每点摘要（+ 指定态终态布居 + npz 路径）。**只读文件。**"""
    sid = sweep_id or _latest_sid(root)
    if not sid:
        return {'ok': False, 'msg': '还没有任何并行扫描'}
    sdir = sweep_dir(root, sid)
    pdir = os.path.join(sdir, 'points')
    st = _read_json(os.path.join(sdir, 'status.json')) or {}
    recs = []
    if os.path.isdir(pdir):
        for nm in sorted(os.listdir(pdir)):
            if not nm.endswith('.json'):
                continue
            r = _read_json(os.path.join(pdir, nm))
            if r is None:
                continue
            if states:
                # 点摘要里只算了提交时声明的态；调用方要了别的态 → 现读 npz 补算
                # （record='none' 时补不出来，那就老实给空）
                sel = dict((r.get('summary') or {}).get('selected_final_pop')
                           or {})
                missing = [f for f in states if f not in sel]
                if missing:
                    extra = _select_from_npz(r.get('npz'), missing)
                    sel.update(extra)
                r = dict(r)
                r['selected_final_pop'] = sel
            recs.append(r)
    good = [r for r in recs if r.get('ok')]
    bad = [r for r in recs if not r.get('ok')]
    # 汇总矩阵：把每点"关注态"的终态布居拉成一张表（AI 最想要的形态）
    table = _summary_table(good, states) if states else None
    return {
        'ok': True, 'sweep_id': sid, 'state': st.get('state'),
        'dir': sdir, 'n_points': st.get('n_points'),
        'n_done': len(good), 'n_failed': len(bad),
        'elapsed_s': st.get('elapsed_s'),
        'tag': st.get('tag'),
        'deltas': st.get('deltas') if False else None,
        'points': (recs if with_points else None),
        'selected_table': table,
        'note': ('仍在运行：已完成 %d 点，结果只含已完成的' % len(good)
                 if st.get('state') not in _TERMINAL else None),
    }


def _select_from_npz(npz_path, states):
    if not npz_path or not os.path.isfile(npz_path):
        return {}
    try:
        import numpy as np
        d = np.load(npz_path, allow_pickle=False)
        labels = [str(x) for x in d['labels']]
        final = np.asarray(d['final_pop'], float)
        out = {}
        for frag in states:
            hits = [i for i, L in enumerate(labels) if frag in L]
            if hits:
                out[frag] = {labels[i]: round(float(final[i]), 6)
                             for i in hits}
        return out
    except Exception:                              # noqa: BLE001
        return {}


def _summary_table(good, states):
    """每点 → 关注态的终态布居（列 = 态片段，行 = 点）。便于直接看趋势。"""
    rows = []
    for r in good:
        sel = r.get('selected_final_pop') or {}
        row = {'k': r.get('k'), 'tag': r.get('tag'), 'elapsed_s':
               r.get('elapsed_s')}
        for frag in states:
            got = sel.get(frag) or {}
            # 同名片段可能命中多个态 → 用合计（最常用的口径）
            row[frag] = (round(sum(got.values()), 6) if got else None)
        rows.append(row)
    return rows


def list_sweeps(root, limit=20):
    """最近若干次扫描（与 list_jobs 同构的作业视图）。"""
    r = sweeps_root(root)
    if not os.path.isdir(r):
        return {'ok': True, 'sweeps': [], 'n': 0}
    rows = []
    for nm in os.listdir(r):
        st = _read_json(os.path.join(r, nm, 'status.json'))
        meta = _read_json(os.path.join(r, nm, 'meta.json')) or {}
        if st is None and not meta:
            continue
        rows.append({
            'sweep_id': nm,
            'state': (st or {}).get('state') or 'starting',
            'tag': (st or {}).get('tag') or '',
            'n_points': (st or {}).get('n_points'),
            'n_done': (st or {}).get('n_done'),
            'n_failed': (st or {}).get('n_failed'),
            'workers': (st or {}).get('workers') or
                       (meta.get('plan') or {}).get('workers'),
            'elapsed_s': (st or {}).get('elapsed_s'),
            'started': (st or {}).get('started') or meta.get('created'),
            'pid': meta.get('pid'),
        })
    rows.sort(key=lambda x: float(x.get('started') or 0), reverse=True)
    return {'ok': True, 'n': len(rows), 'sweeps': rows[:int(limit)]}


# =====================================================================
# 取消
# =====================================================================
def cancel_request(root, sweep_id=None):
    """**只写 cancel 文件就返回**（不等待）——GUI 按钮走这条。

    GUI 主线程绝不能被"等 runner 退出"堵住（本项目的 Qt 事故史），
    等待与结果判断交给 1 Hz 的 status 轮询（`status()` 里会看到
    state 变 cancelling/cancelled）。
    """
    sid = sweep_id or _latest_sid(root)
    if not sid:
        return {'ok': False, 'msg': '还没有任何并行扫描'}
    sdir = sweep_dir(root, sid)
    st = _read_json(os.path.join(sdir, 'status.json')) or {}
    if st.get('state') in _TERMINAL:
        return {'ok': True, 'sweep_id': sid, 'already': True,
                'state': st.get('state')}
    try:
        with open(os.path.join(sdir, 'cancel'), 'w', encoding='utf-8') as f:
            f.write(str(time.time()))
    except Exception as e:                         # noqa: BLE001
        return {'ok': False, 'sweep_id': sid, 'msg': '写 cancel 失败：%s' % e}
    return {'ok': True, 'sweep_id': sid, 'requested': True,
            'msg': '已请求取消（worker 会在安全点退出，已完成点保留）'}


def cancel(root, sweep_id=None, wait_s=20.0, log=None):
    """取消扫描：先**优雅**（写 cancel 文件 → worker 在安全点退出，
    已完成的点保留），超时才硬杀进程树。"""
    sid = sweep_id or _latest_sid(root)
    if not sid:
        return {'ok': False, 'msg': '还没有任何并行扫描'}
    sdir = sweep_dir(root, sid)
    st = _read_json(os.path.join(sdir, 'status.json')) or {}
    if st.get('state') in _TERMINAL:
        return {'ok': True, 'sweep_id': sid, 'already': True,
                'state': st.get('state'), 'msg': '该扫描已结束（%s）'
                % st.get('state')}
    req = cancel_request(root, sid)
    if not req.get('ok'):
        return req
    t0 = time.time()
    while time.time() - t0 < float(wait_s):
        st = _read_json(os.path.join(sdir, 'status.json')) or {}
        if st.get('state') in _TERMINAL:
            return {'ok': True, 'sweep_id': sid, 'state': st.get('state'),
                    'elapsed_s': round(time.time() - t0, 1),
                    'msg': '已取消（优雅退出，已完成点保留）'}
        time.sleep(0.5)
    pid = (_read_json(os.path.join(sdir, 'meta.json')) or {}).get('pid')
    killed = False
    if pid and os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                           capture_output=True, timeout=30)
            killed = True
        except Exception:                          # noqa: BLE001
            pass
    return {'ok': True, 'sweep_id': sid, 'killed_tree': killed,
            'msg': ('优雅退出超时 → 已硬杀进程树（未完成的点丢失，'
                    '已落盘的点保留）' if killed
                    else '已请求取消，但退出较慢，请稍后再查 status')}
