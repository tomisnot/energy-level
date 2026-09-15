# -*- coding: utf-8 -*-
"""并行扫描执行器（headless）：多进程扫描，跑在**独立 runner 进程**里。

## 为什么是独立进程，而不是 GUI 里的进程池

本项目有 Qt 线程生命周期的事故史（2026-09-12：`QThread: Destroyed while
thread is still running` → 0xc0000409，整个进程静默消失、成果全丢）。
把 `multiprocessing` 池塞进 Qt 进程会同时引入"spawn 重新导入 GUI 入口"与
"池子崩了带走软件"两种风险。所以：

    GUI / MCP  ──提交──▶  独立 runner 进程  ──spawn──▶  N 个 worker 进程

runner 崩了软件照常活着；runner 也能**脱离软件单跑**（headless 复现、自检）。

## 三条与现有架构的接缝

1. **引擎调用**走 `src/core/runspec.evolve_run` —— 与"人点运行"同一条路径；
   配置解析走 `runspec.build_run`，输入是**登记册快照**（= 存档里的那份
   params），所以"扫描"在构造上就是"同一份配置跑 N 次"。
2. **进度**用共享内存 `Array('d', 3N)`，每点 3 槽正是引擎现成的
   `progress_arr = [阶段内进度, 取消标志, 阶段]`。父进程只读它写
   `status.json`；写 `[1] = 1.0` 就是引擎**现成的取消路径**（零改引擎）。
3. **成果**每点自己落盘（`pt_XXXX.npz` + `pt_XXXX.json`），**跑完一个是一个**：
   runner 被硬杀也只丢正在跑的那几个点。

## 已知约束

- 需要**非 frozen** 解释器（`sys.executable` 能执行 .py）。PyInstaller 分发包
  下没有可用的 runner 入口，此时提交侧应明确报错而不是静默退化。
- 子进程一律 `OMP/MKL/NUMBA/...=1`：否则 W 个 worker × 16 线程互相超订，
  比串行还慢。
- ARC 的 SQLite 首访会 CREATE TABLE，多进程同时来会 `database is locked`
  （实测不预热只成 1/14）。故**父进程先建一次缓存并预热**，再放 worker。
"""
import json
import os
import sys
import time
import traceback

# 每个 worker 进程的私有上下文（由 initializer 填）
_W = {}


# =====================================================================
# 工具
# =====================================================================
def _now():
    return time.time()


def _write_json(path, obj, retries=10, backoff=0.05):
    """原子写 JSON（tmp + replace），**带重试**。

    ⚠ Windows 实测坑（2026-09-15，聚合回归抓出来的）：`os.replace` 在目标文件
    正被**别的进程读取**时会抛 `PermissionError: [WinError 5] 拒绝访问`。
    而 status.json 天生就是"runner 一直写、GUI/MCP 一直读"的热文件 ⇒
    不重试的话，一次读到的瞬间就能把**整个 runner 打死**（实测 4 点里跑完 3 点
    仍被标 failed，成果虽在但扫描"看起来失败了"）。

    策略：短暂退避重试；仍失败则退化为**原地直写**（可能被读到半截，但
    `_read_json` 会把半截当"暂时读不到"，下一拍自愈）——**绝不因为写进度
    而中断计算**。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    data = json.dumps(obj, ensure_ascii=False, default=_json_default)
    last = None
    for i in range(max(int(retries), 1)):
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(data)
            os.replace(tmp, path)
            return True
        except OSError as e:                       # noqa: PERF203
            last = e
            time.sleep(backoff * (i + 1))
    try:                                           # 退路：直写（非原子）
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
        return True
    except OSError:
        raise last if last is not None else OSError('写 %s 失败' % path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:                              # noqa: BLE001
        pass
    return str(o)


def _tag_of(delta, maxlen=56):
    """扫描点 → 人类可读短标签（给悬浮进度窗与结果表用）。"""
    parts = []
    for k, v in (delta or {}).items():
        sv = repr(v)
        if len(sv) > 18:
            sv = sv[:15] + '…'
        parts.append('%s=%s' % (k.split('.')[-1], sv))
    s = ', '.join(parts)
    return s if len(s) <= maxlen else s[:maxlen - 1] + '…'


# =====================================================================
# worker 侧
# =====================================================================
def _set_env_single_thread():
    """子进程内别让 BLAS/numba 各开一堆线程（W 路并行时等于自伤）。"""
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'NUMBA_NUM_THREADS'):
        os.environ.setdefault(k, '1')


def _worker_init(light, shared, logdir):
    """worker 进程初始化：装共享内存句柄、环境、缓存（懒建）。"""
    _set_env_single_thread()
    _W['opts'] = light.get('options') or {}
    _W['params'] = light['params']
    _W['prog'] = shared['prog']
    _W['state'] = shared['state']
    _W['logdir'] = logdir
    _W['_cache'] = None
    _W['_logf'] = None
    _W['pid'] = os.getpid()


def _wlog(msg):
    """worker 内写自己的日志文件（不与他人交错）。"""
    try:
        if _W.get('_logf') is None:
            os.makedirs(_W['logdir'], exist_ok=True)
            _W['_logf'] = open(os.path.join(
                _W['logdir'], 'worker_%d.log' % os.getpid()), 'a',
                encoding='utf-8')
        _W['_logf'].write('[%s] %s\n' % (time.strftime('%H:%M:%S'), msg))
        _W['_logf'].flush()
    except Exception:                              # noqa: BLE001
        pass


def _cache_of():
    if _W.get('_cache') is None:
        from src.core import static_atom as core
        from arc import Rubidium87
        _W['_cache'] = core.ArcCache(Rubidium87())
    return _W['_cache']


def _slot(arr, idx, width):
    """共享内存 → 该点的 numpy 视图（**真·共享**，父进程能看见引擎的写入）。

    ⚠ `mp.Array(..., lock=False)` 给的是**裸 ctypes 数组**（没有 `.get_obj()`），
    带锁时才是 SynchronizedArray —— 两种都要吃（实测踩过：
    `'c_double_Array_12' object has no attribute 'get_obj'`）。
    """
    import numpy as np
    obj = arr.get_obj() if hasattr(arr, 'get_obj') else arr
    raw = np.ctypeslib.as_array(obj)
    return raw[idx * width:(idx + 1) * width]


def _set_state(idx, value):
    try:
        _W['state'][idx] = int(value)
    except Exception:                              # noqa: BLE001
        pass


def _save_point(pdir, idx, run, res, opts):
    """该点的数组落盘（record='none' 时只留摘要）。"""
    import numpy as np
    rec = str(opts.get('record') or 'final')
    if rec == 'none':
        return None
    arr = res.get('pops')
    if arr is None:
        arr = res.get('pops_2d')
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim < 2:
        arr = arr.reshape(1, -1)
    t = np.asarray(res.get('t', []), float)
    out = {
        'labels': np.asarray([str(x) for x in _labels(run, res)]),
        'final_pop': np.asarray(arr[-1], dtype=np.float64),
        't_ns': np.asarray([float(t[0]), float(t[-1])] if len(t) else [],
                           dtype=float),
    }
    if rec == 'pops':
        want = int(opts.get('record_npts') or 200)
        # 向上取整：保证降采样后**不超过** want 点（floor 会多出 1 点，
        # 实测 101 点 / want=20 → stride=5 → 21 点，越界）
        stride = max(1, -(-len(t) // max(want, 2)))
        out['series_t'] = t[::stride]
        out['series_pop'] = np.asarray(arr[::stride], dtype=np.float32)
    path = os.path.join(pdir, 'pt_%04d.npz' % idx)
    np.savez_compressed(path, **out)
    return path


def _labels(run, res):
    from src.core import runspec
    return runspec.result_labels(run, res)


def _run_point(idx, delta):
    """**worker 主体**：一个扫描点 = 一次完整的"配置 → 引擎 → 落盘"。

    必须是模块级函数（spawn 靠名字导入）。
    """
    t0 = _now()
    _set_state(idx, 1)
    slot = None
    try:
        import numpy as np                             # noqa: F401
        from src.core import runspec
        from src.sweep import spec as _spec
        opts = _W['opts']
        slot = _slot(_W['prog'], idx, 3)
        slot[0] = 0.0
        slot[1] = 0.0
        slot[2] = 0.0

        cache = _cache_of()
        p = _spec.apply_delta(_W['params'], delta)
        run = runspec.build_run(cache, p, vdd=True,
                                diag=bool(opts.get('diag')))
        probs = runspec.check_run(run)
        fatal = [x for x in probs if '空演化' in x or '时间网格' in x]
        if fatal:
            raise ValueError('；'.join(fatal))
        res = runspec.evolve_run(run, progress_arr=slot, log=_wlog,
                                 cache=cache)
        sm = runspec.result_summary(
            run, res, top_n=int(opts.get('top_n') or 8),
            want_labels=opts.get('states'))
        npz = _save_point(_W['opts'].get('dir') or '.', idx, run, res, opts)
        rec = {
            'k': idx, 'ok': True, 'delta': delta, 'tag': _tag_of(delta),
            'elapsed_s': round(_now() - t0, 3),
            'two_body': bool(run['two_body']), 'method': run['method'],
            'basis_n': len(run['states']),
            't_max_ns': run['t_max'], 'npts': run['npts'],
            'summary': sm, 'npz': npz,
            'warnings': probs,
            'vdd_nnz': (None if run.get('vdd_matrix') is None
                        else int(run['vdd_matrix'].nnz)),
        }
        _set_state(idx, 2)
        _wlog('pt%04d done in %.1fs' % (idx, rec['elapsed_s']))
        return rec
    except BaseException as e:                     # noqa: BLE001
        cancelled = (slot is not None and float(slot[1]) != 0.0)
        tb = traceback.format_exc()
        _set_state(idx, 4 if cancelled else 3)
        _wlog('pt%04d %s: %s' % (idx, 'cancelled' if cancelled else 'FAILED',
                                 str(e)[:300]))
        return {
            'k': idx, 'ok': False, 'delta': delta, 'tag': _tag_of(delta),
            'cancelled': bool(cancelled),
            'error': ('用户取消' if cancelled else str(e)[:400]),
            'traceback': tb[-1500:],
            'elapsed_s': round(_now() - t0, 3),
        }


# =====================================================================
# 父进程（runner 主体）
# =====================================================================
def _state_frac(prog, idx):
    """共享内存 3 槽 → 单点显示进度 0..1。

    阶段语义与引擎一致：0 准备/JIT、1 积分、2 收尾。
    映射成 0..0.05 / 0.05..0.95 / 0.95..1，使进度条单调好看。
    """
    f = float(prog[idx * 3 + 0] or 0.0)
    st = float(prog[idx * 3 + 2] or 0.0)
    f = min(max(f, 0.0), 1.0)
    if st <= 0.0:
        return 0.05 * f
    if st < 2.0:
        return 0.05 + 0.90 * f
    return 0.95 + 0.05 * f


def _status_obj(sweep_id, spec, npoints, states, prog, t0, state, note='',
                cancel=False):
    pts = []
    for i in range(npoints):
        st = int(states[i]) if i < len(states) else 0
        pts.append({
            'k': i,
            'state': ('queued', 'running', 'done', 'failed', 'cancelled')[st]
            if 0 <= st <= 4 else 'queued',
            'frac': round(_state_frac(prog, i), 4),
            'tag': _tag_of((spec['deltas'][i] if i < len(spec['deltas'])
                            else {})),
        })
    n_done = sum(1 for p in pts if p['state'] == 'done')
    n_fail = sum(1 for p in pts if p['state'] == 'failed')
    return {
        'sweep_id': sweep_id, 'tag': spec.get('tag') or '',
        'state': state, 'note': note,
        'n_points': npoints, 'n_done': n_done, 'n_failed': n_fail,
        'workers': int(spec.get('_workers') or 1),
        'workers_why': spec.get('_workers_why') or '',
        'started': t0, 'elapsed_s': round(_now() - t0, 2),
        'cancel_requested': bool(cancel),
        'dir': spec.get('dir'),
        'points': pts,
    }


def _status_try(sweep_id, spec, n, state, prog, t0, st, status_path, log,
                cancel=False, warned=None, note=''):
    """写一次 status.json；失败只记一次日志，**绝不中断扫描**。"""
    try:
        _write_json(status_path, _status_obj(sweep_id, spec, n, state, prog,
                                             t0, st, cancel=cancel,
                                             note=note))
    except Exception as e:                         # noqa: BLE001
        if warned is None or not warned.get('done'):
            log('状态写盘失败（扫描继续，进度可能暂时不刷新）：%s' % e)
            if warned is not None:
                warned['done'] = True


def run_spec(spec, spec_path=None, log=print):
    """执行一份扫描规格（阻塞直到全部结束）。返回退出码。"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    sweep_id = spec['sweep_id']
    sdir = spec['dir']
    deltas = spec['deltas']
    n = len(deltas)
    status_path = os.path.join(sdir, 'status.json')
    cancel_path = os.path.join(sdir, 'cancel')
    pdir = os.path.join(sdir, 'points')
    os.makedirs(pdir, exist_ok=True)
    logdir = os.path.join(sdir, 'logs')
    os.makedirs(logdir, exist_ok=True)

    workers = int(spec.get('workers') or 0)
    workers_why = (spec.get('plan') or {}).get('why') or '显式指定'
    t0 = _now()

    # ---- 预热 + 预检（在父进程里做，避免 worker 之间抢 ARC 建表）----
    try:
        from src.core import static_atom as core
        from arc import Rubidium87
        from src.core import runspec
        from src.sweep import spec as _spec
        cache = core.ArcCache(Rubidium87())
        p0 = _spec.apply_delta(spec['params'], deltas[0])
        run0 = runspec.build_run(cache, p0, vdd=False)
        probs0 = runspec.check_run(run0)
        log('预热完成：基组 %d 态，method=%s，%s'
            % (len(run0['states']), run0['method'],
               '；'.join(probs0) if probs0 else '无告警'))
    except BaseException as e:                     # noqa: BLE001
        _write_json(status_path, {
            'sweep_id': sweep_id, 'state': 'failed', 'n_points': n,
            'n_done': 0, 'n_failed': 0, 'workers': workers or None,
            'started': t0, 'elapsed_s': round(_now() - t0, 2),
            'note': '预检失败（规格或基组有问题）：%s' % e,
            'error': traceback.format_exc()[-1500:],
            'dir': sdir, 'points': []})
        log('预检失败：%s' % e)
        return 3

    # ---- 并行度：未定档就在这里定（此时才知道**真实** N 与是否两体）----
    if not workers:
        try:
            from src.sweep.registry import plan_workers
            plan = plan_workers(basis_n=len(run0['states']),
                                two_body=bool(run0['two_body']),
                                n_points=n)
            workers = int(plan['workers'])
            workers_why = plan['why']
        except BaseException as e:                 # noqa: BLE001
            workers = max(1, min((os.cpu_count() or 4) - 1, n))
            workers_why = '自动定档失败（%s）→ 退化为 cpu-1' % e
        log('并行度定档：%d 进程 —— %s' % (workers, workers_why))
    workers = max(1, min(int(workers), n))
    spec['_workers'] = workers
    spec['_workers_why'] = workers_why

    ctx = mp.get_context('spawn')
    prog = ctx.Array('d', 3 * n, lock=False)
    state = ctx.Array('i', n, lock=False)
    shared = {'prog': prog, 'state': state}
    light = {'params': spec['params'], 'options': spec.get('options') or {}}
    light['options'] = dict(light['options'])
    light['options']['dir'] = sdir

    rc = 0
    cancelled = False
    _warned = {}                                   # 状态写盘失败只报一次
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(light, shared, logdir))
    try:
        futs = {ex.submit(_run_point, i, d): i for i, d in enumerate(deltas)}
        pending = set(futs)
        _status_try(sweep_id, spec, n, state, prog, t0, 'running',
                    status_path, log, warned=_warned)
        while pending:
            got, pending = wait(pending, timeout=1.0,
                                return_when=FIRST_COMPLETED)
            for f in got:
                i = futs[f]
                try:
                    rec = f.result()
                except BaseException as e:          # noqa: BLE001
                    rec = {'k': i, 'ok': False, 'error': str(e)[:400],
                           'delta': deltas[i], 'tag': _tag_of(deltas[i])}
                if not rec.get('ok'):
                    rc = 2 if not rec.get('cancelled') else rc
                try:
                    _write_json(os.path.join(pdir, 'pt_%04d.json' % i), rec)
                except Exception as e:              # noqa: BLE001
                    log('写 point json 失败 pt%04d: %s' % (i, e))
            # 取消：写文件是**优雅**路径（worker 自己在安全点退出）
            if not cancelled and os.path.exists(cancel_path):
                cancelled = True
                log('收到取消请求 → 通知 %d 个 worker 在安全点退出' % n)
                for i in range(n):
                    try:
                        prog[3 * i + 1] = 1.0
                    except Exception:               # noqa: BLE001
                        pass
                try:
                    state_list = [i for i in range(n) if state[i] == 0]
                    for i in state_list:
                        state[i] = 4               # 未起跑的标 cancelled
                except Exception:                   # noqa: BLE001
                    pass
            # 进度写盘是**尽力而为**：写不进去也绝不能让扫描死掉
            # （2026-09-15 实测：status.json 被读时的 WinError 5 曾杀死整轮）
            _status_try(sweep_id, spec, n, state, prog, t0,
                        'cancelling' if (cancelled and pending) else 'running',
                        status_path, log, cancel=cancelled, warned=_warned)
    finally:
        try:
            ex.shutdown(wait=not cancelled, cancel_futures=bool(cancelled))
        except TypeError:                           # 老 Python 无 cancel_futures
            ex.shutdown(wait=False)
        except Exception:                           # noqa: BLE001
            pass

    n_done = sum(1 for i in range(n) if state[i] == 2)
    n_fail = sum(1 for i in range(n) if state[i] == 3)
    final = 'cancelled' if cancelled else ('failed' if n_done == 0 and n_fail
                                           else 'done')
    _status_try(sweep_id, spec, n, state, prog, t0, final, status_path, log,
                cancel=cancelled,
                note=('已取消：完成 %d/%d' % (n_done, n) if cancelled else
                      ('%d/%d 点成功%s'
                       % (n_done, n,
                          '' if not n_fail else '，%d 点失败' % n_fail))))
    log('结束：%s（完成 %d/%d，失败 %d，用时 %.1fs）'
        % (final, n_done, n, n_fail, _now() - t0))
    return rc


# =====================================================================
# 入口
# =====================================================================
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    spec_path = None
    for i, a in enumerate(argv):
        if a == '--spec' and i + 1 < len(argv):
            spec_path = argv[i + 1]
    if not spec_path:
        print('用法: python scripts/sweep_runner.py --spec <spec.json>')
        return 2
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    def log(msg):
        line = '[sweep %s %s] %s' % (spec.get('sweep_id', '?'),
                                     time.strftime('%H:%M:%S'), msg)
        print(line, flush=True)
        try:
            with open(os.path.join(spec['dir'], 'runner.log'), 'a',
                      encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:                          # noqa: BLE001
            pass

    try:
        return run_spec(spec, spec_path=spec_path, log=log)
    except BaseException as e:                     # noqa: BLE001
        log('runner 崩溃：%s' % e)
        try:
            _write_json(os.path.join(spec['dir'], 'status.json'), {
                'sweep_id': spec.get('sweep_id'), 'state': 'failed',
                'note': 'runner 崩溃：%s' % e,
                'error': traceback.format_exc()[-1500:],
                'dir': spec.get('dir'), 'points': []})
        except Exception:                          # noqa: BLE001
            pass
        return 4


if __name__ == '__main__':
    sys.exit(main())
