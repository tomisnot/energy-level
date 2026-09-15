# -*- coding: utf-8 -*-
"""CodeAct 运行记录落盘：让 run_python 的结果「算完不丢」。

为什么需要（2026-09-10 实测教训）：
  - 一次 run_python 跑完 40 次演化（293 s）后，脚本末尾因命名空间无 os
    而抛异常，内存里的 result 与整个系综一起消失，只能靠 save_npz 的
    逐次落盘文件 + σ=0 块的可校验性侥幸恢复；
  - 同一次会话还出现客户端 fetch failed 而服务端仍在继续执行的情形，
    AI 侧既不知道它是否还在跑，也拿不到最终结果。

本模块为此提供三件事：
  1. 每次 run_python 在 `runtime/ai_results/` 下留一份 JSON 记录：开始先写
     status=running，结束覆写为 ok/error（原子替换，不留半个文件）；
  2. run_id → 文件名的确定性映射，状态查询**只读文件、不经 Qt 主线程**
     ——这点是硬约束：run_python 的脚本在主线程 exec，长任务期间任何走
     run_on_main 的调用都会排队，所以轮询必须避开主线程；
  3. 保留最近 KEEP 份，自动清理最旧的（与 save_state/_autosnap 同思路）。

有意不做的：**不给 CodeAct 命名空间加 os/sys**。那是刻意的设计边界
（见 codeact 模块 docstring：状态变更必须全部经 ParamRegistry → 控件同路
→ op 归因）。本模块只用 stdlib，且由宿主（run_code）负责调用，模型侧
既不需要也无法直接触碰它。
"""
import json
import os
import time

from .paths import RESULTS_DIRNAME     # 相对项目根（唯一真值源）
KEEP = 200                       # 保留最近多少份运行记录
_CODE_HEAD = 1200                # 列表/概览里保留的代码头部长度
_CODE_FULL = 200000              # 完整代码落盘上限（防超大脚本撑爆记录）

_SEQ = [0]                       # 进程内单调序号：保证同秒内多次调用不撞名


def results_dir(root):
    """运行记录目录（不保证存在）。"""
    return os.path.join(root, RESULTS_DIRNAME)


def new_run_id():
    """秒级时间戳 + 进程内单调序号。

    名字定宽，故字典序即时序（list_runs 直接按名倒排即最新在前）。
    用序号而非毫秒取模，保证同一秒内连续两次调用也唯一。
    """
    _SEQ[0] += 1
    return 'run_%s_%06d' % (time.strftime('%Y%m%d_%H%M%S'), _SEQ[0])


def run_path(root, run_id):
    return os.path.join(results_dir(root), f'{run_id}.json')


def _write(path, rec):
    """原子写：先写 .tmp 再 os.replace，避免读到半个 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def start(root, run_id, code, meta=None):
    """登记一次运行为 running。返回记录 dict；失败返回 None（绝不抛出）。

    落盘失败不应影响脚本执行——记录是旁路，不是主路。
    """
    rec = {
        'run_id': run_id,
        'kind': 'codeact',
        'status': 'running',
        'started': time.time(),
        'started_iso': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pid': os.getpid(),
        'lines': len((code or '').splitlines()),
        'code_head': (code or '')[:_CODE_HEAD],
    }
    # 幂等：run_python_async 会先登记一次以消除竞态窗口，随后 run_code 再
    # 登记一次；保留首次的 started，否则耗时统计会偏小。
    prev = read(root, run_id)
    if prev and prev.get('started'):
        rec['started'] = prev['started']
        rec['started_iso'] = prev.get('started_iso', rec['started_iso'])
    if meta:
        rec.update(meta)
    try:
        _write(run_path(root, run_id), rec)
    except OSError:
        return None
    return rec


def finish(root, run_id, out, code=None):
    """把 out（run_code 的返回值）覆写进记录并收尾。绝不抛出。"""
    rec = read(root, run_id) or {'run_id': run_id, 'kind': 'codeact'}
    ok = bool((out or {}).get('ok'))
    rec['status'] = 'ok' if ok else 'error'
    rec['finished'] = time.time()
    rec['elapsed_s'] = (out or {}).get('elapsed_s')
    rec['ok'] = ok
    rec['stdout'] = (out or {}).get('stdout', '')
    if ok:
        rec['result'] = (out or {}).get('result')
    else:
        rec['error'] = (out or {}).get('error', '')
    if code is not None:
        rec['code'] = code[:_CODE_FULL]
    try:
        _write(run_path(root, run_id), rec)
    except OSError:
        return rec
    prune(root)
    return rec


def read(root, run_id):
    """读一条记录；不存在或损坏返回 None。"""
    if not run_id or os.sep in str(run_id) or '/' in str(run_id):
        return None                     # 防路径穿越
    p = run_path(root, run_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def prune(root, keep=KEEP):
    """只保留最近 keep 份（按文件名时序）。返回删除数；绝不抛出。"""
    d = results_dir(root)
    if keep <= 0 or not os.path.isdir(d):
        return 0
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith('.json'))
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


def list_runs(root, limit=20):
    """最近 limit 条记录的概览（不含 result/stdout，避免上下文爆量）。"""
    d = results_dir(root)
    if not os.path.isdir(d):
        return []
    try:
        names = sorted((n for n in os.listdir(d) if n.endswith('.json')),
                       reverse=True)
    except OSError:
        return []
    now = time.time()
    out = []
    for nm in names[:max(1, int(limit))]:
        rec = read(root, nm[:-5])
        if rec is None:
            continue
        running = rec.get('status') == 'running'
        out.append({
            'run_id': rec.get('run_id'),
            'status': rec.get('status'),
            'started_iso': rec.get('started_iso'),
            'elapsed_s': rec.get('elapsed_s'),
            # running 时 started 起算的实时墙钟；进程若中途崩溃会一直涨，
            # 因此它同时也是「可能有陈旧记录」的信号
            'live_s': round(now - rec['started'], 1)
            if running and rec.get('started') else None,
            'lines': rec.get('lines'),
            'ok': rec.get('ok'),
            'has_result': 'result' in rec,
            'error_head': (rec.get('error') or '')[-200:] or None,
        })
    return out
