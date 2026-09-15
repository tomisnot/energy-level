# -*- coding: utf-8 -*-
"""run_store 自检：CodeAct 运行记录的落盘 / 取回 / 保留策略。

计数模型（2026-09-14 改为**不变式包**，见 checklib）：默认摘要只数包（项数），
包内子条件单独累计为 `SUB`（判据表达式原样搬进包内，覆盖面不变）。

用途: python -u checks/run_store_selfcheck.py

为什么单独有这个自检：run_store 是纯 stdlib 模块（**有意不依赖 Qt/GUI**），
因为它的核心约束是"长任务执行期间也能读"——run_python 的脚本占用 Qt
主线程，任何经 run_on_main 的查询都会被排队，所以状态查询必须只走文件
I/O。这条边界值得被测试钉住：一旦有人往里加 Qt 依赖，这个自检会立刻
在无 GUI 环境下失败。
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checks.checklib import Checker
from src.ai import run_store

C = Checker('run_store_selfcheck')
root = tempfile.mkdtemp(prefix='el_runstore_')
try:
    d = run_store.results_dir(root)
    _p1 = [('results_dir = <root>/runtime/ai_results',
            d == os.path.join(root, 'runtime', 'ai_results'))]

    # ---------- run_id ----------
    rid = run_store.new_run_id()
    _p1.append(('run_id 前缀为 run_', rid.startswith('run_')))
    same_sec = [run_store.new_run_id() for _ in range(5)]
    _p1.append((f'同一秒内 5 次调用 run_id 互不相同（{same_sec}）',
                len(set(same_sec)) == 5))
    _p1.append(('run_id 定宽（字典序=时序）',
                len(rid) == len(same_sec[0]) == len(run_store.new_run_id())))
    C.group('路径与 run_id（results_dir / 前缀 / 唯一 / 定宽）', _p1)

    # ---------- start ----------
    code = 'x = 1\ny = 2\n'
    rec = run_store.start(root, rid, code)
    path = run_store.run_path(root, rid)
    _p2 = [
        ('start 落盘', os.path.exists(path)),
        ('start 为原子写（无 .tmp 残留）', not os.path.exists(path + '.tmp')),
        (f'start status=running（{rec.get("status")}）',
         rec.get('status') == 'running'),
        (f'start 记 lines（{rec.get("lines")}）', rec.get('lines') == 2),
        ('start 记 code_head', 'x = 1' in (rec.get('code_head') or '')),
    ]
    t_first = rec['started']

    # start 幂等（run_python_async 会先登记一次，run_code 再登记一次）
    time.sleep(0.05)
    rec2 = run_store.start(root, rid, code)
    _p2.append((f'start 幂等：保留首次 started'
                f'（{rec2["started"]} vs {t_first}）',
                abs(rec2['started'] - t_first) < 1e-9))
    C.group('start 落盘与幂等（原子写 / status / lines / code_head / first-wins）',
            _p2)

    # ---------- finish(ok) ----------
    run_store.finish(root, rid, {'ok': True, 'result': {'a': 1, 'b': [1, 2]},
                                 'stdout': 'hi\n', 'elapsed_s': 1.23}, code=code)
    r = run_store.read(root, rid)
    C.group('finish(ok)：全字段落盘（status/result/stdout/code/elapsed/无 .tmp）', [
        (f'finish → status=ok（{r["status"]}）', r['status'] == 'ok'),
        (f'finish 存 result（{r.get("result")}）',
         r.get('result') == {'a': 1, 'b': [1, 2]}),
        ('finish 存 stdout', r.get('stdout') == 'hi\n'),
        ('finish 存完整代码', r.get('code') == code),
        ('finish 记 elapsed_s', r.get('elapsed_s') == 1.23),
        ('finish ok 标志', r.get('ok') is True),
        ('finish 后无 .tmp 残留', not os.path.exists(path + '.tmp')),
    ])

    # ---------- finish(error)：这正是 2026-09-10 丢结果的那种情形 ----------
    rid2 = run_store.new_run_id()
    run_store.start(root, rid2, 'boom')
    run_store.finish(root, rid2, {'ok': False, 'stdout': '',
                                  'error': 'NameError: name os is not defined'})
    r2 = run_store.read(root, rid2)
    C.group('finish(error)：失败记录保留（status/error 全文/无 result/可取回）', [
        (f'失败记录 status=error（{r2["status"]}）', r2['status'] == 'error'),
        ('失败记录保留 error 全文', 'NameError' in (r2.get('error') or '')),
        ('失败记录不写 result 字段', 'result' not in r2),
        ('失败记录仍可取回（不再算完就丢）', r2.get('run_id') == rid2),
    ])

    # ---------- read 边界 ----------
    C.group('read 边界（未知 id / 空 id / 路径穿越 ../ 与 /）', [
        ('未知 run_id → None', run_store.read(root, 'run_不存在') is None),
        ('空 run_id → None', run_store.read(root, '') is None),
        ('路径穿越被拦（../）', run_store.read(root, '../secret') is None),
        ('路径穿越被拦（/）', run_store.read(root, 'a/b') is None),
    ])

    # ---------- list_runs ----------
    lst = run_store.list_runs(root, 10)
    C.group('list_runs（计数 / 最新在前 / 瘦身 / has_result / error_head / limit / 容错）', [
        (f'list_runs 计数（{len(lst)}）', len(lst) == 2),
        (f'list_runs 最新在前（{lst[0]["run_id"]}）', lst[0]['run_id'] == rid2),
        (f'list_runs 不夹带 result/stdout（防上下文爆量）（{sorted(lst[0])}）',
         'result' not in lst[0] and 'stdout' not in lst[0]),
        ('list_runs 标 has_result', lst[1]['has_result'] is True),
        ('list_runs 给 error_head',
         'NameError' in ([x for x in lst if x['run_id'] == rid2][0]['error_head']
                         or '')),
        ('list_runs limit 生效', len(run_store.list_runs(root, 1)) == 1),
        ('runs 目录不存在 → 空列表',
         run_store.list_runs(os.path.join(root, 'nope'), 5) == []),
    ])

    # ---------- prune 保留策略 ----------
    for i in range(5):
        ri = run_store.new_run_id() + f'_p{i}'
        run_store.start(root, ri, 'pass')
    n_before = len([f for f in os.listdir(d) if f.endswith('.json')])
    killed = run_store.prune(root, keep=3)
    n_after = len([f for f in os.listdir(d) if f.endswith('.json')])
    kept = [x['run_id'] for x in run_store.list_runs(root, 3)]
    C.group('prune 保留策略（收到 keep 条 / 返回删除数 / 留最新 / keep<=0 不动手）', [
        (f'prune 收到 keep 条（{n_before} -> {n_after}）', n_after == 3),
        (f'prune 返回删除数（{killed} vs {n_before - 3}）',
         killed == n_before - 3),
        (f'prune 保留的是最新 3 条（{kept}）',
         all(x.endswith(('_p2', '_p3', '_p4')) for x in kept)),
        ('prune keep<=0 不动手', run_store.prune(root, keep=0) == 0),
    ])

    # ---------- 损坏文件不应让整表崩 ----------
    bad = os.path.join(d, 'run_99999999_999999.json')
    with open(bad, 'w', encoding='utf-8') as f:
        f.write('{ not json at all')
    C.group('损坏文件容错（read → None / list 仍返回）', [
        ('损坏文件 read → None',
         run_store.read(root, 'run_99999999_999999') is None),
        ('损坏文件不致命（list 仍返回）',
         isinstance(run_store.list_runs(root, 10), list)),
    ])
finally:
    shutil.rmtree(root, ignore_errors=True)

sys.exit(0 if C.summary() else 1)
