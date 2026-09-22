# -*- coding: utf-8 -*-
"""驾驶舱只读后端：直接读事件总线的 journal 文件（**不依赖项目进程存活**）。

纲领 T5「黑匣子」：飞行记录仪必须物理独立于机身——你最需要驾驶舱的时刻，恰是
项目崩溃/失控那一刻。所以本模块**只读磁盘上的 append-only journal**（项目进程
死了文件还在），而不是去连项目的活进程。这样"杀掉项目后仍显示死前最后状态 +
完整事件史"是**构造上成立**的，不是靠运气。

**只读**：本模块（及整个 `src/cockpit`）无任何写回项目/总线的通道——由
`cockpit_selfcheck` 的 AST 守卫致死性看住（纲领 I2：驾驶舱无自己的真值，只投影）。

告警规则（M5 注意力分级）：默认平静 → 异常升级。规则**例外驱动**，且每条都能
被"构造一次异常"点亮（致死性，防哑告警 D4）。
"""
import glob
import json
import os

__all__ = ['journal_files', 'read_events', 'last_seq', 'orientation',
           'evaluate_alerts', 'ALERT_LEVELS', 'config_tree']

# 告警级别（M5：平静 → 升级 → 深挖）
ALERT_LEVELS = ('ok', 'warn', 'error')


# =====================================================================
# journal 只读（diff-since-seq，R1 头号原语）
# =====================================================================
def journal_files(root):
    """当前 + 归档的 journal 文件（轮转归档不丢史）。"""
    return sorted(glob.glob(os.path.join(root, 'journal*.jsonl')))


def read_events(root, since_seq=0, limit=None):
    """从 journal 文件读 `seq > since_seq` 的事件（只读、幂等）。

    这是驾驶舱的**唯一数据源**：读的是写入也走的那条总线的持久化形态（T1/T2：
    两条通道一个真值源；驾驶舱读总线，不读 MCP 流量镜像）。
    """
    since_seq = int(since_seq or 0)
    out = []
    for fp in journal_files(root):
        try:
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # 半行（崩溃时）→ 跳过，不让驾驶舱哑掉
                    if int(ev.get('seq', 0)) > since_seq:
                        out.append(ev)
        except OSError:
            continue
    out.sort(key=lambda e: int(e.get('seq', 0)))
    if limit is not None:
        out = out[:int(limit)]
    return out


def last_seq(root):
    mx = 0
    for ev in read_events(root, 0):
        mx = max(mx, int(ev.get('seq', 0)))
    return mx


def orientation(root, tail_n=12):
    """中途加入的**有界定位包**（R1）：配置摘要 + 在跑什么 + 最近事件 + 告警。"""
    evs = read_events(root, 0)
    recent = evs[-max(int(tail_n), 0):]
    last_cfg = {}
    for ev in reversed(evs):
        if ev.get('kind') == 'set' and ev.get('target'):
            last_cfg.setdefault(ev['target'], ev.get('after'))
    running = [ev for ev in evs if ev.get('kind') == 'run']
    return {
        'last_seq': (int(evs[-1]['seq']) if evs else 0),
        'n_events': len(evs),
        'n_runs': len(running),
        'last_run': (running[-1].get('target') if running else None),
        'config_head': {k: last_cfg[k] for k in sorted(last_cfg)[:12]},
        'recent': recent,
        'alerts': evaluate_alerts(evs[-50:]),
    }


# =====================================================================
# 告警规则（例外驱动 M5；每条都可被构造点亮 = 致死性，防哑告警 D4）
# =====================================================================
def _run_warnings(ev):
    """从 run 事件的 provenance 取 runspec 的 warnings（"被忽略/改道"的不静默出口）。

    类型守卫：provenance/runspec 可能是**字符串**（ops 写 provenance.runspec 时放的是
    summarize_run 的摘要串而非 dict）→ 非 dict 一律视为无 warnings，不崩 /api/orientation。"""
    prov = ev.get('provenance')
    if not isinstance(prov, dict):
        return []
    rs = prov.get('runspec')
    if not isinstance(rs, dict):
        return []
    w = rs.get('warnings') or []
    return list(w)


def evaluate_alerts(events):
    """事件流 → 告警列表 `{level, rule, seq, message}`（例外驱动）。

    规则（都**能被构造点亮**，见 cockpit_selfcheck 的致死性证明）：
      degradation : run 的 provenance.runspec.warnings 非空——sambe 改道 hard、
                    运行级参数被丢、任意波形降级等"为省事/兼容性"的静默改道，
                    runspec 已把它写进 warnings（D5 不静默）；驾驶舱必须把它**升级成灯**。
      run_failed  : run 结果 after.ok == False（演化没成）。
      empty_run   : run 结果报"空演化"（既无波形也无耗散，pops 恒 0 的经典事故）。
      error_event : kind == 'error' 的事件（引擎/契约显式报错）。
    """
    alerts = []
    for ev in events or []:
        seq = ev.get('seq')
        kind = ev.get('kind')
        if kind == 'error':
            alerts.append({'level': 'error', 'rule': 'error_event', 'seq': seq,
                           'message': '错误事件：%s' % (ev.get('reason') or
                                                        ev.get('target') or '')})
            continue
        if kind != 'run':
            continue
        w = _run_warnings(ev)
        if w:
            alerts.append({'level': 'warn', 'rule': 'degradation', 'seq': seq,
                           'message': '运行有降级/改道告警 %d 条：%s'
                                      % (len(w), '；'.join(str(x) for x in w[:3]))})
        # P0：同一跃迁多条时间重叠驱动（扫描忘清）——静默给错答案的头号陷，
        # 由 ops.run 写进 provenance.overlap_warning（驾驶舱只读标志，不重算）
        ovl = (ev.get('provenance') or {}).get('overlap_warning')
        if ovl:
            alerts.append({'level': 'warn', 'rule': 'overlap_drives', 'seq': seq,
                           'message': ovl.get('warning') if isinstance(ovl, dict)
                           else str(ovl)})
        after = ev.get('after') or {}
        if isinstance(after, dict):
            if after.get('ok') is False:
                alerts.append({'level': 'error', 'rule': 'run_failed', 'seq': seq,
                               'message': '演化未成功：%s' % after.get('msg', '')})
            # 空演化：runspec.check_run 的"既无波形也无耗散"经典坑
            txt = json.dumps(after, ensure_ascii=False)
            if '空演化' in txt or 'empty' in txt.lower():
                alerts.append({'level': 'error', 'rule': 'empty_run', 'seq': seq,
                               'message': '疑似空演化（结果恒 0）——先加波形或开衰变'})
    return alerts


def top_level(alerts):
    """一批告警里的最高级别（HUD 归因灯用；无告警 → 'ok'）。"""
    lvl = 'ok'
    for a in alerts or []:
        if a.get('level') == 'error':
            return 'error'
        if a.get('level') == 'warn':
            lvl = 'warn'
    return lvl


# =====================================================================
# 配置态结构树（需求②：AI 模式的 web 配置展示）——schema join 现值
# =====================================================================
def _resolve_gated(root, val):
    """把被 I4 体积闸截断的 `$truncated` 字段读回原文（只读 payloads/<sha1>.json）。

    `$pointer`（numpy npz 载荷）不解析——那是数组，配置树只展示其摘要（shape/sha1）。
    """
    if isinstance(val, dict) and val.get('$truncated') and val.get('where'):
        try:
            with open(os.path.join(root, val['where']), encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return val.get('head')
    return val


def config_tree(root):
    """配置态**结构树**：最近一条 `registry_schema` 事件的 schema（family/type/unit/
    desc/tier/readonly…）**join** 每键从 `set` 事件重建的现值（+ 谁改的/为何/第几条）。

    纯读 journal、**不 import mecha**（黑匣子纯净 T5）；schema 大时被体积闸截成指针，
    这里顺 `where` 读回原文（不丢）。无 schema 事件 → has_schema=False（树不凭空造）；
    有值但 schema 无的键 → 归入 orphans（契约外/已移除，不静默丢）。
    """
    evs = read_events(root, 0)
    schema = None
    for ev in evs:
        if ev.get('kind') == 'registry_schema':
            resolved = _resolve_gated(root, ev.get('after'))
            if isinstance(resolved, list):
                schema = resolved          # 取最近一条
    last = {}
    for ev in evs:
        if ev.get('kind') == 'set' and ev.get('target'):
            last[ev['target']] = {
                'value': _resolve_gated(root, ev.get('after')),
                'reason': ev.get('reason'), 'actor': ev.get('actor'),
                'seq': ev.get('seq'),
            }
    fams = {}
    for rec in schema or []:
        name = rec.get('name')
        node = dict(rec)
        st = last.get(name)
        node['value'] = st['value'] if st else None
        node['set'] = ({'reason': st['reason'], 'actor': st['actor'], 'seq': st['seq']}
                       if st else None)
        fams.setdefault(rec.get('family') or 'other', []).append(node)
    known = {r.get('name') for r in (schema or [])}
    orphans = [{'name': k, 'value': v['value'],
                'set': {'reason': v['reason'], 'actor': v['actor'], 'seq': v['seq']}}
               for k, v in last.items() if k not in known]
    return {
        'has_schema': schema is not None,
        'families': {k: fams[k] for k in sorted(fams)},
        'orphans': sorted(orphans, key=lambda r: r['name']),
        'n_keys': sum(len(v) for v in fams.values()),
        'last_seq': (int(evs[-1]['seq']) if evs else 0),
    }
