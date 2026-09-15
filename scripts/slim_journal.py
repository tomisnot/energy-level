# -*- coding: utf-8 -*-
"""slim_journal.py —— 把历史 `runtime/_journal/*.jsonl` 的巨型 detail 迁成摘要（一次性维护工具）。

背景（2026-09-14 用户裁决"巨型 detail 瘦身 + 修写入侧"）：
  2026-09-10 的日志 37 MB，几乎全是 `set_param('dynamic.waves_A')` 的
  detail.old/new 带着**整条波形包络**（实测单条 748 KB）。
  写入侧已在 `src/ai/journal.py:Op` 加闸（`_MAX_DETAIL_BYTES`），本脚本负责
  **把闸之前写下的老文件**按同一把闸重写。

纪律：读 → 断言 → 原子写（临时文件 + os.replace）。任何一条不通过就**不写**。
原始文件内容在 git 历史里（`git show <commit>:runtime/_journal/2026-09-10.jsonl`），
另有一份仓库外打包（见报告）。

用法：
  python scripts/slim_journal.py            # dry-run：只打印前后体积
  python scripts/slim_journal.py --apply    # 真正重写（逐文件原子替换）
  python scripts/slim_journal.py --min-kb 64 --apply
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.ai.journal import slim_detail, _json_bytes        # noqa: E402

_KEEP = ('seq', 'ts', 'actor', 'kind', 'reason')


def _slim_file(path, min_bytes, apply_it):
    with io.open(path, encoding='utf-8') as f:
        raw = f.readlines()
    out, n_changed, worst = [], 0, 0
    before = after = 0
    for i, line in enumerate(raw):
        before += len(line.encode('utf-8'))
        line = line.rstrip('\n')
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception as e:                             # noqa: BLE001
            raise SystemExit('!! %s 第 %d 行不是合法 JSON：%s' % (path, i + 1, e))
        det, meta = slim_detail(rec.get('detail'))
        if meta:
            n_changed += 1
            rec['detail'] = det
            rec['detail_slim'] = meta
            worst = max(worst, meta['orig_bytes'])
        new = json.dumps(rec, ensure_ascii=False)
        after += len(new.encode('utf-8'))
        out.append(new)
    # ---- 断言：行数/身份字段逐条不变 ----
    if len(out) != len([x for x in raw if x.strip()]):
        raise SystemExit('!! %s 行数不一致，拒绝写入' % path)
    for i, new in enumerate(out):
        rec = json.loads(new)
        old = json.loads(raw[i])
        for k in _KEEP:
            if rec.get(k) != old.get(k):
                raise SystemExit('!! %s 第 %d 行 %s 被改动，拒绝写入'
                                 % (path, i + 1, k))
    if apply_it and n_changed:
        tmp = path + '.tmp'
        with io.open(tmp, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(out) + '\n')
        # 复核临时文件：行数与体积
        with io.open(tmp, encoding='utf-8') as f:
            chk = [json.loads(x) for x in f if x.strip()]
        if len(chk) != len(out):
            os.remove(tmp)
            raise SystemExit('!! 临时文件行数不对，已放弃')
        os.replace(tmp, path)
    return before, after, n_changed, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真正重写（默认 dry-run）')
    ap.add_argument('--min-kb', type=float, default=64.0,
                    help='只处理大于该体积的文件（KB）')
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, 'runtime', '_journal')
    tot_b = tot_a = 0
    print('%-26s %10s %10s %8s %10s' % ('file', 'before', 'after', 'slimmed',
                                        'worst_detail'))
    for name in sorted(os.listdir(d)):
        if not name.endswith('.jsonl'):
            continue
        p = os.path.join(d, name)
        if os.path.getsize(p) < a.min_kb * 1024:
            continue
        b, aft, n, worst = _slim_file(p, a.min_kb * 1024, a.apply)
        tot_b += b
        tot_a += aft
        print('%-26s %9.2fM %9.2fM %8d %9.1fK' % (
            name, b / 1e6, aft / 1e6, n, worst / 1024))
    print('%-26s %9.2fM %9.2fM   %s' % ('TOTAL', tot_b / 1e6, tot_a / 1e6,
                                        '已写入' if a.apply else 'dry-run'))


if __name__ == '__main__':
    main()
