# -*- coding: utf-8 -*-
"""dsh 面板**共享资产**守卫（EL 侧）：常量指纹 + 共享单测真跑。

## 它守什么

`dsh/src/dsh-panel/` 是 mecha 参考实现 `mecha/dsh-panel/`（提交态 **58e2f8a**）的
**逐字副本**——按该目录 `README.md` 的复制约定：**只有参数块 `panel-config.ts` 允许
项目化**（EL 的路由 / 端口文件名 / 标题），**其余文件必须与参考实现逐字一致**。

本判据把**行为载体**（代码 + 单测）的**行尾归一 sha256** 钉成常量：

* **优先「钉常量指纹」，运行时零跨仓读取**（资产 README 的裁决）：上游任何在飞改动都
  不会红本判据；上游真发新版时**整份重新复制 + 同步更新常量**，而不是就地打补丁。
  `REFERENCE_REVISION` 只作**出处说明**。
* ⚠ 比的是**归一化行尾后的正文**：`core.autocrlf` 会让"同一内容"字节不同（实测过）
  ⇒ `\\r\\n` 归一成 `\\n` 再取指纹；裸字节比对会被行尾配置打败。
* ⚠ **参数块与 README.md 不进表**：前者必然不同（它就是参数块），后者是 prose、
  就地批注不改行为（钉它只制造噪声）。
* ⚠ 本判据**按内容读文件、不经 git**：本工程两次教训——gitignore 的文件逃出
  `git diff` 验收；目录 junction 让"两条路径"其实是同一个文件。

## 三组断言

- **A 指纹**：逐行为载体存在 + 归一后 sha256 == 常量（不符即"漂移"）。
- **B 致死性（R7）**：把合成内容喂给**同一个**探测函数——改一个字符 / 少一个文件 /
  干净内容——前两者必须报、后者必须零问题（否则它是哑守卫）。
- **C 共享单测真跑**：`node --test` 跑表里的共享单测，解析 **tests/pass/fail/skipped**
  四个计数，要求 **`pass > 0` 且 `skipped == 0`**（R16 实例：本工程有一次判据**漏了
  `skipped`**，把"全 skip"判成 PASS ⇒ "全被跳过"是红，不是绿）。

## 非退化自证（R8）

打印并断言：钉了几个载体、哈希了多少字节、共享单测跑出几个用例。否则"表是空的"
也会"看起来通过"。

用法：python checks\\dsh_panel_asset_selfcheck.py     退出码 0 全过 / 1 有问题。
"""
import hashlib
import io
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402

C = Checker('dsh_panel_asset')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join('dsh', 'src', 'dsh-panel')

#: **出处说明**（只作说明，不参与比对——比对靠下面那组常量）。
REFERENCE_REVISION = 'mecha/dsh-panel @ 58e2f8a（2026-09-25 逐字复制）'

#: **行为载体**（代码 + 单测）→ 行尾归一 sha256。改这里 = "上游发新版后整份重新复制"。
CARRIERS = {
    'monitor-url.ts':
        'c6a948106a480c67b76447d9e160f29519637a1ae5e3716a824552497efcf1ba',
    'monitor-url.test.ts':
        '62994cf218fc195890a7dc966f5159ee62aaf926a25a032d1a7f70bdf6d99c07',
    'monitor-client.ts':
        'eec71f3f7e47cdbbafdece1c9649bd68f3a86e6631b486e9e8e3d18cac9260c9',
    'monitor-client.test.ts':
        '754dee85d2a329a5da3c1425368e3c46595ac8605f4df88a0acc6bdf95f03616',
}


def _norm_sha(b):
    """行尾归一（CRLF→LF）后取 sha256（byte-exact 会被 core.autocrlf 打败）。"""
    return hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest()


def audit(table, read_fn):
    """纯函数：→ (问题列表, 统计)。喂合成 read_fn 即可证明它抓得住（R7）。"""
    problems, total_bytes = [], 0
    for name, want in sorted(table.items()):
        try:
            b = read_fn(name)
        except (OSError, KeyError) as e:      # 读不到（缺文件/路径不对）⇒ 记问题，不炸
            problems.append('%s：读不到（%s）' % (name, e.__class__.__name__))
            continue
        total_bytes += len(b)
        got = _norm_sha(b)
        if got != want:
            problems.append('%s：指纹不符（漂移）期望 %s… 实得 %s…'
                            % (name, want[:12], got[:12]))
    return problems, {'载体数': len(table), '哈希字节': total_bytes}


def _read_asset(name):
    with io.open(os.path.join(ROOT, ASSET_DIR, name), 'rb') as f:
        return f.read()


def _run_shared_tests(names):
    """`node --test` 跑共享单测 → (计数 dict, 原始输出尾部)。

    ⚠ node 缺失/调用失败 ⇒ 返回空计数（调用方判红）：**"无法判定" ≠ "判定通过"**。
    """
    files = [os.path.join(ROOT, ASSET_DIR, n) for n in names if n.endswith('.test.ts')]
    if not files:
        return {}, '（表里没有共享单测）'
    try:
        p = subprocess.run(['node', '--test'] + files, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=180,
                           cwd=os.path.join(ROOT, 'dsh'))
    except (OSError, subprocess.SubprocessError) as e:
        return {}, '（node 调用失败：%s）' % e.__class__.__name__
    out = (p.stdout or '') + (p.stderr or '')
    counts = {}
    for m in re.finditer(r'^\s*(?:#|\u2139)\s*(tests|pass|fail|skipped|cancelled)\s+(\d+)',
                         out, re.M):
        counts[m.group(1)] = int(m.group(2))
    return counts, '\n'.join(out.splitlines()[-6:])


def main():
    problems, stats = audit(CARRIERS, _read_asset)
    print('[dsh_panel_asset] 出处：%s' % REFERENCE_REVISION)
    C.group('A 行为载体与参考实现逐字一致（行尾归一 sha256 == 常量；按内容读、不经 git）',
            [('无漂移（首个问题：%s）' % (problems[0][:90] if problems else '共 0 条'),
              not problems)])

    # ---- B 致死性：合成内容喂给同一个 audit（R7）----
    good = b'export const x = 1\n'
    want = _norm_sha(good)
    cases = [
        ('干净内容 → 零问题', {'a.ts': want}, {'a.ts': good}, True),
        ('改一个字符 → 报出', {'a.ts': want}, {'a.ts': good + b' '}, False),
        ('删掉一个字符 → 报出', {'a.ts': want}, {'a.ts': b'export const x = '}, False),
        ('文件读不到 → 报出', {'missing.ts': want}, {}, False),
    ]
    subs = []
    for label, table, files, want_clean in cases:
        got, _st = audit(table, lambda n, f=files: f[n])
        subs.append(('%s（问题 %d 条）' % (label, len(got)), (not got) == want_clean))
    C.group('B 致死性：改一个字符 / 少一个文件必须报（非哑守卫）', subs)

    # ---- C 共享单测真跑（skipped 必须为 0）----
    counts, tail = _run_shared_tests(list(CARRIERS))
    stats['共享单测'] = counts
    C.group('C 共享单测真跑：pass>0 且 skipped==0（"全被跳过"是红，不是绿）', [
        ('node --test → %s' % (counts or '（没解析到计数）'),
         bool(counts) and counts.get('pass', 0) > 0 and counts.get('skipped', 0) == 0
         and counts.get('fail', 0) == 0),
        ('R8 自证：跑出的用例数 > 0（%d）' % counts.get('tests', 0),
         counts.get('tests', 0) > 0),
    ])
    if counts.get('fail') or counts.get('skipped') or not counts:
        print('[dsh_panel_asset] node 输出尾部：\n%s' % tail)

    print('[dsh_panel_asset] 非退化自证：载体 %d 个 / 哈希 %d 字节 / %s'
          % (stats['载体数'], stats['哈希字节'], stats['共享单测']))
    C.group('R8 非退化自证：钉了载体、真哈希了字节', [
        ('载体 %d 个 > 0、哈希字节 %d > 0' % (stats['载体数'], stats['哈希字节']),
         stats['载体数'] > 0 and stats['哈希字节'] > 0),
    ])

    ok = C.summary()
    for p in problems:
        print('   - %s' % p)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
