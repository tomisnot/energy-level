# -*- coding: utf-8 -*-
"""个人路径守卫：**被 git 跟踪**的文件里出现"个人路径特征" ⇒ 红（公开仓不泄漏本机路径）。

形状照抄 `mecha/checks/personal_paths_selfcheck.py`（那边把债清到 0 了），落到 EL 的
`Checker` 惯例上。三条纪律是它的核心，逐条保留：

1. ⚠ **判定必须在 Python 里做**（`subprocess` + `read_text`，路径不经 shell 引号）：
   PowerShell 的 `git ls-files` 会给**非 ASCII 路径加引号转义** ⇒ `Select-String` 打不开
   那些文件 ⇒ 扫描结果假报 0，而"**没命中**"与"**没读到**"长得一模一样。
   ⇒ 所以本判据**必须 `-z` + `-c core.quotePath=false`**。
2. ⚠ **读不到必须响亮 + 自证读到几个**：单列 `unreadable`（不许静默 `continue`），
   并打印"**实际读到 N 个**"（"列了 188 个"≠"读了 188 个"——`mecha` 那版第一版就瞎在这）。
3. ⚠ **特征拆开拼 + 自排除（双保险）**：
   * 拆开拼 ⇒ 本文件里**不出现**任何完整特征字面量（否则守卫自己泄漏）；
   * 再把本文件与免除表**显式跳过**（只靠拆开拼，后人读不出"为什么这么写"；
     只靠自排除，任何人读这个文件都会看到一串本机路径示例——那本身就是泄漏）。

## 免除表（**缩小的债**，不是永久豁免）

`checks/personal_paths_exempt.txt`：`<仓库内路径> <允许处数>`（**只写仓库内路径**）。
命中数**超过**允许 ⇒ 红；**未登记**的文件有命中 ⇒ 红。本仓当前**免除表为空文件/不存在**
（债已清零：31 处全部真改）。判据会**打印**"可收紧"的条目，但不因它红（债可见 ≠ 债必还）。

## Known Limitations and Deferred Work

- 只扫**被跟踪**的文本文件（二进制按扩展名跳过，且**逐个列出跳过项**）。未跟踪文件不扫
  ——不入库就不会泄漏，但后果说清：**刚写好还没 `git add`** 的文件里有本机路径时看不见。
- 特征集**有限**（盘符 `D:\\`、`C:\\Users`、同步盘两种写法、仓库根名、本机用户名、邮箱）：
  **别的主机专属信息**（内网域名、别人的用户名…）抓不到。
- 免除表是手抄位、不会自己变小；判据只打印"可收紧"、不红。

用法：python checks\\personal_paths_selfcheck.py     退出码 0 全过 / 1 有问题。
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                          # noqa: E402

C = Checker('personal_paths')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXEMPT = os.path.join(ROOT, 'checks', 'personal_paths_exempt.txt')
SELF = 'checks/personal_paths_selfcheck.py'
SKIP_EXT = ('.png', '.jpg', '.jpeg', '.gif', '.whl', '.pdf', '.ico', '.lock',
            '.pyc', '.pyo', '.so', '.dll', '.exe', '.npz')
SKIP_DIRS = ('__pycache__', '.pytest_cache', 'node_modules')

#: 特征（**拆开拼**，理由见模块 docstring；每条都别写成完整字面量）
BS = chr(92)
PATTERNS = (
    'D:' + BS,                        # 本机把仓放在 D 盘 ⇒ 盘符绝对路径
    'C:' + BS + 'Users',              # Windows 用户目录
    '百度' + '同步盘',                  # 同步盘（中文名）
    'Baidu' + 'Syncdisk',             # 同步盘（ASCII 名）
    'code' + '-nosync',               # 本机仓库根目录名（两仓的父目录）
    'LENOVO',                         # 本机用户名
    '1037769406' + '@qq.com',         # 作者邮箱（公开仓用 GitHub 身份，不写私人邮箱）
)
#: 对偶用的"肯定不存在的特征"。
NONSENSE = 'NoSuchPersonalFeature' + '_zzz'


def _tracked_files():
    """被跟踪文件（**必须 `-z` + `core.quotePath=false`**，否则中文路径静默逃逸）。"""
    try:
        r = subprocess.run(['git', '-C', ROOT, '-c', 'core.quotePath=false', 'ls-files', '-z'],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        print('  ✗ git ls-files 跑不起来：%s' % type(e).__name__)
        return None
    if r.returncode != 0:
        print('  ✗ git ls-files 非零退出：%s' % (r.stderr or '')[:120])
        return None
    return [x for x in r.stdout.split('\0') if x.strip()]


def _load_exempt():
    """免除表 → {路径: 允许处数}；无文件 = 空表（本仓债已清零）。"""
    if not os.path.isfile(EXEMPT):
        return {}
    table = {}
    with open(EXEMPT, encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print('  ✗ 免除表格式错（要 "<路径> <处数>"）：%r' % s)
                return None
            table[parts[0]] = int(parts[1])
    return table


def hits_of(text):
    """一段文本里的特征命中数（拆开拼的 PATTERNS 逐个数）。"""
    return sum(text.count(p) for p in PATTERNS)


def scan():
    """→ (命中表, 列出数, 免除表, 读不到, 跳过二进制, 实际读到数)；前置失败 → None。"""
    files = _tracked_files()
    exempt = _load_exempt()
    if files is None or exempt is None:
        return None
    rows, unreadable, skipped = [], [], []
    read = 0
    for rel in files:
        if rel in (SELF, 'checks/personal_paths_exempt.txt'):
            continue
        if rel.endswith(SKIP_EXT):
            skipped.append(rel)
            continue
        if any(part in SKIP_DIRS for part in rel.split('/')):
            skipped.append(rel)
            continue
        fp = os.path.join(ROOT, rel.replace('/', os.sep))
        try:
            with open(fp, encoding='utf-8', errors='replace') as f:
                text = f.read()
        except OSError as e:
            # ⚠ 响亮：读不到就不能对它下任何结论（"读不到的"与"没命中的"长得一模一样）
            print('  ✗ 读不到被跟踪文件，**不能**对它下结论：%s（%s）' % (rel, type(e).__name__))
            unreadable.append(rel)
            continue
        read += 1
        n = hits_of(text)
        if n:
            rows.append((rel, n))
    return rows, len(files), exempt, unreadable, skipped, read


def main():
    res = scan()
    if res is None:
        C.group('个人路径守卫：前置（拿不到清单/免除表 ⇒ 不静默放行）', [
            ('前置失败', False),
        ])
        return 0 if C.summary() else 1
    rows, listed, exempt, unreadable, skipped, read = res

    # ⚠ **显式打印**自证行（不只放在子条件标签里：`Checker` 的标签在通过时不打印，
    #    而"列了 N 个 ≠ 读了 N 个"这条必须**每次都能被人看见**——mecha 那版第一版就是
    #    非 ASCII 路径一个都没读到、却打印"扫描 151 个文件"）。
    print('[personal_paths] 自证：列出 %d 个被跟踪文件，**实际读到 %d 个**，'
          '按扩展名/目录跳过 %d 个（二进制等：%s），读不到 %d 个'
          % (listed, read, len(skipped), skipped[:3] or '无', len(unreadable)))

    C.group('A 前置自证：清单拿得到、**真读到东西**、跳过的（二进制）逐个列出', [
        ('列出 %d 个被跟踪文件' % listed, listed > 0),
        ('**实际读到 %d 个**（≠ 列出数；读到 0 就是判据瞎了）' % read, read > 0),
        ('读不到的文件：%s' % (unreadable or '无'), not unreadable),
        ('按扩展名/目录跳过 %d 个（二进制与字节码）：%s'
         % (len(skipped), skipped[:3]), True),
    ])

    offenders = []
    for rel, n in rows:
        allowed = exempt.get(rel)
        if allowed is None:
            offenders.append('%s（%d 处，未登记免除）' % (rel, n))
        elif n > allowed:
            offenders.append('%s（%d 处 > 允许 %d）' % (rel, n, allowed))
    C.group('B 被跟踪文件里无个人路径特征（命中即红）', [
        ('命中文件 %d 个 / 未登记或超额 %d 个：%s'
         % (len(rows), len(offenders), offenders[:6] or '无'), not offenders),
    ])

    if rows:
        print('    · 命中明细（路径：处数）：%s' % rows)
    shrinkable = [rel for rel, n in rows if exempt.get(rel, 0) > n]
    for rel in shrinkable:
        print('    · 可收紧（命中已低于登记）：%s' % rel)

    C.group('C 对偶：真特征命中、假特征 0 命中（非哑判据）', [
        ('真特征在样例里命中', hits_of('x ' + PATTERNS[0] + 'tmp') > 0),
        ('假特征 0 命中', hits_of(NONSENSE) == 0),
    ])

    ok = C.summary()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
