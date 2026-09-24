# -*- coding: utf-8 -*-
"""EL capability map 新鲜度守卫（§5.4 交付 2 的 EL 侧同款纪律）。

docs/capabilities.md 是 scripts/gen_capability_map.py 的生成物——**手改 = 红**；
声明漂了（EL 注册项变化没重新生成）也 = 红。mecha 仓判据读这份真表做并表
（交付 3），所以它的"生成物身份"必须在 EL 侧自己守住，不能等跨仓 diff 才发现。

用法：python checks/capability_map_selfcheck.py
退出码：0 一致；1 漂移或被手改。

## Known Limitations and Deferred Work

- 只验"文件 = 生成结果"，不验生成脚本本身对不对（那是 mecha 侧并表判据的
  责任面：共有集非空、独有不误并）。
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from checks.checklib import Checker                        # noqa: E402

C = Checker('capability_map')


def main():
    p = subprocess.run([sys.executable,
                        os.path.join(_ROOT, 'scripts', 'gen_capability_map.py'),
                        '--check'],
                       cwd=_ROOT, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    out = (p.stdout or '').strip() or (p.stderr or '').strip()
    C.group('EL docs/capabilities.md 是生成物（手改/漂移即红，交付 2 同款）', [
        ('--check 通过（%s）' % (out or '无输出'), p.returncode == 0),
    ])
    sys.exit(0 if C.summary() else 1)


if __name__ == '__main__':
    main()
