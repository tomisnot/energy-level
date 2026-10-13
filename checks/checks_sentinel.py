"""兼容包导出回归：src 的子包属性应仍是可导入模块。

测试脚本的语法、导入和退出码由实际运行暴露，不另扫测试源码。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src
import src.core
import src.ui


def main():
    for name in ('core', 'ui'):
        assert getattr(src, name) is sys.modules[f'src.{name}'], name
    print('[checks_sentinel] PASS=1 FAIL=0 SKIP=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
