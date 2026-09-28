# -*- coding: utf-8 -*-
"""把 ARC 数据缓存子集复制到项目内 .arcdata/（免外部写权限，可复现）。

ARC（Alkali Rydberg Calculator）默认把数据缓存放在 ~/.arc-data，并在
Rubidium87() 实例化时写入 sqlite 缓存。本项目 src/__init__.py 会把
AlkaliAtom.dataFolder 重定向到项目根 .arcdata/；本脚本负责把 Rb-87 所需的
缓存文件复制过去（约 40 MB，不含其他原子的大文件）。

来源优先取 ~/.arc-data（已由 ARC 初始化过的完整缓存），缺失文件回退到
arc 包自带的 data 目录。重复运行安全（只复制缺失文件）。

用法: python -u scripts\\setup_arcdata.py [--all]
  --all：复制全部缓存（~234 MB，含所有原子；换原子时无需再跑本脚本）
"""
import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根
_DST = os.path.join(_HERE, '.arcdata')

# Rb-87 + Cs 及其共享角动量数据所需的文件（球谐/3j/6j 预计算与原子无关；
# Cs 供"换原子"与 core_selfcheck 的 Caesium 测试用；其余原子用 --all 全量）
_DEFAULT_FILES = [
    'version.txt',
    'rb87_precalculated.db',
    'rb_dipole_matrix_elements.npy',
    'rb_quadrupole_matrix_elements.npy',
    'rb_NIST_level_data.ascii',
    'rb87_hfs_data.csv',
    'rubidium_literature_dme.csv',
    'NIST_rb_dipole_matrix_elements.npy',
    'NIST_rb_quadrupole_matrix_elements.npy',
    'cs_precalculated.db',
    'cs_dipole_matrix_elements.npy',
    'cs_quadrupole_matrix_elements.npy',
    'cs_NIST_level_data.ascii',
    'cs_hfs_data.csv',
    'caesium_literature_dme.csv',
    'angularMatrix.npy',
    'angularMatrix_meta.npy',
    'precalculated3j.npy',
    'precalculated6j.npy',
]
_RB_DIRS = ['refractive_index_data']


def _sources():
    """返回候选源目录列表（~/.arc-data → arc 包 data 目录）。"""
    srcs = [os.path.join(os.path.expanduser('~'), '.arc-data')]
    try:
        import arc
        pkg = os.path.dirname(os.path.abspath(arc.__file__))
        srcs.append(os.path.join(pkg, 'data'))
    except Exception:
        pass
    return srcs


def main():
    all_atoms = '--all' in sys.argv[1:]
    srcs = _sources()
    os.makedirs(_DST, exist_ok=True)
    copied = skipped = missing = 0
    files = None
    if all_atoms:
        # 复制源目录全部文件/子目录（取第一个存在的源）
        for src in srcs:
            if os.path.isdir(src):
                for name in os.listdir(src):
                    p = os.path.join(src, name)
                    if os.path.isfile(p):
                        files = [(name, p)]
                    # 目录整体复制
                    d = os.path.join(_DST, name)
                    if os.path.isdir(p) and not os.path.exists(d):
                        shutil.copytree(p, d)
                        copied += 1
                break
    else:
        files = [(n, None) for n in _DEFAULT_FILES]
    for name, exact in (files or []):
        dst = os.path.join(_DST, name)
        if os.path.exists(dst):
            skipped += 1
            continue
        found = None
        if exact is not None and os.path.isfile(exact):
            found = exact
        else:
            for src in srcs:
                p = os.path.join(src, name)
                if os.path.isfile(p):
                    found = p
                    break
        if found is None:
            missing += 1
            print(f'  [缺] {name}')
            continue
        shutil.copy2(found, dst)
        copied += 1
        print(f'  [复制] {name}  ({os.path.getsize(dst)/1e6:.1f} MB)')
    for d in _RB_DIRS:
        dst = os.path.join(_DST, d)
        if os.path.exists(dst):
            continue
        for src in srcs:
            p = os.path.join(src, d)
            if os.path.isdir(p):
                shutil.copytree(p, dst)
                print(f'  [复制] {d}/')
                break
    size = sum(os.path.getsize(os.path.join(_DST, f))
               for f in os.listdir(_DST) if os.path.isfile(os.path.join(_DST, f)))
    print(f'\n.arcdata/ 就绪：新复制 {copied}，已存在跳过 {skipped}，'
          f'缺失 {missing}，总计 {size/1e6:.1f} MB')
    if missing:
        print('⚠ 有缺失文件——首次请用完整权限跑一次（或 --all 全量复制）')


if __name__ == '__main__':
    main()
