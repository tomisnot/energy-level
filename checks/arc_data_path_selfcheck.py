# -*- coding: utf-8 -*-
"""ARC 数据目录优先使用 EL 仓内 .arcdata。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401
from arc.alkali_atom_functions import AlkaliAtom

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
want = os.path.join(root, '.arcdata')
if os.path.isdir(want):
    assert os.path.normcase(os.path.abspath(AlkaliAtom.dataFolder)) == \
        os.path.normcase(os.path.abspath(want))
    print('PASS ARC dataFolder -> %s' % AlkaliAtom.dataFolder)
else:
    print('SKIP .arcdata missing; ARC default=%s' % AlkaliAtom.dataFolder)
