# -*- coding: utf-8 -*-
"""Compatibility shim: the dense TDVP kernel now lives in the product tree.

The independent benchmark and the torch prototype import this module; keeping
one implementation avoids a second definition drifting from the product
backend.  Use ``src.core.manybody.tdvp_dense`` for new code.
"""
from src.core.manybody.tdvp_dense import *  # noqa: F401,F403
