# -*- coding: utf-8 -*-
"""核心计算包：物理计算（无 UI 依赖）。

- core.py      单原子物理（能级 / AC/DC-Stark / 偶极矩阵元 / 谱线 / 偏振）
- pair_core.py 两体静态（pair 哈密顿 / vdW / 交换耦合 / 耦合强度雷达图）
- driver.py    统一引擎（build_system / drive / 线性响应谱）
- evolve.py    单原子含时演化引擎（ip4 / lindblad / auto；floquet、pwc 已
               封存至 src._archive.evolve_floquet_pwc，2025-08）
- waveforms.py 波形数据层（Wave / 包络 / build_fields → 演化场）
"""
