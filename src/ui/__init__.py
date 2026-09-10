# -*- coding: utf-8 -*-
"""界面包：Qt 组件与页面（依赖 core 包，被 core 单向向下）。

- fields.py         场面板（磁场/电场/光场表格）
- ui_theme.py       全局 QSS 主题
- ui_single.py      静态页画布（能级图 + off/on 镜像频谱）
- ui_radar.py       两体耦合雷达图
- ui_pair.py        响应谱画布（计算层保留，UI 暂缓）
- ui_waves.py       波形管理面板（可折叠波形组）
- ui_dynamic.py     动态页面板
- evolve_worker.py  演化后台线程
- app_state.py      共享状态层（LevelGroupStore / FieldState）
- pages/            组件页面目录（保留通用布局结构）

存档序列化已迁至 src/pm_setup.py（登记）+ src/archive.py（v3 快照/迁移）。
"""
