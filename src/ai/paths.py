# -*- coding: utf-8 -*-
"""运行期产物目录（**唯一真值源**）——2026-09-14 使用者裁决迁移。

    runtime/_journal/<日期>.jsonl   操作审计流水（**入库**：是记录，不是垃圾）
    runtime/ai_results/*.json       CodeAct 运行记录 + 作业摘要（入库与否见 .gitignore）
    runtime/ai_results/*.npz        作业全量数组（按 _NPZ_KEEP 保留最近若干）

与"人手草稿"分开放（那条线是 scripts/tmp/，gitignore）：
**机器写的运行产物** 与 **人用完即弃的草稿** 是两种生命周期，不混住。
"""
import os

RUNTIME_DIRNAME = 'runtime'
JOURNAL_DIRNAME = os.path.join(RUNTIME_DIRNAME, '_journal')
RESULTS_DIRNAME = os.path.join(RUNTIME_DIRNAME, 'ai_results')
SCRATCH_DIRNAME = os.path.join('scripts', 'tmp')      # 草稿（gitignore）
