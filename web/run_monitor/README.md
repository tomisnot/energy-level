# Energy Level · AI Run Monitor 原型

这是一个独立的、无构建依赖的运行监控页面，用来先对齐视觉和信息层级。它不改动物理引擎，也不依赖真实 MCP 端点；页面里的四张卡片使用接近真实 `runspec.build_run()`、`runspec.evolve_run()` 和 `runspec.result_summary()` 的样例数据。

AI 模式通过 `app_entry.py --mcp` 启动时，项目会在同一进程启动只读运行卡片服务，并自动打开浏览器。端口写入项目根的 `.run-monitor-port`；页面从 `/api/runs` 读取当前进程的真实 RunSpec 投影、三槽进度和 `result_summary`。浏览器只收到主要态的降采样曲线，不会拉取完整 `pops` 大数组。

页面采用“运行卡片牌堆”：最新运行在最前，当前卡片占据主要视窗，历史卡片以缩小和降低透明度的方式叠在后面；使用鼠标滚轮、上下方向键或右侧按钮翻页。页面没有新建、删除或重跑入口，明确保持 AI 模式下人类只读。页面锁定在单个全屏视口，进行中的卡片只在右侧输出区显示一条短进度条，运行完成后结果直接覆盖同一位置。

## 本地预览

在项目根目录执行：

```powershell
python -m http.server 8765 --directory web/run_monitor
```

然后打开 <http://127.0.0.1:8765>。

页面内置三种状态：

- 进行中：显示阶段、百分比、预计剩余时间，并由演示 ticker 推进；
- 已完成：显示终态高布居、平均布居或多体平衡态摘要；
- 需关注：显示失败原因和下一步建议。

配置区参考 GUI 动态页的语义拆成：体系*算法、基组/时间网格、磁电场、两体几何、具体子能级列表、初态、耗散与相互作用。能级过多时先按能级组显示数量，再展开完整具体能级。波形不再堆叠 GUI 参数控件，而是用同源的包络图和一行轻量元数据表达。结果区同时兼容含时布居摘要与多体平衡态摘要。

## 数据映射

`app.js` 中的 `mapRunInput()`、`mapRunOutput()` 和 `mapProgress()` 是后续接真实监控数据的边界：

- `mapRunInput(run)` 把 RunSpec 压缩成卡片所需的模型、方法、基组、时间网格、波形、初态、耗散和运行旋钮；
- `mapRunOutput(result)` 同时兼容含时 `result_summary` 和多体平衡态摘要；
- `mapProgress(record)` 将 `progress_arr` 或后台 job 进度归一为百分比、阶段、耗时和 ETA。

离线静态预览仍可使用上面的 `python -m http.server` 命令，此时页面会回退到示例数据。真实运行由 AI 或 GUI 发起后，记录会进入同一进程的只读 store；完成结果保留摘要和小规模绘图数据，大数组继续留在 artifact。
