# CUDA TDVP 与 CPU SVD

`manybody.solver = "tdvp-cuda"` 使用有限开链两站点 TDVP。局域传播、MPO 环境和张量收缩使用 CUDA，SVD 通过 `manybody.svd_device` 选择 CPU 或 CUDA。

## 来源

CPU SVD 分支原已存在于 `scripts/tdvp_fast/torch_tdvp.py`。产品化将该数值内核移到 `src/core/manybody/tdvp_cuda_kernel.py`，研究脚本的旧导入路径通过兼容入口引用同一份实现。标准入口位于 `src/core/manybody/tdvp_cuda.py`。

新增工作包括后端注册、schema/RunSpec 接线、分段场与中点规则、原生状态续算、公共局域观测、Pauli 串、取消响应和数值元数据。驱动变化时复用固定交换通道，仅更新 onsite MPO 并重新计算环境。

## 配置

以下为已测混合配置，可在现有模型与初态配置上设置。

```json
{
  "manybody.solver": "tdvp-cuda",
  "manybody.dtype": "float32",
  "manybody.svd_device": "cpu",
  "manybody.chi_max": 128,
  "manybody.max_step_ns": 10.0,
  "manybody.drive_interp": "midpoint"
}
```

`float32` 对应 complex64 MPS，`float64` 对应 complex128 MPS。全局 schema 的 dtype 缺省值为 float64，复现快速配置需显式指定 float32。SVD 缺省使用 CPU。

| 配置 | 含义 |
|---|---|
| `manybody.svd_device` | `cpu` 或 `cuda` |
| `manybody.svd_driver` | CUDA SVD 可用 `auto`、`gesvd`、`gesvdj`，CPU SVD 配 `auto` |
| `manybody.tdvp_taylor_tol` | 局域 Taylor 阶数估计容差，省略时 complex64 为 2e-7，complex128 为 1e-10 |
| `manybody.tdvp_taylor_order` | 局域 Taylor 最高阶，默认24 |
| `manybody.truncerr` | 相对奇异值幅度阈值，默认1e-10 |
| `manybody.mps_order` | 显式原子下标排列 |
| `manybody.return_state` | 返回现有 `tdvp-dense` 格式的原生有限 MPS |

本后端的 `truncerr` 与 TeNPy 丢弃权重参数语义不同。Taylor 阶数估计为启发式，容差需结合步长和键维对拍评价。`error_report` 记录实际 χ、dtype、SVD 设备、截断语义、范数偏差和 TF32 标志。

当前支持厄米 Hamiltonian、任意长程两体通道、乘积态、同排序 dense-MPS 续算、分段驱动、局域观测与 Pauli 串。固定荷表示、投影约束、非厄米损耗、corr/link_pairs/loops 请求会显式报错。Pauli 原语可组合所需的高阶关联。

## MCP 接线

EL 现有 `query_params` 从统一 schema 枚举本后端。现有 `set_config` 设置上述配置，`run_sim` 或 `submit_job` 经标准 RunSpec 调度本后端。传输层验收由调用方在开启 MCP 后进行。

一次设置示意如下，使用已有 calib、初态与时间配置。

```json
{"key":"manybody.solver","value":"tdvp-cuda"}
{"key":"manybody.dtype","value":"float32"}
{"key":"manybody.svd_device","value":"cpu"}
```

这三条分别作为 `set_config` 的参数。随后调用 `run_sim`，其参数为 `{}`。MCP 返回公共摘要，大数组和原生状态遵循 EL 已有结果保存机制。

## 验证

```powershell
python run_checks.py --only manybody_tdvp_cuda
```

判据使用 N24 Kagome 夹具，覆盖公开 RunSpec、显式排序、单/双精度、CPU 对拍、Pauli、复相位与仅 Omega 改变的分段场、HDF5 续算、GPU SVD、取消和不支持的模型请求。另使用24个独立二能级系统的解析指数传播核查相位约定。

2026-11-09 的项目基准为 N24、χ128、10 ns 步长。真实撤场前200 ns中，原型 complex64/CPU SVD 耗时52.19秒，守恒两站点参考为333.48秒，占据最大偏差4.65e-8，态失配7.82e-11。所有计时为单次测量。

已有输入之间的表现有差异。独立 N24 逆序夹具的 complex64 短时占据偏差约4.7e-6。该配置适合作为速度与精度的候选，完整4 μs波形的步长、键维和累计误差仍需核查。

产品入口复跑同一项目200 ns撤场用时30.76秒，占据数据与原适配器末态一致。该入口复用固定交换通道，只刷新 onsite MPO 和环境。相对同一守恒参考的占据偏差4.65e-8，态失配7.82e-11。单次计时差同时受系统状态影响。
