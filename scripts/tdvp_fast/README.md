# Dense TDVP prototype (CPU / CUDA)

Independent performance prototype for the finite two-site TDVP backend.  The
production entry point now lives in `src/core/manybody/tdvp_dense.py`
(`manybody.solver = "tdvp-dense"`); this directory keeps the benchmark driver
and the experimental CUDA port.

## Files

| File | Role |
|---|---|
| `dense_tdvp.py` | Thin re-export of the product kernel `src/core/manybody/tdvp_dense.py` |
| `torch_tdvp.py` | CUDA port (torch; complex64 fast path, complex128 + CPU SVD accurate path); not yet a product backend |
| `mpo_compress.py` | Two-site SVD MPO compression experiments |
| `exp_mpo.py` | Constant-D exponential-sum MPO for 1D power-law couplings |
| `bench.py` | Reproducible benchmark: CPU/GPU vs TeNPy, optional ED reference |
| `bench_exp_mpo.py` | Reproducible 1D power-law benchmark: exact MPO vs exponential-sum MPO |

## Design

The numerical algorithm is the same two-site TDVP sweep as TeNPy
(`TwoSiteTDVPEngine`): half two-site update, SVD, one-site correction, then the
left-moving sweep; the last bond uses the full step.  The device-layer changes
are:

1. **No `np_conserved` / `SingleSiteH` objects.**  The state is plain numpy
   tensors with the same `(fL, fR)` canonical-form algebra as TeNPy.
2. **Position-dependent MPO bond dimensions.**  A channel starts at a fixed
   site; at bond `i` only channels started before `i` are active.  The local
   MPO dimension is `2 + #started`, not the full `2 + #channels`.  This is exact
   and roughly halves the average contraction cost.
3. **BLAS-shaped matvecs.**  The two-site effective-H application is two
   `matmul` calls on reshaped matrices instead of 5-D `tensordot`/`einsum`
   chains.  Environment updates use explicit pairwise contractions (a
   four-operand `einsum` was the original bottleneck).
4. **Taylor local propagator with Lanczos fallback.**  For the usual
   `max_step_ns` the local phase `|alpha|*||H||` is small, so 3–5 matvecs of a
   Taylor series reach the target tolerance; Lanczos is used if the series
   does not converge.  This removes most of the 20-step Krylov cost without
   changing the result (CPU vs Lanczos pops ≈ 5e-11).
5. **Segments and Pauli.**  A drive segment change rebuilds only the onsite
   MPO entries and re-initializes the environments; the MPS is carried across.
   Pauli strings use a finite-MPS transfer contraction (no 2^N vector), in
   atom order or with an explicit `mps_order`.
6. **CUDA port** keeps the MPS/MPO/environments on device; complex64 is the
   useful dtype because cuSOLVER complex128 SVD is slower than CPU.

## Measured results

Machine: RTX 4050 Laptop (6.4 GB), 16 logical CPU cores, Python 3.10,
numpy 2.2.6, torch 2.7.1+cu128.  `kagome_n12/n24` fixtures, `Omega=0.5 MHz`,
`t=20 ns`, `npts=3`, `max_step_ns=2`; dense times are medians of 2–3 runs,
TeNPy times are single runs.  Same MPS order (cylinder).

| Case | TeNPy `tdvp` | dense CPU (complex128) | speedup | max\|Δpops\| |
|---|---:|---:|---:|---:|
| N=12, χ=64 | 11.65 s | 0.85 s | **13.7×** | 4.3e-7 |
| N=24, χ=64 | 87.34 s | 18.46 s | **4.7×** | 9.6e-7 |

GPU (complex64):

| Case | dense CPU | CUDA complex64 + GPU SVD | GPU vs CPU | max\|Δpops\| vs CPU |
|---|---:|---:|---:|---:|
| N=24, χ=64 | 18.46 s | 6.63 s | 2.8× | ~4.4e-5 |
| N=24, χ=128, truncerr=0 | 159.2 s | 6.41 s | **24.8×** | ~3.6e-5 |

CUDA precision ladder (N=24, χ=64, t=20 ns, one run):

| dtype / SVD device | wall | max\|Δpops\| vs CPU complex128 |
|---|---:|---:|
| complex64 / GPU | 4.8 s | ~4e-5 |
| complex64 / CPU | 6.8 s | ~1e-6 |
| complex128 / CPU | 10.4 s | ~1.8e-6 |
| CPU complex128 | 19.1 s | reference |

`complex64 + GPU SVD` is the fastest; `complex128 + CPU SVD` is the accurate GPU
choice (cuSOLVER complex128 SVD is slower than numpy at these sizes).  At χ=128
the complex64 CPU-SVD path costs 21 s, so keep GPU SVD there.

Accuracy anchors: N=4 vs ED <1e-6; N=12 (t=10) vs ED 1.8e-7; product
`tdvp-dense` vs `tdvp` at N=12 t=2 is 7.4e-9.  The CPU path is the accurate
one; complex64 is an explicit speed/precision tradeoff.

## Reproduce

```powershell
python scripts/tdvp_fast/bench.py --case n12 --chi 64 --device cpu --runs 3 --reference tenpy
python scripts/tdvp_fast/bench.py --case n24 --chi 64 --device cpu --runs 2 --reference tenpy
python scripts/tdvp_fast/bench.py --case n24 --chi 64 --device gpu --dtype complex64 --runs 3
python run_checks.py --only manybody_tdvp_dense
python scripts/tdvp_fast/bench_exp_mpo.py --n 96 --k 8
```

## Support surface (product `tdvp-dense`)

Finite open chain, piecewise-constant `drive_segments` (default left;
`manybody.drive_interp='midpoint'` for second-order ramp control points), Hermitian
(`gamma=0`), product initial states, optional `mps_order`, `pauli_strings`,
optional local `observables` (`sx/sy/sz/sp/sm/phase`).  Rejected explicitly:
fixed charge, `return_state`, `corr`, `link_pairs`/`loops`, projected plans.
`auto` does not select it; callers must ask for `tdvp-dense`.

Performance numbers must carry `chi_actual`, not only `chi_max`.
N=12 spot checks (same order, χ=64): segments 0.37 s vs `tdvp` 7.37 s
(max|Δpops| 3.2e-8); 24 Pauli strings 0.63 s vs 12.8 s
(max|ΔP| 4.3e-6).

## 2026-11-07 round: what was tried

| 方向 | 结果 | 结论 |
|---|---|---|
| B1 guarded `gesvda` | 宽松 guard：441 fast/9 fallback，max\|Δpops\|=0.70；严格 guard（cond<1e3）：33 fast/417 fallback，1.8e-6 但 23.7s vs CPU 18.5s | 负：能保精度时已无速度优势 |
| A1 位置感知 MPO 压缩 | N=24 χ=64：Dc=48 精确但 18.5s（exact 17.6s）；Dc=40 21.2s / 7.6e-5；Dc=32 27.6s / 5.6e-4 | 负：压缩改变纠缠/条件数，省下的 D 抵不过 χ 增长 |
| A2 指数和 MPO（1D 1/r³） | N=36：K=8 1.27s vs 2.14s（1.69×），Δpops 4e-6。N=48：K=8 1.68s vs 3.80s（2.26×），1.9e-5。**N=96，OMP-grid 拟合，K=8 D=18 vs exact D=192：16.89s → 3.56s（4.74×），fit max_rel 1.5e-4，Δpops 4.3e-9** | **正**：1D 幂律有真实收益；研究构造器可用，产品化需新的近似模型表示 |
| B2 单站点 TDVP | TeNPy 单站点仍做每站点 gauge SVD；跳过 one-site 的简化版 N=24 误差 0.85 且慢 2.3× | 负/不做：不是免费加速，需要 zero-site projector + 子空间扩张 |
| C1 `torch.compile`/CUDA graph | complex 算子 inductor 不支持，compiled 更慢；graph 仅 ~5% | 负：matvec 本身已足够高效 |
| C2 CPU BLAS 线程 | N=12 1 线程 0.53s vs 16 0.61s；N=24 8 线程 16.1s vs 16 18.6s | 正，约 15%，需 opt-in |
| C3 批量 | bmm 无单矩阵收益；batched eigh 每矩阵约快 20%（B≥4） | 只对扫描吞吐有意义 |

## Final decisions (2026-11-07)

| 方向 | 决定 | 理由 |
|---|---|---|
| CPU `tdvp-dense` | **已产品化** | 13–18×，对 ED/TeNPy <1e-6 |
| C2 BLAS threads | **已接入**（env opt-in） | 约 15%，不改变默认全局线程 |
| A2 指数和 MPO | **研究构造器保留，暂不产品化** | 只覆盖 1D 平移不变幂律；是近似模型，产品需要新 calib 表示 + 逐通道拟合 + 误差门；当前 `bench_exp_mpo.py` 可直接复现 N=96 的 4.74× |
| B1 guarded gesvda | 拒绝 | 宽松 guard 错；严格 guard 慢于 CPU |
| A1 MPO 压缩 | 拒绝 | 精确压缩中性；激进压缩更慢 |
| B2 单站点 TDVP | 拒绝（本轮） | TeNPy 单站点仍需 gauge SVD；简化版崩；真做是独立算法项目 |
| C1 融合/CUDA graph | 拒绝 | complex compile 不支持；graph 仅 ~5%；matvec 占比低 |
| C3 批量 | 只保留“扫描吞吐”用途 | 单次延迟无收益 |

## Measured dead ends (do not repeat blindly)

- **SVD via `eigh(A A^H)`**: 4–22× faster than cuSOLVER `gesvdj`, but the
  finite-TDVP trajectory diverges (N=24: max|Δpops| 2.9e-4 at t=4, 0.52 at t=6)
  because squaring the Gram matrix loses the small singular values that control
  truncation.  Rejected.
- **cuSOLVER `gesvda`**: 20× faster than `gesvdj` for complex128 (13.5 ms vs
  285 ms at 256×256), accurate on random full-rank matrices, but the same TDVP
  trajectory diverges (0.43).  Only usable with a residual/condition guard, not
  as a drop-in; rejected for now.
- **QR + SVD**, **`svd_lowrank`**, **batched SVD**: no gain at these sizes
  (256×256), or slower.
- **TF32 for complex64 matmul**: 1.2× faster but introduces ~3e-4 relative
  error, larger than the useful complex64 budget; not used.
- **CPU BLAS threads**: N=12/χ=17 is fastest with 1 thread (0.53 s vs 0.61 s at
  16); N=24/χ=64 is fastest with 8 threads (16.1 s vs 18.6 s at 16).  This is a
  process-global setting, so it would need `threadpoolctl`/an opt-in knob.
- **Batched bmm/eigh**: bmm has no per-matrix gain at these sizes; batched eigh
  is ~20% faster per matrix at B≥4.  Useful for sweep throughput, not single-run
  latency.

## Open items

- Promote the CUDA port to `tdvp-cuda` only after its complex64 error budget is
  documented and the result contract is wired (it currently lives in the
  prototype only).
- CPU SVD inside the CUDA path is only a win for complex128 (cuSOLVER complex128
  is slow); complex64 should stay on GPU SVD.
- Check N=36 and a more entangled quench; the current fixtures reach χ≈17/64,
  not the worst case for the dense kernel.
- Consider a shared “local propagator” helper if more backends start using
  Taylor/Lanczos; do not duplicate the tolerance logic.
