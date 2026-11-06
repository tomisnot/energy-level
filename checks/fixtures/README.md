# 基准模型夹具

这些 JSON 是**真实的 12/24 点 kagome 偶极 calib**（atoms + pairs），用于性能与
跨后端基准，不参与默认 `--smoke`。加载方式：

```python
import json
calib = json.load(open('checks/fixtures/kagome_n24.json'))
```

- `kagome_n12.json`：12 原子、66 对；B3 Pauli ED↔TDVP 对拍/基准使用。
- `kagome_n24.json`：24 原子、276 对；P2 GPU 24 点标定、P4 Pauli 开销使用。

它们只是数据，不含运行参数；`Psi0/t_max/npts/solver` 仍由调用方给。
