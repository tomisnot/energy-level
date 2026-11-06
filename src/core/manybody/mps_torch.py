"""Device-resident finite MPS kernels, independent of Hamiltonian/model adapters.

Physical dimensions may vary by site.  A tracked orthogonality centre makes
all truncations Schmidt truncations; arbitrary raw MPS input is canonicalized
before use.  PyTorch is supplied by the caller and remains an optional import.
"""


class TorchMPS:
    def __init__(self, torch, tensors, chi_max, truncerr, svd_driver=None):
        self.torch = torch
        self.tensors = [a.clone() for a in tensors]
        self.chi_max = int(chi_max)
        self.truncerr = float(truncerr)
        self.svd_driver = svd_driver
        if self.chi_max < 1 or not 0 <= self.truncerr < 1:
            raise ValueError('MPS needs chi_max >= 1 and 0 <= truncerr < 1')
        if not self.tensors or self.tensors[0].shape[0] != 1 or self.tensors[-1].shape[-1] != 1:
            raise ValueError('finite MPS must have boundary bond dimension 1')
        for i, a in enumerate(self.tensors):
            if a.ndim != 3 or min(a.shape) < 1 or not bool(torch.isfinite(a).all()):
                raise ValueError('MPS tensors must be finite rank-3 arrays (left, physical, right)')
            if i and self.tensors[i-1].shape[-1] != a.shape[0]:
                raise ValueError('MPS adjacent bond dimensions do not match')
        self.center = len(tensors) - 1
        self.move_center(0)
        norm = torch.linalg.vector_norm(self.tensors[0])
        if not bool(norm > 0):
            raise ValueError('MPS initial state has zero norm')
        self.tensors[0] /= norm
        self.discarded_weight = torch.zeros((), dtype=norm.dtype, device=norm.device)
        self.max_chi = max([a.shape[-1] for a in self.tensors])
        self.gate_count = 0
        self.swap_count = 0

    def move_center(self, target):
        torch = self.torch
        while self.center < target:
            i = self.center
            a = self.tensors[i]
            dl, d, dr = a.shape
            q, r = torch.linalg.qr(a.reshape(dl*d, dr), mode='reduced')
            self.tensors[i] = q.reshape(dl, d, q.shape[1])
            self.tensors[i+1] = torch.einsum('ab,bpc->apc', r, self.tensors[i+1])
            self.center += 1
        while self.center > target:
            i = self.center
            a = self.tensors[i]
            dl, d, dr = a.shape
            q, r = torch.linalg.qr(a.reshape(dl, d*dr).mH, mode='reduced')
            self.tensors[i] = q.mH.reshape(q.shape[1], d, dr)
            self.tensors[i-1] = torch.einsum('apb,bc->apc', self.tensors[i-1], r.mH)
            self.center -= 1

    def apply_one(self, i, gate):
        self.tensors[i] = self.torch.einsum('pq,aqb->apb', gate, self.tensors[i])

    def apply_two(self, i, gate=None, swap=False):
        """Apply a two-site gate or a physical-index SWAP in Schmidt gauge."""
        torch = self.torch
        self.move_center(i)
        a, b = self.tensors[i:i+2]
        dl, da, _ = a.shape
        _, db, dr = b.shape
        # Contract the internal bond as GEMM.  This is materially faster than
        # a generic einsum on CUDA while remaining independent of the model.
        theta = torch.matmul(a.reshape(dl*da, -1), b.reshape(-1, db*dr))
        theta = theta.reshape(dl, da, db, dr)
        if gate is not None:
            # For each (left,right) boundary, physical rows are a vector;
            # multiplying by gate.T applies U|psi> without a temporary 4-D
            # einsum operand.
            theta = torch.matmul(theta.permute(0, 3, 1, 2).reshape(dl*dr, da*db), gate.mT)
            theta = theta.reshape(dl, dr, da, db).permute(0, 2, 3, 1)
        if swap:
            theta = theta.permute(0, 2, 1, 3)
            da, db = db, da
            self.swap_count += 1
        mat = theta.reshape(dl*da, db*dr)
        if self.svd_driver:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False, driver=self.svd_driver)
        else:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        limit = min(self.chi_max, s.numel())
        w = s.square()
        # tail[k] = sum_{j>k} w[j], computed without catastrophic subtraction.
        tail = torch.cat((torch.cumsum(w.flip(0), 0).flip(0)[1:], w.new_zeros(1)))
        keep = limit
        if self.truncerr > 0 and limit > 1:
            # TeNPy/CPU MPS exposes trunc_cut as an amplitude threshold;
            # compare its square with the discarded Schmidt weight.
            keep = int((tail[:limit-1] > (self.truncerr**2)*w.sum()).sum().item()) + 1
        self.discarded_weight += tail[keep-1] / w.sum()
        s = s[:keep]
        s = s / torch.linalg.vector_norm(s)
        self.tensors[i] = u[:, :keep].reshape(dl, da, keep)
        self.tensors[i+1] = (s[:, None] * vh[:keep]).reshape(keep, db, dr)
        self.center = i + 1
        self.max_chi = max(self.max_chi, keep)
        self.gate_count += 1

    def apply_pair(self, a, b, gate):
        """Route any pair with adjacent swaps, apply gate, restore site order.

        SWAPs are representation operations and use the same Schmidt error
        control as physical gates.  Their discarded weight is included in the
        error report.  The algorithm has no lattice or model assumptions.
        """
        if not a < b:
            raise ValueError('pair endpoints must satisfy a < b')
        for i in range(b-1, a, -1):
            self.apply_two(i, swap=True)
        self.apply_two(a, gate)
        for i in range(a+1, b):
            self.apply_two(i, swap=True)

    @staticmethod
    def transfer_left(torch, env, a, op=None):
        temp = torch.einsum('ab,bqr->aqr', env, a)
        if op is not None:
            temp = torch.einsum('pq,aqr->apr', op, temp)
        return torch.einsum('api,apr->ir', a.conj(), temp)

    @staticmethod
    def transfer_right(torch, env, a):
        temp = torch.einsum('apb,bc->apc', a.conj(), env)
        return torch.einsum('apc,dpc->ad', temp, a)

    def environments(self):
        torch = self.torch
        ts = self.tensors
        one = ts[0].new_ones((1, 1))
        left = [one]
        for a in ts:
            left.append(self.transfer_left(torch, left[-1], a))
        right = [None]*len(ts) + [one]
        for i in range(len(ts)-1, -1, -1):
            right[i] = self.transfer_right(torch, right[i+1], ts[i])
        return left, right

    def expectations(self, ops, pairs=(), pair_ops=None):
        """Local operators and grouped two-point functions, in MPS order."""
        torch = self.torch
        left, right = self.environments()
        norm = left[-1][0, 0].real
        local = {}
        for name, op in ops.items():
            vals = []
            for i, a in enumerate(self.tensors):
                env = self.transfer_left(torch, left[i], a, op)
                vals.append((env * right[i+1]).sum()/norm)
            local[name] = torch.stack(vals)
        links = self.tensors[0].new_empty((len(pairs),))
        groups = {}
        for q, (a, b) in enumerate(pairs):
            p, m = pair_ops
            if a == b:
                env = self.transfer_left(torch, left[a], self.tensors[a], p @ m)
                links[q] = (env * right[a+1]).sum()/norm
            else:
                lo, hi = min(a, b), max(a, b)
                op1, op2 = (p, m) if a < b else (m, p)
                groups.setdefault((lo, a < b), []).append((hi, q, op1, op2))
        for (lo, _orientation), vals in groups.items():
            vals.sort(key=lambda v: v[0])
            env = self.transfer_left(torch, left[lo], self.tensors[lo], vals[0][2])
            last = lo+1
            for hi, q, _op1, op2 in vals:
                for k in range(last, hi):
                    env = self.transfer_left(torch, env, self.tensors[k])
                val = self.transfer_left(torch, env, self.tensors[hi], op2)
                links[q] = (val * right[hi+1]).sum()/norm
                last = hi
        return local, links, norm

    def pauli_expectation(self, specs, ops):
        """Arbitrary Pauli strings on the device-resident MPS (MPS order).

        ``specs`` terms are already in MPS order.  σˣ/σʸ are expanded into
        P/M exactly as the CPU path does; σᶻ uses the registered Z operator.
        The contraction reuses the same left/right environments as the local
        observables, so a batch of strings costs one environment sweep.
        """
        torch = self.torch
        left, right = self.environments()
        norm = left[-1][0, 0].real
        out = []
        for spec in specs:
            terms = [(int(s), str(a)) for s, a in spec['terms']]
            if not terms:
                out.append(complex(norm.item()))
                continue
            choices = []
            for _site, axis in terms:
                if axis == 'z':
                    choices.append([(1.0 + 0j, ops['Z'])])
                elif axis == 'x':
                    choices.append([(1.0 + 0j, ops['P']), (1.0 + 0j, ops['M'])])
                elif axis == 'y':
                    choices.append([(1j, ops['P']), (-1j, ops['M'])])
                else:
                    raise ValueError('Pauli 串轴须为 x/y/z，得到 %r' % (axis,))
            first = min(s for s, _ in terms)
            last = max(s for s, _ in terms)
            total = 0j
            stack = [(0, 1.0 + 0j, {})]
            while stack:
                idx, coef, chosen = stack.pop()
                if idx == len(terms):
                    env = left[first]
                    for k in range(first, last + 1):
                        env = self.transfer_left(torch, env, self.tensors[k],
                                                 chosen.get(k))
                    total += coef * complex((env * right[last + 1]).sum().item() / norm.item())
                    continue
                site, _axis = terms[idx]
                for c, op in choices[idx]:
                    nxt = dict(chosen)
                    nxt[site] = op
                    stack.append((idx + 1, coef * c, nxt))
            out.append(complex(total))
        return out

    def entropy(self, cut):
        if len(self.tensors) < 2:
            return 0.0
        self.move_center(cut-1)
        a = self.tensors[cut-1]
        w = self.torch.linalg.svdvals(a.reshape(-1, a.shape[-1])).square()
        w = w / w.sum()
        return float(-(w * w.clamp_min(1e-30).log()).sum().item())
