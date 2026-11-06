"""Bounded, data-driven projection of engine results for the read-only Web UI.

Numerical results stay in artifacts. This projection keeps true coordinates,
site identities, complex components, missing values and explicit sampling notes.
"""
from __future__ import annotations

import numpy as np

FRAMES = 160
MAX_CELLS = 240_000
MAX_LINKS = 700

LABELS = {
    'pops':'态 / 构型概率',
    'pops_atom': '激发占据 ⟨nᵢ⟩', 'n_atom': '激发占据 ⟨nᵢ⟩',
    'occupations': '激发占据 ⟨nᵢ⟩', 'S': '纠缠熵 S', 'energy': '能量',
    'E': '内能 E', 'C': '热容 C', 'F': '自由能 F', 'rho': '态密度 ρ(E)',
    'corr': '连通占据关联 Cᵢⱼ', 'observables.xy_corr': '横向关联 ⟨σ⁺ᵢσ⁻ⱼ⟩',
    'observables.sx': '局部 σˣ', 'observables.sy': '局部 σʸ',
    'observables.sz': '局部 σᶻ', 'observables.sp': '局部 ⟨σ⁺⟩',
    'observables.sm': '局部 ⟨σ⁻⟩', 'observables.phase': '局部相干相位',
    'observables.link_corr': '链路关联 ⟨σ⁺ᵢσ⁻ⱼ⟩',
    'observables.loop_phase': '闭环关联相位', 'observables.loop_winding': '闭环绕数',
    'observables.loop_phase_series': '闭环关联相位',
    'observables.loop_winding_series': '闭环绕数',
    'observables.pauli_strings': 'Pauli 串期望',
    'physical_loss': '物理损失', 'norm_drift': '数值范数漂移',
    'no_jump_weight_deficit': '无跳跃权重亏损', 'survival_weight': '无跳跃存活权重',
    'spins': '局部自旋向量', 'gaps': '能隙', 'ns': '平均激发密度',
    'Om_t': '驱动 Ω', 'de_t': '失谐 δ', 'evals': '本征能谱',
    'one_particle_energies': '单粒子能谱', 'jump_counts': '跳跃次数分布',
    'E0':'基态能量 E₀','gap':'能隙 Δ',
    'average_sign':'平均符号','samples':'统计样本数',
    'spectral':'源态谱 S(ω)','spectrum_weights':'谱权重','source_norm':'源态权重',
}


def at(data, path):
    for key in path.split('.'):
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def sampled_indices(length, limit=FRAMES):
    return np.unique(np.linspace(0, length-1, min(length, max(2,limit)), dtype=int)) if length else np.array([],int)


def numbers(array):
    """Strict JSON numbers; complex arrays are encoded explicitly elsewhere."""
    a = np.asarray(array, float)
    return np.where(np.isfinite(a), a, None).tolist()


def project_drives(run, config):
    """Sample authoritative input envelopes, never inventing missing waves."""
    end=run.get('t_max')
    if end is None or end<=0:
        return []
    t=np.linspace(0,float(end),160)
    traces=[]
    if run.get('nbody')=='many':
        from src.core.manybody.runtime import segment_drive
        n=run.get('n_atoms',0)
        om=np.asarray(run.get('Om',np.zeros(n)),complex)
        delta=np.asarray(run.get('delta',np.zeros(n)),float)
        drive=[segment_drive(float(x),run.get('segments'),n) for x in t]
        for site in range(min(n,3)):
            traces.append(dict(label=f'Ω · 原子 {site}',x=numbers(t),
                values=numbers([abs(v[0][site]) if v else abs(om[site]) for v in drive]),unit='MHz'))
            if any(v is not None for v in drive) or np.any(delta):
                traces.append(dict(label=f'δ · 原子 {site}',x=numbers(t),
                    values=numbers([v[1][site] if v else delta[site] for v in drive]),unit='MHz'))
        return traces
    from src.core.waveforms import restore_waves
    for channel in ('single','A','B'):
        for wave in restore_waves(config.get('dynamic.waves_'+channel) or []):
            traces.append(dict(label=channel+' · '+wave.name,x=numbers(t),
                values=numbers(wave.intensity_mwcm2*np.asarray(wave.waveform(t))),unit='mW/cm²'))
    return traces


def model_view(calib):
    """模型本体 → 卡片可视化需要的三个数组（耦合/δ/Ω）。不做展示加工。"""
    if not isinstance(calib, dict) or not (calib.get('atoms') or []):
        return None
    atoms = calib.get('atoms') or []
    pairs = calib.get('pairs') or []
    return dict(
        n_atoms=len(atoms),
        delta=numbers(np.array([float(a.get('delta_mhz') or 0.0) for a in atoms])),
        omega=numbers(np.abs(np.array([complex(a.get('Omega_mhz') or 0.0)
                                       * np.exp(1j * float(a.get('Omega_phase_rad') or 0.0))
                                       for a in atoms]))),
        pairs=[[int(p['i']), int(p['j']),
                float(np.real(p.get('J') or 0.0)),
                float(np.real(p.get('V') or 0.0))]
               for p in pairs],
        terms=list(calib.get('terms') or []))


def project_result(run, result):
    result = result or {}
    sites = result.get('sites') or run.get('atoms') or (run.get('calib') or {}).get('atoms') or []
    sites = [dict(site_id=s.get('site_id',s.get('id',i)),
                  pos=numbers(s['pos']) if s.get('pos') is not None else None)
             for i,s in enumerate(sites)]
    schema = result.get('data_schema') or {}
    descriptors = dict(schema.get('quantities') or {})
    n = len(sites)
    # Compatibility for single/pair results and results produced before v1.
    if not descriptors:
        for key in ('pops_atom','n_atom','S','energy','E','C','F','rho','corr','spins'):
            value = result.get(key)
            if value is None:
                continue
            a = np.asarray(value)
            axes = (['time','site'] if key in ('pops_atom','n_atom') and a.ndim==2 else
                    ['site'] if key=='n_atom' else ['site','site'] if key=='corr' else
                    ['time','site','component'] if key=='spins' else
                    ['beta'] if key in ('E','C','F') else ['energy'] if key=='rho' else
                    ['time'] if a.ndim else [])
            descriptors[key] = dict(axes=axes, unit='MHz' if key in ('E','F','energy') else '1')
    if result.get('pops') is not None and 'pops' not in descriptors:
        descriptors['pops'] = dict(axes=['time','basis'],unit='1')
    if result.get('one_particle_energies') is not None:
        descriptors['one_particle_energies'] = dict(axes=['eigenstate'],unit='MHz')
    for key in ('average_sign','samples'):
        if result.get(key) is not None and np.asarray(result[key]).ndim==1:
            descriptors[key]=dict(axes=['beta'],unit='1')
    axes = {'time': result.get('t'), 'beta': result.get('betas'),
            'energy': result.get('energies'), 'tau': result.get('taus')}
    datasets, scalars, notes, represented = [], [], [], set()
    remaining = MAX_CELLS

    def encode(a):
        nonlocal remaining
        if a.size * (2 if np.iscomplexobj(a) else 1) > remaining:
            return None
        remaining -= a.size * (2 if np.iscomplexobj(a) else 1)
        if np.iscomplexobj(a):
            return dict(values=numbers(a.real), imaginary=numbers(a.imag))
        return dict(values=numbers(a))

    # Prefer physical site data; compatibility aliases never create duplicate views.
    ordered = sorted(descriptors, key=lambda p: (0 if p in ('pops_atom','n_atom') else 1,p))
    for path in ordered:
        meta = descriptors[path]
        value = at(result,path)
        if value is None or meta.get('alias_of') or path in ('stderr','E_err'):
            continue
        try:
            a = np.asarray(value)
            if a.size == 0 or a.dtype.kind not in 'fciu':
                continue
            axis_names = meta.get('axes',[])
            label = LABELS.get(path,path)
            if a.ndim == 0:
                scalars.append(dict(id=path,label=label,value=numbers(a),unit=meta.get('unit','1')))
                represented.add(path)
                continue
            if path=='pops' and result.get('pops_kind')=='site_occupation' and 'pops_atom' in represented:
                represented.add(path)
                continue
            if path=='basis_probabilities' and result.get('pops_kind')=='basis_probability':
                represented.add(path)
                continue
            ds = dict(id=path,label=label,unit=meta.get('unit','1'),shape=list(a.shape))
            axis = axis_names[0] if axis_names and axis_names[0] in axes else None
            ids = None
            if axis:
                coordinate = np.asarray(axes.get(axis))
                if coordinate.ndim != 1 or len(coordinate)!=len(a) or not np.isfinite(coordinate).all():
                    notes.append(label+'：缺少匹配的真实坐标轴，未绘制')
                    continue
                per_frame = max(1,int(a.size/len(a)))
                ids = sampled_indices(len(a),min(FRAMES,max(2,remaining//max(1,per_frame*2))))
                ids=ids[np.argsort(coordinate[ids],kind='stable')]
                ds.update(axis=axis,x=numbers(coordinate[ids]),indices=ids.tolist(),original_frames=len(a))
                # 轴单位由**数据**声明（data_schema.axes[axis].unit）；前端的常量表只作兜底
                ds['axis_unit'] = ((schema.get('axes') or {}).get(axis) or {}).get('unit')
                if len(ids)<len(a):
                    ds['sampling'] = f'{len(a)} → {len(ids)} 个实际采样点，包含首尾'
                a = a[ids]
            else:
                ds['axis'] = None
            if path == 'observables.pauli_strings':
                labels = ((result.get('observables') or {}).get('pauli_string_labels')
                          or [])
                if a.ndim != 2:
                    notes.append(label + '：Pauli 串数组维度不匹配，未绘制')
                    continue
                for q in range(a.shape[1]):
                    enc = encode(a[:, q])
                    if enc is None:
                        notes.append(label + '：Pauli 串超出卡片预算，未绘制')
                        break
                    ds_q = dict(id='%s.%d' % (path, q),
                                label='%s · %s' % (label,
                                                   labels[q] if q < len(labels) else q),
                                unit=ds['unit'], shape=[len(a)], kind='curve',
                                axis='time', x=ds.get('x'),
                                axis_unit=ds.get('axis_unit'), complex=True)
                    ds_q.update(enc)
                    datasets.append(ds_q)
                represented.add(path)
                continue
            if axis_names == ['site','site']:
                if a.ndim!=2 or a.shape!=(n,n):
                    notes.append(label+'：矩阵与原子身份不匹配')
                    continue
                diagonal=encode(np.diag(a))
                if diagonal:
                    datasets.append(dict(id=path+'.diagonal',label='局部占据涨落 Cᵢᵢ' if path=='corr' else label+' · 对角元',
                        kind='sites',axis=None,unit=ds['unit'],complex=np.iscomplexobj(a),**diagonal))
                i,j=np.triu_indices(n,1)
                z=a[i,j]
                keep=np.flatnonzero(np.isfinite(z) & (abs(z)>1e-14))
                if len(keep)>MAX_LINKS:
                    keep=keep[np.argpartition(abs(z[keep]),-MAX_LINKS)[-MAX_LINKS:]]
                    ds['sampling']=f'显示 |值| 最强的 {MAX_LINKS} 条关联；其余边未显示'
                ds.update(kind='links',edges=np.column_stack([i[keep],j[keep]]).tolist(),
                          temporal=False,scope='末态关联',total_edges=int(len(z)))
                a=z[keep]
            elif path=='observables.link_corr':
                pairs=np.asarray((result.get('observables') or {}).get('link_pairs',[]),int)
                if pairs.shape!=(a.shape[-1],2):
                    notes.append(label+'：缺少匹配的链路身份，未绘制')
                    continue
                if len(pairs)>MAX_LINKS:
                    strength=np.max(np.where(np.isfinite(a),abs(a),0),axis=0)
                    keep=np.argsort(-strength,kind='stable')[:MAX_LINKS]
                    pairs=pairs[keep];a=a[:,keep]
                    ds['sampling']=f'显示全序列中 |值| 最强的 {MAX_LINKS} 条链路'
                ds.update(kind='links',edges=pairs.tolist(),temporal=bool(axis))
            elif 'loop' in axis_names:
                ds.update(kind='loops',loops=result.get('loops') or [])
                ds['label'] += ' · 含时' if axis else ' · 末态'
            elif path=='spins':
                ds['kind']='spins'
            elif 'site' in axis_names:
                if a.shape[-1]!=n or not n:
                    notes.append(label+'：缺少原子坐标或身份不匹配')
                    continue
                ds['kind']='sites'
                err=result.get('stderr') if path=='pops_atom' else None
                if err is not None and ids is not None and np.asarray(err).shape==np.asarray(value).shape:
                    ds['stderr']=numbers(np.asarray(err)[ids])
            elif 'basis' in axis_names:
                if a.ndim!=2:
                    continue
                ranking=np.nanmax(np.abs(a),axis=0)
                chosen=np.argsort(-ranking,kind='stable')[:16]
                a=a[:,chosen]
                basis=result.get('basis_states')
                labels=result.get('states') or []
                ds.update(kind='basis',codes=[str(int(basis[i])) if basis is not None else str(int(i)) for i in chosen],
                    labels=[str(labels[i].get('label',i)) if i<len(labels) and isinstance(labels[i],dict) else str(i) for i in chosen],
                    sampling=(f'主要 {len(chosen)} / {len(ranking)} 个构型；不是完整分布'
                              if len(chosen)<len(ranking) else f'完整 {len(chosen)} 个构型'))
            elif path=='jump_counts':
                unique,counts=np.unique(a[np.isfinite(a)],return_counts=True)
                if len(unique)<=40:
                    x=unique
                else:
                    counts,bins=np.histogram(a[np.isfinite(a)],bins=40)
                    x=(bins[:-1]+bins[1:])/2
                    ds['sampling']='合并为 40 个计数区间'
                a=counts; ds.update(kind='histogram',x=numbers(x))
            elif axis:
                ds['kind']='curve'
                if path=='E' and result.get('E_err') is not None:
                    ds['stderr']=numbers(np.asarray(result['E_err'])[ids])
            elif path in ('evals','one_particle_energies'):
                ds['kind']='spectrum'
                if len(a)>256:
                    ds['sampling']=f'显示前 256 / {len(a)} 个能级'
                a=a[:256]
                if path=='one_particle_energies':
                    ds['occupied']=[int(i) for i in result.get('occupied',[])]
            else:
                notes.append(label+'：未支持此数据形状；数值保留在完整运行存档')
                continue
            encoded=encode(a)
            if encoded is None:
                notes.append(label+'：超出浏览器数据预算；完整结果保留在运行存档')
                continue
            if 'stderr' in ds:
                uncertainty=encode(np.asarray(ds['stderr'],float))
                if uncertainty is None:
                    ds.pop('stderr')
                    notes.append(label+'：误差带超出浏览器预算，完整误差保留在运行存档')
                else:
                    ds['stderr']=uncertainty['values']
            ds.update(encoded)
            if np.iscomplexobj(a):
                ds['complex']=True
            datasets.append(ds); represented.add(path)
        except (ValueError,TypeError,IndexError) as exc:
            notes.append(path+'：无法映射（'+str(exc)[:120]+'）')

    # Entropy belongs to a bipartition, never to individual atoms. Only show
    # groups when the solver supplies the actual cut and site ordering.
    entropy=next((d for d in datasets if d['id']=='S'),None)
    if entropy is None:
        scalar=next((s for s in scalars if s['id']=='S'),None)
        if scalar:
            entropy=dict(scalar,values=scalar['value'],axis=None)
    err=result.get('error_report') or {}
    cut=result.get('entropy_cut',err.get('entropy_cut'))
    order=result.get('order',err.get('mps_order',list(range(n))))
    if entropy and n and cut is not None and sorted(order)==list(range(n)) and 0<int(cut)<n and np.asarray(entropy['values']).size<=remaining:
        datasets.append(dict(entropy,id='entropy_partition',kind='partition',
            label='纠缠熵 · 子系统划分',part_a=list(order[:int(cut)]),
            sampling='两色仅表示实际 A/B 子系统；S 是整体二分纠缠熵'))
        remaining-=np.asarray(entropy['values']).size

    # Local quantum spin arrows combine actual expectation values; this is
    # not a classical approximation or a pure-state reconstruction.
    components=[next((d for d in datasets if d['id']=='observables.'+key),None)
                for key in ('sx','sy','sz')]
    if all(components):
        shared=sorted(set(components[0]['x']).intersection(components[1]['x'],components[2]['x']))
        if shared and len(shared)*n*3<=remaining:
            values=[np.stack([np.asarray(d['values'])[d['x'].index(x)] for d in components],axis=-1)
                    for x in shared]
            datasets.insert(1,dict(id='local_bloch',kind='spins',label='局部自旋 ⟨σ⟩',
                axis=components[0]['axis'],x=shared,values=numbers(values),unit='1',
                sampling='箭头来自 ⟨σˣ⟩、⟨σʸ⟩，节点颜色为 ⟨σᶻ⟩'))
            remaining-=len(shared)*n*3
    for key in ('chi','average_sign','samples'):
        value=result.get(key)
        if value is not None and np.asarray(value).ndim==0:
            diagnostics_value=numbers(value)
            scalars.append(dict(id=key,label={'chi':'键维 χ','average_sign':'平均符号','samples':'统计样本'}[key],value=diagnostics_value,unit='1'))
    # Optional native states are never serialized as opaque strings or huge tensors.
    state=result.get('state')
    state_info=None
    if isinstance(state,dict):
        state_info=dict(kind=state.get('kind'),backend=state.get('backend'),
                        order=state.get('order'),charge_sector=state.get('charge_sector'),
                        note='原生张量态保留在运行存档；此处仅展示元信息')
    for path in ('vec','psi_amp'):
        vec=result.get(path)
        if vec is None:
            continue
        vec=np.asarray(vec).reshape(-1)
        if remaining<48:
            notes.append('态振幅预览超出浏览器预算；原态保留在运行存档')
            continue
        chosen=np.argsort(-abs(vec)**2,kind='stable')[:16]
        basis=result.get('sector_idx')
        if basis is None:
            basis=result.get('basis_states')
        if basis is None:
            basis=(result.get('basis') or {}).get('indices')
        datasets.append(dict(id=path,kind='basis',label='末态振幅' if path=='psi_amp' else '基态振幅',
            axis=None,unit='1',values=numbers(abs(vec[chosen])**2),
            phases=numbers(np.angle(vec[chosen])),codes=[str(int(basis[i])) if basis is not None else str(int(i)) for i in chosen],
            labels=[],sampling=f'主要 {len(chosen)} / {len(vec)} 个构型；概率为 |振幅|²'))
        remaining-=len(chosen)*2
    diagnostics={k:v for k,v in (result.get('error_report') or {}).items()
                 if isinstance(v,(str,int,float,bool,type(None)))}
    # 模型本身（哈密顿量）随结果一起下发：卡片要能**看见**它在算什么，
    # 而不只是"N 原子 · P 对"。只给渲染用得到的三个数组，不做展示加工。
    model = model_view(run.get('calib') if isinstance(run, dict) else None)
    return dict(version=1,sites=sites,datasets=datasets,scalars=scalars,notes=notes,
                state=state_info,diagnostics=diagnostics,playback_seconds=8,
                model=model,
                backend=result.get('backend') or result.get('method'),
                approximation=schema.get('approximation'),artifact_id=None)
