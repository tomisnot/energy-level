/*
 * Energy Level run monitor prototype.
 *
 * The three mapping functions are the future real-data boundary:
 *   mapRunInput(run)   : runspec.build_run(...) -> card.input
 *   mapRunOutput(res)  : runspec.result_summary(...) -> card.output
 *   mapProgress(event)  : progress_arr / job event -> card.progress
 *
 * This prototype is deliberately read-only. The demo ticker only advances a
 * local fixture so the visual state of a running job can be inspected.
 */

const $ = (selector, root = document) => root.querySelector(selector)
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)]
const clone = (value) => JSON.parse(JSON.stringify(value))

const DEMO_RUNS = [
  {
    id: 'run-20261015-014', status: 'running', createdAt: '刚刚', progress: 58,
    stage: '积分 · 两体 Sambe', elapsed: '00:12', eta: '约 9 秒',
    run: {
      version: 1, nbody: 'pair', two_body: true, method: 'sambe', method_in: 'sambe',
      t_max: 1200, npts: 2401, field: { B_gauss: 2.5, E_vpcm: 0 },
      states: Array.from({ length: 54 }, (_, i) => ({ label: i < 2 ? ['60S₁/₂ mJ=−0.5', '60P₃/₂ mJ=+0.5'][i] : `state-${i}` })),
      labels: ['60S₁/₂ mJ=−0.5', '60P₃/₂ mJ=+0.5'],
      n_waves: { A: 1, B: 1 },
      waves: { A: [{ name: 'mw-A', freq_mhz: 6832.4, intensity: 0.48, pol: 'σ−', env: ['gauss', 600, 180, 1] }], B: [{ name: 'mw-B', freq_mhz: 6832.4, intensity: 0.48, pol: 'σ−', env: ['gauss', 600, 180, 1] }] },
      psi0_A: '60S₁/₂ mJ=−0.5', psi0_B: '60S₁/₂ mJ=−0.5',
      decays_A: [{ from: '60P₃/₂', to: null, gamma_mhz: 0.11 }],
      pair: { R_um: 7.5, theta_deg: 90, phi_deg: 0 },
      vdd_matrix: { nnz: 418 },
      engine_kwargs: { frozen_h_ns: 24, per_freq_M: 3, max_step_hard: 2 },
      warnings: ['两体 V_dd 已预计算；结果将在积分阶段更新']
    }
  },
  {
    id: 'run-20261015-013', status: 'completed', createdAt: '2 分钟前', progress: 100,
    stage: '已完成 · 运行摘要可用', elapsed: '00:17',
    run: {
      version: 1, nbody: 'single', two_body: false, method: 'mixed', method_in: 'mixed',
      t_max: 500, npts: 1001, field: { B_gauss: 1.2, E_vpcm: 0 },
      states: Array.from({ length: 12 }, (_, i) => ({ label: ['60S₁/₂ mJ=−0.5', '60P₃/₂ mJ=+0.5', '59D₃/₂ mJ=+0.5'][i] || `state-${i}` })),
      n_waves: { single: 1 }, waves: { single: [{ name: 'π-pulse', freq_mhz: 6832.4, intensity: 0.21, pol: 'π', env: ['square', 0, 12, 1, 500] }] },
      psi0: '60S₁/₂ mJ=−0.5', decay_index: 0,
      decays: [{ from: '60P₃/₂', to: null, gamma_mhz: 0.12 }],
      engine_kwargs: { max_step: 0.5, phase_budget: 12, tol: 1e-8 }, warnings: []
    },
    result: {
      ok: true, n_t: 1001, n_states: 12,
      top_states_avg_pop: { '60S₁/₂ mJ=−0.5': 0.53, '60P₃/₂ mJ=+0.5': 0.38, '59D₃/₂ mJ=+0.5': 0.04 },
      final_top_pop: { '60S₁/₂ mJ=−0.5': 0.08, '60P₃/₂ mJ=+0.5': 0.87, '59D₃/₂ mJ=+0.5': 0.03 },
      selected_final_pop: { '60P₃/₂': { '60P₃/₂ mJ=+0.5': 0.87 } }
    }
  },
  {
    id: 'run-20261015-012', status: 'completed', createdAt: '11 分钟前', progress: 100,
    stage: '已完成 · 平衡态', elapsed: '00:42',
    run: {
      version: 1, nbody: 'many', two_body: false, equilibrium: true, method: 'dmrg', method_in: 'dmrg',
      n_atoms: 18, dim: 262144, terms: ['I1', 'I2', 'I3'], field: { B_gauss: 0, E_vpcm: 0 },
      states: [{ label: 'g / r 两能级' }], n_waves: {}, engine_kwargs: { chi_max: 240, truncerr: 1e-9, max_sweeps: 45 }, warnings: []
    },
    result: { ok: true, equilibrium: true, want: 'gs', backend: 'dmrg', n_atoms: 18, dim: 262144, E0: -12.483201, gap: 0.08142, chi: 194, S: 1.274, n_atom: [0.08, 0.11, 0.12, 0.09] }
  },
  {
    id: 'run-20261015-011', status: 'failed', createdAt: '28 分钟前', progress: 34,
    stage: '已停止 · 输入需要调整', elapsed: '00:05',
    run: {
      version: 1, nbody: 'single', two_body: false, method: 'hard', method_in: 'hard',
      t_max: 5000, npts: 10001, field: { B_gauss: 0, E_vpcm: 0 },
      states: Array.from({ length: 92 }, (_, i) => ({ label: `state-${i}` })), n_waves: { single: 2 },
      waves: { single: [{ name: 'drive-1', freq_mhz: 6832.4, intensity: 0.8, pol: 'π', env: ['square', 0, 3, 1, 5000] }, { name: 'drive-2', freq_mhz: 7001.2, intensity: 0.1, pol: 'σ+', env: ['gauss', 2200, 500, 1] }] },
      psi0: '61S₁/₂ mJ=−0.5', decays: [], engine_kwargs: { max_step_hard: 0.01 }, warnings: ['基组过大，估算内存超过当前限制']
    },
    error: '资源估算未通过：预计积分步数过高，建议减少基组或缩短 t_max。'
  }
]

function modelName(run) {
  if (run.equilibrium || run.nbody === 'many') return '多体平衡态'
  if (run.two_body || run.nbody === 'pair') return '两体演化'
  return '单体演化'
}

function countStates(run) { return Number(run.n_states || run.states?.length || run.dim || 0) }

function initialStates(run) {
  if (run.psi0) return [{ channel: 'single', value: run.psi0 }]
  if (run.psi0_A || run.psi0_B) return [{ channel: 'A', value: run.psi0_A || '—' }, { channel: 'B', value: run.psi0_B || '—' }]
  if (run.psi0_pair) return [{ channel: 'pair', value: '纠缠叠加态' }]
  if (run.nbody === 'many') return [{ channel: 'many', value: '全基态 |g…g⟩' }]
  return [{ channel: 'single', value: '未指定' }]
}

function readWaves(run) {
  if (run.waves && typeof run.waves === 'object') return Object.entries(run.waves).flatMap(([channel, waves]) => (waves || []).map(wave => ({ channel, ...wave })))
  const config = run.config || {}
  const configured = []
  for (const channel of ['single', 'A', 'B']) for (const wave of config[`dynamic.waves_${channel}`] || []) configured.push({ channel, ...wave })
  if (configured.length) return configured
  const waves = []
  for (const channel of ['single', 'A', 'B']) for (const wave of run[`waves_${channel}`] || []) waves.push({ channel, ...wave })
  return waves
}

function waveEnv(wave) { return Array.isArray(wave.env) ? String(wave.env[0] || 'env') : 'env' }

function levelLabel(group, m) {
  const letters = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k']
  const orbital = letters[Number(group.l)] || `l${group.l}`
  return `${group.n}${orbital}<sub>${String(group.j).replace('.5', '⁄₂')}</sub> mJ=${Number(m) >= 0 ? '+' : ''}${m}`
}

function expandedLevels(run) {
  const states = run.states || []
  if (states.length) return states.map((state, index) => ({ label: state.label || state.qstr || `state-${index}`, group: state.group_label || '', kind: state.kind || 'mJ' }))
  const labels = run.labels || []
  if (labels.length) return labels.map((label, index) => ({ label, group: '', kind: 'mJ' }))
  const config = run.config || {}
  const groups = run.groups || config['groups.list'] || []
  const levels = []
  for (const group of groups) {
    if (Array.isArray(group.ms)) {
      for (const m of group.ms) levels.push({ label: levelLabel(group, m), group: `${group.n}/${group.l}/${group.j}`, kind: 'mJ' })
      continue
    }
    if (Array.isArray(group.Fs)) {
      for (const [F, mF] of group.Fs) levels.push({ label: `${group.n} l=${group.l} j=${group.j} F=${F}, mF=${mF >= 0 ? '+' : ''}${mF}`, group: `${group.n}/${group.l}/${group.j}`, kind: 'F' })
      continue
    }
    const j2 = Math.round(Number(group.j) * 2)
    for (let m2 = -j2; m2 <= j2; m2 += 2) levels.push({ label: levelLabel(group, m2 / 2), group: `${group.n}/${group.l}/${group.j}`, kind: 'mJ' })
  }
  return levels
}

function wavePath(wave, width = 720, height = 116, tMax = 800) {
  const points = []
  const env = Array.isArray(wave.env) ? wave.env : []
  const kind = String(env[0] || '')
  const center = Number(env[1] || tMax * .35), sigma = Math.max(1, Number(env[2] || tMax * .12))
  for (let i = 0; i <= 80; i++) {
    const t = tMax * i / 80
    let amp = 0
    if (kind === 'gauss') amp = Math.exp(-0.5 * ((t - center) / sigma) ** 2) * Number(env[3] || 1)
    else if (kind === 'square') amp = t >= Number(env[1] || 0) && t <= Number(env[4] || tMax) ? Number(env[3] || 1) : 0
    else if (kind === 'file' && Array.isArray(env[2])) amp = Number(env[2][Math.min(env[2].length - 1, Math.floor(i / 80 * env[2].length))] || 0)
    else amp = Math.max(0, Math.sin(Math.PI * i / 80))
    points.push([26 + (width - 40) * i / 80, height - 20 - amp * (height - 38)])
  }
  return points.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
}

function mapRunInput(run) {
  const waves = readWaves(run)
  const config = run.config || {}
  const initial = { ...run }
  if (!initial.psi0 && config['dynamic.psi_single']) initial.psi0 = config['dynamic.psi_single']
  if (!initial.psi0_A && config['dynamic.psi_A']) initial.psi0_A = config['dynamic.psi_A']
  if (!initial.psi0_B && config['dynamic.psi_B']) initial.psi0_B = config['dynamic.psi_B']
  const waveCount = waves.length || Object.values(run.n_waves || {}).reduce((sum, value) => sum + Number(value || 0), 0)
  return {
    model: modelName(run), nbody: run.nbody || (run.two_body ? 'pair' : 'single'), equilibrium: Boolean(run.equilibrium),
    method: run.method || 'unknown', methodIn: run.method_in || run.method || 'unknown', basis: countStates(run),
    tMaxNs: run.t_max ?? null, npts: run.npts ?? null, waves, waveCount,
    levels: expandedLevels(run),
    initialStates: initialStates(initial), decays: (run.decays || run.decays_A || config['dynamic.decays'] || []).length,
    decayMode: run.decay_index === 1 ? '禁用' : run.decay_index === 2 ? '自定义' : run.decays || run.decays_A ? '自动' : '—',
    pair: run.pair || null, atoms: run.n_atoms ?? null, dim: run.dim ?? null, terms: run.terms || [],
    atomSites: run.atoms || config['manybody.atoms'] || [],
    field: run.field || { B_gauss: config['field.B_gauss'], E_vpcm: config['field.E_vpcm'] }, engine: run.engine_kwargs || {}, warnings: run.warnings || [], vddNnz: run.vdd_matrix?.nnz ?? null
  }
}

function mapRunOutput(result) {
  if (!result) return null
  if (result.equilibrium) return { kind: 'equilibrium', backend: result.backend, want: result.want, E0: result.E0, gap: result.gap, chi: result.chi, entropy: result.S, atoms: result.n_atoms, dim: result.dim, nAtom: result.n_atom || [], betas: result.betas || [], E: result.E || [], C: result.C || [], F: result.F || [], energies: result.energies || [], rho: result.rho || [] }
  return { kind: 'dynamic', nT: result.n_t, states: result.n_states, final: result.final_top_pop || {}, average: result.top_states_avg_pop || {}, selected: result.selected_final_pop || {}, series: result.series || result.pops_preview || null }
}

function mapProgress(record) { return { percent: Math.max(0, Math.min(100, Number(record.progress || 0))), stage: record.stage || '排队', elapsed: record.elapsed || '—', eta: record.eta || '' } }
function mapRecord(record) { return { ...record, input: mapRunInput(record.run), output: mapRunOutput(record.result), progressView: mapProgress(record) } }

let records = DEMO_RUNS.map(mapRecord)
let activeFilter = 'all'
let selectedIndex = 0
let liveMode = false

const statusLabel = { running: '进行中', completed: '已完成', failed: '需关注' }
const statusDot = { running: 'live', completed: 'live', failed: '' }
function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value !== 'number') return String(value)
  if (Math.abs(value) >= 1000) return value.toLocaleString('zh-CN', { maximumFractionDigits: digits })
  return value.toFixed(digits).replace(/\.0+$/, '').replace(/(\.[0-9]*?)0+$/, '$1')
}
function escapeHtml(text) { return String(text).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[char])) }
function safeAttr(text) { return escapeHtml(text).replace(/\n/g, ' ') }

function kvBox(label, value) { return `<div class="kv-box"><div class="kv-name">${label}</div><div class="kv-value" title="${safeAttr(value ?? '—')}">${value ?? '—'}</div></div>` }

function renderWaves(input) {
  if (!input.waves.length) return `<div class="pending-box compact"><strong>静态 / Ωδ</strong><span>当前模型没有光场波形</span></div>`
  const colors = ['#6b8cff', '#64d8c1', '#f2c96d', '#ef7e9d', '#b990ed']
  const paths = input.waves.map((wave, index) => `<path d="${wavePath(wave, 720, 116, input.tMaxNs || 800)}" stroke="${colors[index % colors.length]}" fill="none" stroke-width="2"/><text x="${34 + index * 118}" y="14" fill="${colors[index % colors.length]}" font-size="9">${escapeHtml(wave.channel)} · ${escapeHtml(wave.name || `wave${index + 1}`)}</text>`).join('')
  const grid = [0, 1, 2, 3, 4].map(i => `<line x1="26" y1="${18 + i * 19}" x2="706" y2="${18 + i * 19}" stroke="rgba(160,186,226,.12)"/><text x="5" y="${22 + i * 19}" fill="#647795" font-size="8">${(1 - i / 4).toFixed(1)}</text>`).join('')
  const meta = input.waves.map((wave, index) => `<span class="wave-meta"><b style="color:${colors[index % colors.length]}">${escapeHtml(wave.channel)} / ${escapeHtml(wave.name || `wave${index + 1}`)}</b> ${formatNumber(wave.freq_mhz, 2)} MHz · ${wave.pol || '—'} · ${waveEnv(wave)}</span>`).join('')
  return `<div class="wave-plot"><svg viewBox="0 0 720 116" role="img" aria-label="波形包络图">${grid}<line x1="26" y1="96" x2="706" y2="96" stroke="rgba(160,186,226,.3)"/>${paths}<text x="674" y="111" fill="#647795" font-size="8">t (ns)</text></svg><div class="wave-meta-list">${meta}</div></div>`
}

function renderGeometry(input) {
  const sites = input.atomSites || []
  if (!sites.length) return ''
  const width = 720, height = 150, pad = 38
  const xs = sites.map(site => Number(site.pos?.[0] || 0)), ys = sites.map(site => Number(site.pos?.[1] || 0))
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
  const sx = value => pad + (width - pad * 2) * ((value - minX) / Math.max(maxX - minX, 1))
  const sy = value => height - pad + (pad - height + pad) * ((value - minY) / Math.max(maxY - minY, 1))
  const grid = [0, .25, .5, .75, 1].map(t => `<line x1="${pad + (width - pad * 2) * t}" y1="${pad}" x2="${pad + (width - pad * 2) * t}" y2="${height - pad}" stroke="rgba(160,186,226,.1)"/><line x1="${pad}" y1="${pad + (height - pad * 2) * t}" x2="${width - pad}" y2="${pad + (height - pad * 2) * t}" stroke="rgba(160,186,226,.1)"/>`).join('')
  const points = sites.map((site, index) => {
    const x = sx(xs[index]), y = sy(ys[index]), z = Number(site.pos?.[2] || 0)
    return `<circle cx="${x}" cy="${y}" r="7" fill="#6b8cff" stroke="#dbe7ff" stroke-width="1.5"><title>atom${index} · (${xs[index]}, ${ys[index]}, ${z}) μm</title></circle><text x="${x + 10}" y="${y + 3}" fill="#edf4ff" font-size="9">${index}</text>`
  }).join('')
  return `<div class="geometry-plot"><div class="geometry-head"><span>二维点阵 / 原子几何</span><em>${sites.length} sites · XY 投影</em></div><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="多体二维原子点阵">${grid}<line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="rgba(160,186,226,.3)"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="rgba(160,186,226,.3)"/>${points}<text x="${width - 25}" y="${height - 8}" fill="#647795" font-size="9">x</text><text x="8" y="${pad - 8}" fill="#647795" font-size="9">y</text></svg><div class="geometry-caption">范围 x=${formatNumber(minX, 2)}…${formatNumber(maxX, 2)} μm · y=${formatNumber(minY, 2)}…${formatNumber(maxY, 2)} μm</div></div>`
}

function renderInput(input) {
  const field = input.field || {}
  const time = input.tMaxNs === null ? (input.atoms ? `${input.atoms} atoms` : '平衡态') : `${formatNumber(input.tMaxNs, 0)} ns`
  const basis = input.basis ? `${formatNumber(input.basis, 0)} states` : input.dim ? `dim ${formatNumber(input.dim, 0)}` : '—'
  const pair = input.pair ? `R ${formatNumber(input.pair.R_um, 1)} μm · θ ${formatNumber(input.pair.theta_deg, 0)}° · φ ${formatNumber(input.pair.phi_deg, 0)}°` : '—'
  const states = input.initialStates.map(state => `<span class="state-chip"><b>${state.channel}</b>${escapeHtml(state.value)}</span>`).join('')
  const grouped = new Map()
  input.levels.forEach(level => {
    let key = level.group || String(level.label).split(/[, ]/)[0] || '当前基组'
    if (/^state-\d+$/i.test(key)) key = '其他具体能级'
    if (!grouped.has(key)) grouped.set(key, [])
    grouped.get(key).push(level)
  })
  const openGroups = input.levels.length <= 24
  const levelGroups = [...grouped.entries()].map(([name, levels]) => {
    const rows = levels.map((level, index) => `<span class="level-row"><b>${String(index + 1).padStart(2, '0')}</b><span>${level.label}</span><em>${level.kind}</em></span>`).join('')
    return `<details class="level-group-fold"${openGroups ? ' open' : ''}><summary><b>${name}</b><em>${levels.length} 个具体能级</em><span>⌄</span></summary><div class="level-list">${rows}</div></details>`
  }).join('')
  const levelBlock = `<div class="level-groups">${levelGroups || '<span class="level-empty">等待 build_run 展开能级</span>'}</div>`
  const decays = input.decays ? `<div class="decay-row"><span class="decay-chip">${input.decayMode} · ${input.decays} 通道</span>${input.vddNnz ? `<span class="decay-chip">Vdd nnz ${formatNumber(input.vddNnz, 0)}</span>` : ''}</div>` : '<div class="decay-row"><span class="decay-chip muted-chip">未启用耗散</span></div>'
  return `<div class="input-section"><div class="panel-heading"><span class="section-label">输入 / RUNSPEC</span><span class="section-note">build_run 解析后</span></div><div class="input-overview">${kvBox('体系*算法', `${input.model} · ${input.methodIn} → ${input.method}`)}${kvBox('基组 / 维度', basis)}${kvBox('演化时间', time)}${kvBox('时间点', input.npts ? formatNumber(input.npts, 0) : '—')}${kvBox('B / E', `${formatNumber(field.B_gauss, 2)} G · ${formatNumber(field.E_vpcm, 2)} V/cm`)}${kvBox('两体几何', pair)}${kvBox('相互作用', input.terms.join(' + ') || (input.nbody === 'many' ? '模型默认' : '—'))}</div>${input.nbody === 'many' ? renderGeometry(input) : ''}<div class="subsection"><div class="subsection-title">具体能级 · ${input.levels.length} 个</div>${levelBlock}</div><div class="subsection"><div class="subsection-title">驱动波形 · ${input.waveCount} 条</div>${renderWaves(input)}</div><div class="subsection"><div class="subsection-title">初态</div><div class="state-strip">${states}</div></div><div class="subsection"><div class="subsection-title">耗散与相互作用</div>${decays}</div></div>`
}

function resultEntries(values) { return Object.entries(values || {}).slice(0, 4) }
function renderBars(values, color = 'var(--blue)') {
  const entries = resultEntries(values)
  if (!entries.length) return '<span class="result-label">无摘要数据</span>'
  return `<div class="mini-bars">${entries.map(([label, value]) => `<div class="mini-bar-row"><div class="mini-bar-label" title="${safeAttr(label)}">${escapeHtml(label)}</div><div class="mini-bar-number">${formatNumber(value, 3)}</div><div class="mini-bar-track"><i style="width:${Math.min(100, Number(value) * 100)}%;background:${color}"></i></div></div>`).join('')}</div>`
}

function demoSeries(output) {
  const finalEntries = resultEntries(output.final)
  const avgEntries = resultEntries(output.average)
  return finalEntries.map(([label, finalValue], index) => {
    const average = Number(avgEntries[index]?.[1] ?? finalValue)
    const values = Array.from({ length: 72 }, (_, i) => {
      const u = i / 71
      const oscillation = Math.sin(u * Math.PI * (2.4 + index * .35) + index) * (0.07 + index * .015)
      return Math.max(0, Math.min(1, average + (Number(finalValue) - average) * u + oscillation))
    })
    return { label, values }
  })
}

function renderDynamicPlot(output) {
  const series = Array.isArray(output.series) && output.series.length ? output.series : demoSeries(output)
  if (!series.length) return '<div class="pending-box"><strong>无曲线数据</strong><span>引擎未返回可绘制的 pops</span></div>'
  const colors = ['#6b8cff', '#64d8c1', '#f2c96d', '#ef7e9d', '#b990ed', '#8bd3ff']
  const width = 720, height = 210, left = 34, right = 12, top = 17, bottom = 25
  const n = Math.max(...series.map(item => item.values?.length || 0), 2)
  const x = i => left + (width - left - right) * i / (n - 1)
  const y = value => top + (height - top - bottom) * (1 - Math.max(0, Math.min(1, Number(value))))
  const grid = [0, .25, .5, .75, 1].map(value => `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}" stroke="rgba(160,186,226,.12)"/><text x="5" y="${y(value) + 3}" fill="#647795" font-size="9">${value.toFixed(2)}</text>`).join('')
  const paths = series.slice(0, 6).map((item, index) => {
    const values = item.values || []
    const d = values.map((value, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(value).toFixed(1)}`).join(' ')
    return `<path d="${d}" fill="none" stroke="${colors[index % colors.length]}" stroke-width="2"/><text x="${45 + index * 112}" y="12" fill="${colors[index % colors.length]}" font-size="9">${escapeHtml(String(item.label).slice(0, 18))}</text>`
  }).join('')
  return `<div class="result-plot dynamic-plot"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="态布居随时间演化">${grid}<line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="rgba(160,186,226,.3)"/>${paths}<text x="${width - 53}" y="${height - 7}" fill="#647795" font-size="9">t (ns)</text></svg><div class="plot-caption">物理态布居 · ${output.nT || n} 个时间采样点 · 仅显示主要态</div></div>`
}

function renderDynamicResult(output) {
  if (!output) return '<div class="pending-box"><strong>计算中</strong><span>结果摘要将在演化结束后出现</span></div>'
  const entries = resultEntries(output.final)
  return `<div class="result-box"><div class="result-main">终态最高布居 <strong>${formatNumber(entries[0]?.[1] ?? null, 3)}</strong></div>${renderDynamicPlot(output)}</div>`
}

function renderRunningResult(progress) {
  return `<div class="pending-box running-result"><strong>计算中</strong><span>${escapeHtml(progress.stage)} · ${progress.percent}%${progress.eta ? ` · ${escapeHtml(progress.eta)}` : ''}</span><div class="pending-progress-track"><i style="width:${progress.percent}%"></i></div></div>`
}

function renderEquilibriumResult(output) {
  if (!output) return '<div class="pending-box"><strong>求解中</strong><span>平衡态结果尚未返回</span></div>'
  if (output.want === 'thermal' || output.want === 'dos') {
    const values = output.want === 'thermal' ? (output.E || []) : (output.rho || [])
    const maxValue = Math.max(...values.map(Number), 1e-9)
    const points = values.map((value, index) => `${30 + index * (660 / Math.max(values.length - 1, 1))},${145 - Number(value) / maxValue * 105}`).join(' ')
    const plot = values.length ? `<div class="result-plot equilibrium-plot"><svg viewBox="0 0 720 175" role="img" aria-label="平衡态分布图"><line x1="30" y1="145" x2="690" y2="145" stroke="rgba(160,186,226,.3)"/><polyline points="${points}" fill="none" stroke="#64d8c1" stroke-width="2"/><text x="610" y="168" fill="#647795" font-size="9">${output.want === 'thermal' ? 'β' : 'E'}</text></svg><div class="plot-caption">${output.want === 'thermal' ? '热力学分布 · E(β)' : '态密度分布 · ρ(E)'}</div></div>` : '<div class="pending-box compact"><strong>无分布数组</strong><span>当前摘要没有可绘制数据</span></div>'
    return `<div class="result-box"><div class="result-main">${output.want === 'thermal' ? '有限温分布' : '态密度分布'} <strong>${output.backend || '—'}</strong></div>${plot}</div>`
  }
  const values = output.nAtom || []
  const bars = values.map((value, index) => `<rect x="${40 + index * 25}" y="${150 - Number(value) * 105}" width="15" height="${Number(value) * 105}" rx="2" fill="#6b8cff"/><text x="${47 + index * 25}" y="168" text-anchor="middle" fill="#647795" font-size="8">${index}</text>`).join('')
  const plot = values.length ? `<div class="result-plot equilibrium-plot"><svg viewBox="0 0 720 185" role="img" aria-label="平衡态逐原子占据分布"><line x1="34" y1="150" x2="706" y2="150" stroke="rgba(160,186,226,.3)"/><line x1="34" y1="45" x2="706" y2="45" stroke="rgba(160,186,226,.12)"/><text x="8" y="48" fill="#647795" font-size="9">1.0</text>${bars}<text x="610" y="180" fill="#647795" font-size="9">atom index</text></svg><div class="plot-caption">基态逐原子占据分布 · ${values.length} 个原子</div></div>` : '<div class="pending-box compact"><strong>无分布数组</strong><span>当前摘要没有 n_atom</span></div>'
  return `<div class="result-box"><div class="result-main">基态能量 <strong>${formatNumber(output.E0, 4)}</strong></div><div class="equilibrium-metrics">${kvBox('gap', formatNumber(output.gap, 4))}${kvBox('χ', formatNumber(output.chi, 0))}${kvBox('纠缠熵 S', formatNumber(output.entropy, 3))}${kvBox('backend', output.backend)}</div>${plot}</div>`
}

function renderDiagnostics(input) {
  const entries = Object.entries(input.engine || {}).slice(0, 5)
  const chips = entries.map(([key, value]) => `<span class="diag-chip">${key} <b>${typeof value === 'object' ? JSON.stringify(value) : formatNumber(value, 3)}</b></span>`).join('')
  return chips ? `<div class="engine-diagnostics">${chips}</div>` : ''
}

function renderCard(record, depth) {
  const input = record.input, progress = record.progressView, status = record.status
  const warning = input.warnings?.[0]
  const outputMarkup = status === 'running' ? renderRunningResult(progress) : status === 'failed' ? `<div class="error-box">${escapeHtml(record.error || '运行失败')}<small>输入仍已保留，AI 可以基于这张卡片调整后重试。</small></div>` : input.equilibrium ? renderEquilibriumResult(record.output) : renderDynamicResult(record.output)
  const detailPayload = { run: record.run, result: record.result || null, progress: record.progress }
  return `<article class="deck-card ${status} depth-${depth < 4 ? depth : 'hidden'}" data-id="${record.id}" data-depth="${depth}"><div class="card-inner"><div class="card-head"><div><div class="run-id-line"><span class="run-id">${record.id}</span><span class="run-time">${record.createdAt}</span></div><div class="card-subtitle">${status === 'running' ? progress.stage : record.stage}</div><div class="tag-row">${input.equilibrium ? '<span class="tag">equilibrium</span>' : ''}${warning ? '<span class="tag warn">有提醒</span>' : ''}</div></div><div class="status-pill ${status}"><span class="status-dot ${statusDot[status]}"></span>${statusLabel[status]}</div></div><div class="card-grid">${renderInput(input)}<div class="result-panel"><div class="panel-heading"><span class="section-label">输出 / RESULT SUMMARY</span><span class="section-note">${input.equilibrium ? '平衡态摘要' : '演化摘要'}</span></div>${outputMarkup}${status !== 'running' ? renderDiagnostics(input) : ''}${warning ? `<div class="warning-callout">⚠ ${escapeHtml(warning)}</div>` : ''}</div></div><div class="details-toggle"><button class="text-button details-button">展开原始映射 ↗</button></div><div class="details"><div class="details-grid"><div><h4>RunSpec 输入</h4><pre>${escapeHtml(JSON.stringify(record.run, null, 2))}</pre></div><div><h4>运行输出</h4><pre>${escapeHtml(JSON.stringify(record.result || { status: record.status, error: record.error }, null, 2))}</pre></div></div></div></div></article>`
}

function filteredRecords() {
  const query = ($('#search').value || '').trim().toLowerCase(), sort = $('#sort-select').value
  let list = records.filter(record => {
    const matchesFilter = activeFilter === 'all' || (activeFilter === 'running' && record.status === 'running') || (activeFilter === 'completed' && record.status === 'completed') || (activeFilter === 'attention' && record.status === 'failed')
    const haystack = `${record.id} ${record.input.model} ${record.input.method} ${record.input.initialStates.map(state => state.value).join(' ')}`.toLowerCase()
    return matchesFilter && (!query || haystack.includes(query))
  })
  if (sort === 'oldest') list = [...list].reverse()
  if (sort === 'status') list.sort((a, b) => ({ running: 0, failed: 1, completed: 2 }[a.status] - ({ running: 0, failed: 1, completed: 2 }[b.status])))
  return list
}

function updateCounts() {
  const total = records.length, running = records.filter(record => record.status === 'running').length, completed = records.filter(record => record.status === 'completed').length, attention = records.filter(record => record.status === 'failed').length
  $('#nav-run-count').textContent = total; $('#count-all').textContent = total; $('#count-running').textContent = running; $('#count-completed').textContent = completed; $('#count-attention').textContent = attention; $('#last-sync').textContent = '刚刚'
}

function renderDeck() {
  updateCounts()
  const list = filteredRecords()
  if (selectedIndex >= list.length) selectedIndex = Math.max(0, list.length - 1)
  const deck = $('#run-deck'); deck.innerHTML = list.map((record, index) => renderCard(record, index - selectedIndex)).join('')
  $('#empty-state').hidden = list.length > 0; $('#deck-position-total').textContent = String(list.length).padStart(2, '0'); $('#deck-position-current').textContent = list.length ? String(selectedIndex + 1).padStart(2, '0') : '00'
  $('#deck-prev').disabled = !list.length || selectedIndex <= 0; $('#deck-next').disabled = !list.length || selectedIndex >= list.length - 1
  $$('.deck-card').forEach(card => {
    const depth = Number(card.dataset.depth)
    card.addEventListener('click', event => { if (event.target.closest('button')) return; if (depth > 0) { selectedIndex += depth; renderDeck() } })
  })
  $$('.details-button').forEach(button => button.addEventListener('click', () => button.closest('.deck-card').querySelector('.details').classList.toggle('open')))
}

function moveDeck(delta) { const list = filteredRecords(); if (!list.length) return; selectedIndex = Math.max(0, Math.min(list.length - 1, selectedIndex + delta)); renderDeck() }

$('#search').addEventListener('input', () => { selectedIndex = 0; renderDeck() })
$('#sort-select').addEventListener('change', () => { selectedIndex = 0; renderDeck() })
$$('.filter-tab').forEach(tab => tab.addEventListener('click', () => { activeFilter = tab.dataset.filter; selectedIndex = 0; $$('.filter-tab').forEach(item => item.classList.toggle('active', item === tab)); renderDeck() }))
$('#deck-prev').addEventListener('click', () => moveDeck(-1)); $('#deck-next').addEventListener('click', () => moveDeck(1))
document.addEventListener('keydown', event => { if (event.key === 'ArrowUp') moveDeck(-1); if (event.key === 'ArrowDown') moveDeck(1) })
$('#deck-stage').addEventListener('wheel', event => { if (Math.abs(event.deltaY) > 15) { event.preventDefault(); moveDeck(event.deltaY > 0 ? 1 : -1) } }, { passive: false })

function startDemoTicker() {
  clearInterval(window.__demoTicker)
  window.__demoTicker = setInterval(() => {
    const running = records.find(record => record.status === 'running'); if (!running) return
    running.progress = Math.min(100, running.progress + Math.ceil(Math.random() * 4)); running.progressView = mapProgress(running)
    if (running.progress >= 100) {
      running.status = 'completed'; running.stage = '已完成 · 运行摘要可用'; running.eta = ''; running.progressView = mapProgress(running); running.result = { ok: true, n_t: 2401, n_states: 54, top_states_avg_pop: { '60S₁/₂ mJ=−0.5': .48, '60P₃/₂ mJ=+0.5': .42 }, final_top_pop: { '60S₁/₂ mJ=−0.5': .12, '60P₃/₂ mJ=+0.5': .81 } }; running.output = mapRunOutput(running.result)
    }
    renderDeck()
  }, 1500)
}

async function loadLiveRuns() {
  try {
    const response = await fetch('/api/runs', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    if (!payload || !Array.isArray(payload.runs)) throw new Error('bad run payload')
    liveMode = true
    records = payload.runs.map(mapRecord)
    selectedIndex = Math.min(selectedIndex, Math.max(0, records.length - 1))
    renderDeck()
  } catch (_error) {
    if (!liveMode) startDemoTicker()
  }
}

renderDeck(); loadLiveRuns(); setInterval(() => { if (liveMode) loadLiveRuns() }, 1500)
window.energyLevelRunMonitor = { mapRunInput, mapRunOutput, mapProgress, mapRecord }
