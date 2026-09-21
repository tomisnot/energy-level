/* Re0 机甲驾驶舱前端 · 只读投影
 * 数据源：/api/*（读事件总线 journal）；实时用 SSE /api/stream（diff-since-seq）。
 * 无任何写回：前端只 GET/订阅，物理上无写通道（I2）。
 */
(function () {
  'use strict';
  // 远程只读鉴权：token 从 URL 查询带进来（?token=...），附加到每个请求
  const TOKEN = new URLSearchParams(location.search).get('token') || '';
  const q = (p) => p + (TOKEN ? (p.includes('?') ? '&' : '?') + 'token=' +
    encodeURIComponent(TOKEN) : '');

  const $ = (id) => document.getElementById(id);
  let events = [];           // 全量事件（按 seq）
  let lastSeq = 0;
  let alertSeqs = new Set();

  // ---------------- 渲染 ----------------
  function fmtVal(v) {
    if (v === null || v === undefined) return '∅';
    if (typeof v === 'object') {
      if (v.$pointer) return '⟨数组 ' + (v.shape || []).join('×') + '⟩';
      if (v.$truncated) return '⟨截断 ' + v.kept + '/' + v.total + '⟩';
      return JSON.stringify(v);
    }
    return String(v);
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  function renderRecorder() {
    const filter = $('filter').value.trim().toLowerCase();
    const aiOnly = $('ai-only').checked;
    const list = $('events');
    list.innerHTML = '';
    const rows = events.slice().reverse();   // 最新在上
    for (const ev of rows) {
      const isAI = /ai|agent|harness/i.test(ev.actor || '');
      if (aiOnly && !isAI) continue;
      if (filter) {
        const hay = [ev.actor, ev.kind, ev.target, ev.reason,
          fmtVal(ev.before), fmtVal(ev.after)].join(' ').toLowerCase();
        if (!hay.includes(filter)) continue;
      }
      // 逐行折叠：summary=概括(seq/actor/kind/reason)，展开=全量多行
      const d = document.createElement('details');
      d.className = 'ev' + (isAI ? ' ai' : '') +
        (ev.kind === 'run' ? ' run' : '') +
        (alertSeqs.has(ev.seq) ? ' alerted' : '');
      const sum = document.createElement('summary');
      sum.innerHTML =
        '<span class="seq">#' + ev.seq + '</span>' +
        '<span class="actor">' + esc(ev.actor) + '</span>' +
        '<span class="tag ' + esc(ev.kind) + '">' + esc(ev.kind) + '</span>' +
        '<span class="reason" title="' + esc(ev.reason || '') + '">' +
          esc(ev.reason || '') + '</span>';
      const body = document.createElement('div');
      body.className = 'ev-body';
      body.innerHTML =
        '<div class="kv"><span>target</span><b>' + esc(ev.target) + '</b></div>' +
        '<div class="kv"><span>before → after</span><b class="ba">' +
          esc(fmtVal(ev.before)) + '<span class="arrow">→</span>' +
          esc(fmtVal(ev.after)) + '</b></div>' +
        '<div class="kv"><span>reason</span><b>' + esc(ev.reason || '') + '</b></div>' +
        '<div class="kv"><span>可逆</span><b>' + esc(ev.reversibility || '') + '</b></div>';
      d.appendChild(sum);
      d.appendChild(body);
      list.appendChild(d);
    }
    $('n-events').textContent = events.length + ' 事件';
  }

  function renderAlerts(alerts, level) {
    const ul = $('alert-list');
    ul.innerHTML = '';
    alertSeqs = new Set((alerts || []).map((a) => a.seq));
    for (const a of (alerts || [])) {
      const li = document.createElement('li');
      li.className = a.level;
      li.innerHTML = '<div class="rule">' + esc(a.rule) + ' · seq ' + a.seq +
        '</div>' + esc(a.message);
      ul.appendChild(li);
    }
    $('alert-count').textContent = (alerts || []).length;
    $('alert-badge').className = 'alert-badge ' + (level || 'ok');
    $('alerts-empty').style.display = (alerts || []).length ? 'none' : '';
    const light = $('light-alert');
    light.className = 'light ' + (level || 'ok');
    light.querySelector('.lbl').textContent =
      level === 'error' ? '异常' : level === 'warn' ? '注意' : '平静';
  }

  function renderHUD(o) {
    $('last-seq').textContent = o.last_seq != null ? o.last_seq : lastSeq;
    $('hud-nruns').textContent = o.n_runs != null ? o.n_runs : '—';
    $('hud-running').textContent = o.last_run || '空闲';
    $('hud-method').textContent = (o.config_head &&
      o.config_head['dynamic.method']) || o.method || '—';
  }

  function addEvent(ev) {
    if (ev && ev.seq != null) {
      if (ev.seq > lastSeq) lastSeq = ev.seq;
      if (!events.some((e) => e.seq === ev.seq)) events.push(ev);
    }
  }

  // ---------------- 配置态结构树（需求②：schema join 现值） ----------------
  let treeData = null;
  function renderConfigTree() {
    const body = $('tree-body');
    if (!body) return;
    const filter = ($('tree-filter').value || '').trim().toLowerCase();
    const showEngine = $('tree-show-engine').checked;
    body.innerHTML = '';
    if (!treeData || !treeData.has_schema) {
      $('tree-meta').textContent = '尚无 schema（Hub 未落 registry_schema 事件，或未起）。';
      body.innerHTML = '<p class="muted">配置态树需 Hub 启动时落一条 schema 事件（python -m src.adapter.serve）。</p>';
      return;
    }
    const fams = treeData.families || {};
    $('tree-meta').textContent = treeData.n_keys + ' 键 · ' +
      Object.keys(fams).length + ' 族 · as of seq ' + treeData.last_seq;
    for (const fam of Object.keys(fams)) {
      if (fam === 'engine' && !showEngine) continue;   // engine.* 默认折叠（只读常量，别干扰）
      const rows = fams[fam] || [];
      const group = document.createElement('div');
      group.className = 'tree-fam';
      const h = document.createElement('h3');
      h.innerHTML = esc(fam) + ' <span class="badge">' + rows.length + '</span>' +
        (fam === 'engine' ? ' <span class="ro">只读常量</span>' : '');
      group.appendChild(h);
      const tbl = document.createElement('table');
      tbl.className = 'tree';
      for (const r of rows) {
        const hay = [r.name, r.desc, r.unit, fmtVal(r.value)].join(' ').toLowerCase();
        if (filter && !hay.includes(filter)) continue;
        const set = r.set;
        const tr = document.createElement('tr');
        if (set) tr.className += ' set';
        if (r.readonly) tr.className += ' ro';
        tr.innerHTML =
          '<td class="k" title="' + esc(r.name) + '">' + esc(r.name) + '</td>' +
          '<td class="v" title="' +
            esc(fmtVal(r.value) +
              (r.unit ? ' ' + r.unit : '') +
              (r.type ? ' · ' + r.type : '') +
              (r.desc ? ' · ' + r.desc : '') +
              (set ? ' · ' + set.actor + ' · ' + set.reason + ' · #' + set.seq : '')) +
            '">' + esc(fmtVal(r.value)) +
            (r.unit ? ' <span class="u">' + esc(r.unit) + '</span>' : '') + '</td>';
        tbl.appendChild(tr);
      }
      group.appendChild(tbl);
      body.appendChild(group);
    }
    if ((treeData.orphans || []).length) {
      const g = document.createElement('div');
      g.className = 'tree-fam orphan';
      g.innerHTML = '<h3>orphans <span class="badge">' + treeData.orphans.length +
        '</span> <span class="ro">契约外/已移除</span></h3>';
      const ul = document.createElement('ul');
      for (const o of treeData.orphans) {
        const li = document.createElement('li');
        li.textContent = o.name + ' = ' + fmtVal(o.value);
        ul.appendChild(li);
      }
      g.appendChild(ul);
      body.appendChild(g);
    }
  }

  async function refreshConfigTree() {
    try {
      const r = await fetch(q('/api/config-tree'));
      treeData = await r.json();
      renderConfigTree();
    } catch (e) { /* 静默：配置树刷新失败不打断记录 */ }
  }

  async function refreshAlerts() {
    try {
      const r = await fetch(q('/api/alerts'));
      const d = await r.json();
      renderAlerts(d.alerts, d.level);
      if (d.as_of_seq != null) $('last-seq').textContent = d.as_of_seq;
    } catch (e) { /* 静默：告警刷新失败不打断记录 */ }
  }

  // ---------------- 初始全量 + 定位包 ----------------
  async function bootstrap() {
    const [o, h] = await Promise.all([
      fetch(q('/api/orientation')).then((r) => r.json()),
      fetch(q('/api/history?since_seq=0')).then((r) => r.json()),
    ]);
    events = h.events || [];
    lastSeq = h.last_seq || 0;
    renderHUD(o);
    renderRecorder();
    await refreshAlerts();
    await refreshConfigTree();
    return lastSeq;
  }

  // ---------------- SSE：diff-since-seq 增量 ----------------
  let es = null;
  function connectStream(since) {
    if (es) es.close();
    es = new EventSource(q('/api/stream?since_seq=' + since));
    es.onopen = () => {
      $('light-conn').className = 'light ok';
      $('light-conn').querySelector('.lbl').textContent = '实时';
    };
    es.onmessage = (m) => {
      try {
        const ev = JSON.parse(m.data);
        addEvent(ev);
        $('last-seq').textContent = lastSeq;
        renderRecorder();
        if (ev.kind === 'run' || ev.kind === 'error') refreshAlerts();
        if (ev.kind === 'set' || ev.kind === 'registry_schema') refreshConfigTree();
      } catch (e) { /* 忽略半帧 */ }
    };
    es.onerror = () => {
      $('light-conn').className = 'light warn';
      $('light-conn').querySelector('.lbl').textContent = '重连中';
      // 项目可能死了：EventSource 自动重连；黑匣子仍能读死前全史（journal 在盘上）
    };
  }

  // ---------------- 控件 ----------------
  $('filter').addEventListener('input', renderRecorder);
  $('ai-only').addEventListener('change', renderRecorder);
  $('reload').addEventListener('click', async () => {
    await bootstrap();
    connectStream(lastSeq);
  });
  $('tree-filter').addEventListener('input', renderConfigTree);
  $('tree-show-engine').addEventListener('change', renderConfigTree);
  $('tree-reload').addEventListener('click', refreshConfigTree);
  // 告警角落徽标：点击展开/收起弹层
  $('alert-badge').addEventListener('click', () => {
    const pop = $('alert-pop');
    pop.hidden = !pop.hidden;
  });
  // 两页签切换：飞行记录仪 / 配置态
  document.querySelectorAll('#cockpit-tabs .tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-tab');
      document.querySelectorAll('#cockpit-tabs .tab').forEach((b) => {
        b.classList.toggle('active', b === btn);
      });
      $('recorder').classList.toggle('active', target === 'recorder');
      $('config-tree').classList.toggle('active', target === 'config');
    });
  });

  // ---------------- 启动 ----------------
  bootstrap().then((since) => connectStream(since)).catch(() => {
    $('light-conn').className = 'light error';
    $('light-conn').querySelector('.lbl').textContent = '连不上';
  });
})();
