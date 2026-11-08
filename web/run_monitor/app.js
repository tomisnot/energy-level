import {ResultView} from './visualization.js'
import {escape,fmt,finite,canvasContext,projection,bestPlane,drawLines,palette} from './plot.js'
const $=s=>document.querySelector(s)
const statusNames={running:'计算中',completed:'已完成',failed:'运行失败',cancelled:'已取消'}
const results=new Map(),inflight=new Map(),saved=new Map(),inputState=new Map(),models=new Map()
let records=[],selected=null,filter='all',page='runs',widget=null,signature='',lastWheel=0,connected=false,renderToken=0,renderedId=null
const bodyName=r=>r.nbody==='many'?(r.equilibrium?'多体平衡态':'多体演化'):r.two_body||r.nbody==='pair'?'两体演化':'单体演化'
const filtered=()=>records.filter(r=>(filter==='all'||filter==='attention'?filter==='all'||['failed','cancelled'].includes(r.status):r.status===filter)&&`${r.id} ${r.reason||''} ${r.run.method||''} ${r.run.want||''} ${((r.run.setup||{}).intent||{}).label||''}`.toLowerCase().includes($('#search').value.trim().toLowerCase()))
function dispose(){if(widget){saved.set(renderedId,{...widget.state,playing:false});widget.destroy();widget=null}}
function setConnection(text,mode){$('#connection').textContent=text;$('#connection').className='connection '+mode}
function pair(label,value,wide=false){return `<div${wide?' class="wide"':''}><dt>${escape(label)}</dt><dd>${escape(value??'—')}</dd></div>`}
function fraction(v){if(!finite(Number(v)))return String(v);const x=Number(v);return Number.isInteger(x)?String(x):`${Math.round(x*2)}/2`}
function level(s){if(!Array.isArray(s))return String(s);return `${s[0]}${['S','P','D','F','G','H'][s[1]]||(' l='+s[1])} ${fraction(s[2])} · m=${fraction(s[3])}`}
function levels(run){
  const grouped=new Map(),seen=new Set()
  if(run.nbody==='many'){
    for(const atom of run.atoms||[])for(const role of ['g','r']){const q=atom[role];if(!q)continue;const key=JSON.stringify(q);if(seen.has(key))continue;seen.add(key);const group=Array.isArray(q)?`${q[0]} / l=${q[1]} / j=${fraction(q[2])}`:'局部能级';if(!grouped.has(group))grouped.set(group,[]);grouped.get(group).push(`${role} · ${level(q)}`)}
  }else for(const s of run.states||[]){const group=s.group_label||(finite(s.n)?`${s.n} / l=${s.l} / j=${fraction(s.j)}`:'具体能级');if(!grouped.has(group))grouped.set(group,[]);grouped.get(group).push(s.label||s.qstr||'未命名态')}
  const total=[...grouped.values()].reduce((sum,values)=>sum+values.length,0)
  return [...grouped].map(([name,values])=>`<details data-group="${escape(name)}"${total<16?' open':''}><summary>${escape(name)} · ${values.length}</summary><div class="level-rows">${values.map(s=>`<span>${escape(s)}</span>`).join('')}</div></details>`).join('')||'<div class="small-note">没有返回局部能级身份</div>'
}
function initial(run){const value=run.psi0??run.config?.['manybody.psi0'];if(value==null)return run.nbody==='many'?'|g…g⟩':'未指定';if(typeof value==='object')return value.kind?`${value.kind}${value.backend?' / '+value.backend:''}`:JSON.stringify(value).slice(0,100);return String(value)}
// ---- 装配：`manybody.build` 是构造器**真正用掉的入参**；回执由 calib 现算 ----
const buildOf=run=>(run&&run.build)||{}
const SOURCE_LABEL={arc:'ARC 标定',file:'文件'}
function chip(text,cls=''){return `<span class="chip ${cls}">${escape(String(text))}</span>`}
function qchip(label,value,state){return `<span class="q ${state||''}">${escape(label)} ${escape(String(value))}</span>`}
function runTitle(r){const label=((r.run.label)||'').trim();const reason=(r.reason||'').trim();return label||((reason&&reason!=='AI 运行')?reason:null)||r.run.want||r.run.method||'准备配置'}
function captionMarkup(run,record){
  const label=((run.label)||'').trim(),reason=((record&&record.reason)||'').trim()
  const b=buildOf(run),source=SOURCE_LABEL[b.source]||null
  const titleText=label||((reason&&reason!=='AI 运行')?reason:null)
  if(!titleText&&!source)return ''
  const title=titleText?`<div class="title">${escape(titleText)}</div>`:''
  return `<div class="caption">${title}${source?`<div class="chips">${chip(source,'accent')}${b.file?chip(b.file):''}</div>`:''}</div>`
}
function buildMarkup(run){
  const b=buildOf(run),r=run.build_receipt||{},a=b.arc||{},parts=[]
  if(a.B_gauss!=null)parts.push(qchip('B',fmt(a.B_gauss,3)+' G'))
  if(a.r_cut_um!=null)parts.push(qchip('r_cut',fmt(a.r_cut_um,3)+' µm'))
  if(a.n_window!=null)parts.push(qchip('窗口',a.n_window))
  if(r.tail_norm_bound!=null){const t=Number(r.tail_norm_bound);parts.push(qchip('尾界',fmt(t,2),t<1e-6?'ok':t<1e-4?'warn':'bad'))}
  if(r.n_dropped!=null)parts.push(qchip('丢对',r.n_dropped,r.n_dropped?'warn':'ok'))
  if(r.n_guarded!=null)parts.push(qchip('剔除',r.n_guarded,r.n_guarded?'warn':'ok'))
  if(!parts.length&&!b.source)return{row:'',details:''}
  const rows=[]
  if(b.source)rows.push(['来源',SOURCE_LABEL[b.source]||b.source])
  if(b.file)rows.push(['文件',String(b.file)])
  if(Array.isArray(b.atoms)&&b.atoms.length){const one=b.atoms[0];rows.push(['能级',`${level(one.g)} → ${level(one.r)}（×${b.atoms.length}）`])}
  const wins=[a.n_window!=null?`窗口 ${a.n_window}`:'',a.second_order!=null?`二阶${a.second_order?'开':'关'}`:'',a.guard_mhz!=null?`guard ${fmt(a.guard_mhz,3)} MHz`:'',a.periods_um?`周期 ${escape(JSON.stringify(a.periods_um))}`:''].filter(Boolean)
  if(wins.length)rows.push(['标定',wins.join(' · ')])
  if(run.notes)rows.push(['备注',String(run.notes)])
  if(r.fingerprint)rows.push(['回执',`${r.fingerprint} · ${r.n_atoms??'—'} 原子 · ${r.n_pairs??'—'} 对 · Σ|J| ${fmt(r.sum_abs_J,3)}`])
  const grid=rows.map(([k,v])=>`<div>${escape(k)}</div><div>${escape(v)}</div>`).join('')
  return{row:`<div class="wide"><dt>装配</dt><dd class="chips">${parts.join('')}</dd></div>`,
         details:`<details class="setup-block"><summary>装配（入参与回执）</summary><div class="grid">${grid}</div></details>`}
}
function inputMarkup(run,record){
  const cfg=run.config||{},many=run.nbody==='many',drives=run.drives_preview||[]
  const cs=run.calib_summary||{}
  const model=many?`calib · ${cs.n_atoms??run.atoms?.length??'—'} 原子 · ${cs.n_pairs??'—'} 对${cs.fingerprint?' · 指纹 '+escape(cs.fingerprint):''}`:null
  const levelBlock=`<h3>具体能级 · 可按能级组折叠</h3>${levels(run)}`
  const build=buildMarkup(run)
  const hBlock=many?`<h3>哈密顿量</h3><div class="model-formula"><code>载入中…</code></div>`:''
  const geo=`${many?`<h3>原子几何 · 系数视图</h3><div class="geo-tools"><label>原子 <select class="atom-quantity" disabled><option value="delta">δ</option><option value="omega">Ω</option><option value="initial">初态 ⟨σᶻ⟩</option><option value="none">不显示</option></select></label><label>耦合 <select class="pair-quantity" disabled><option value="J">J</option><option value="V">V</option><option value="none">不显示</option></select></label></div><canvas class="input-geometry" role="img" aria-label="原子几何与系数"></canvas><div class="geo-legend"></div>`:''}<div class="geo-readout"></div>`
  return `${captionMarkup(run,record)}<div class="section-label">输入 / RUNSPEC</div><dl class="params">${pair('体系 × 方法',bodyName(run)+' · '+(run.method||'准备中'),true)}${pair('请求',run.want||'含时演化')}${many?pair('模型',model,true):''}${build.row}${pair(many?'有效维度':'能级数',many?`${run.dim??'—'}`:run.states?.length)}${pair('物理时间',finite(run.t_max)?fmt(run.t_max)+' ns':'静态')}${finite(run.npts)?pair('采样点',run.npts):''}${run.field?pair('磁场 B',fmt(run.field.B_gauss)+' G'):''}${pair('初态',run.psi0_label??initial(run),true)}${run.plan_summary?.kind&&run.plan_summary.kind!=='none'?pair('约束',run.plan_summary.kind+' · '+run.plan_summary.mode,true):''}${run.drive?.kind==='segments'?pair('分段驱动',`${run.drive.n_segments} 段 · t0 ${run.drive.segments.map(s=>fmt(s.t0)).join(' / ')}${run.drive.interp==='midpoint'?' · 中点':''}`,true):''}${run.observables?.pauli_count?pair('Pauli 串',`${run.observables.pauli_count} 条`):''}</dl>${hBlock}${geo}${many?'':levelBlock}${run.equilibrium?'':`<h3>${many?'驱动 Ω / δ':'波形光强包络'}</h3>${drives.length?'<canvas class="drive-mini" role="img" aria-label="真实输入波形"></canvas><div class="small-note">'+escape(drives.slice(0,3).map(d=>d.label).join(' · '))+(drives.length>3?' …':'')+'</div>':'<div class="small-note">'+escape(run.drive_preview_note||'静态驱动 / 无光场波形')+'</div>'}`}${build.details}<h3>耗散与配置提醒</h3><div class="small-note">${escape((run.warnings||[]).join('；')||((run.decays||run.decays_A||[]).length+' 条显式衰减通道'))}</div><details><summary>输入详情</summary><pre class="small-note">${escape(JSON.stringify({calib_summary:cs,build:run.build||null,build_receipt:run.build_receipt||null,engine_kwargs:run.engine_kwargs||{}},null,2))}</pre></details>`
}
// ---- 哈密顿量可视化：单原子项（节点）+ 两原子项（边）同时画，各自独立选择 ----
let currentModel=null,atomMode='delta',pairMode='J',geoRun=null
let geoZoom=1,geoPanX=0,geoPanY=0,geoDragging=false,geoDragStart=null,geoMoved=false
const POS_LO=[148,163,184],POS_HI=[15,118,110],NEG_LO=[148,163,184],NEG_HI=[194,65,12]
function rampColor(v,vmax){
  const t=Math.min(1,Math.abs(Number(v)||0)/Math.max(vmax,1e-12))
  const lo=v>=0?POS_LO:NEG_LO,hi=v>=0?POS_HI:NEG_HI
  const c=lo.map((x,i)=>Math.round(x+(hi[i]-x)*t))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}
function paintGeometry(run){
  geoRun=run||{}
  const c=$('.input-geometry');if(!c)return
  const b=canvasContext(c);if(!b)return
  const sites=(run&&run.atoms)||[],{ctx,w,h}=b
  if(!sites.length)return
  const p=projection(sites,bestPlane(sites),w,h,1)
  const cx=w/2,cy=h/2
  p.positions=p.positions.map(pt=>pt?[(pt[0]-cx)*geoZoom+cx+geoPanX,(pt[1]-cy)*geoZoom+cy+geoPanY]:null)
  const model=currentModel
  const showPair=!!model&&pairMode!=='none',showAtom=!!model&&atomMode!=='none'
  let pairMax=1e-12
  if(showPair&&(model.pairs||[]).length){
    const key=pairMode==='J'?2:3
    const vals=model.pairs.map(pr=>Math.abs(Number(pr[key])||0))
    pairMax=Math.max(...vals,pairMax)
    const order=vals.map((v,k)=>k).sort((a,b)=>vals[b]-vals[a]).slice(0,90)
    for(const k of order){
      const pr=model.pairs[k],a=p.positions[pr[0]],bb=p.positions[pr[1]]
      if(!a||!bb)continue
      const v=Number(pr[key])||0,t=Math.min(1,Math.abs(v)/pairMax)
      ctx.strokeStyle=rampColor(v,pairMax)
      ctx.lineWidth=0.4+3.4*t
      ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(bb[0],bb[1]);ctx.stroke()
    }
  }
  const siteVals=showAtom?(atomMode==='initial'?model.initial_sz:atomMode==='delta'?model.delta:model.omega):null
  const vmax=siteVals?(atomMode==='initial'?1:Math.max(...siteVals.map(v=>Math.abs(Number(v)||0)),1e-12)):0
  p.positions.forEach((point,i)=>{if(!point)return
    let fill='#0f766e'
    if(siteVals){const v=Number(siteVals[i])||0
      fill=Math.abs(v)<1e-12?'#cbd5e1':rampColor(v,vmax)}
    ctx.fillStyle=fill;ctx.beginPath();ctx.arc(point[0],point[1],showPair?3.4:4.6,0,Math.PI*2);ctx.fill()
    if(siteVals&&sites.length<=36){ctx.fillStyle='#334155';ctx.font='9px sans-serif';ctx.textAlign='center';ctx.fillText(fmt(siteVals[i],2),point[0],point[1]-7)}
    else if(sites.length<=24){ctx.fillStyle='#475569';ctx.font='10px sans-serif';ctx.textAlign='center';ctx.fillText(String(sites[i].site_id??i),point[0],point[1]+15)}
  })
  const lg=$('.geo-legend')
  if(lg){
    const bits=[]
    if(siteVals&&atomMode==='initial')bits.push(`<span><i style="background:${rampColor(1,1)}"></i>+1 · |g⟩</span><span><i style="background:${rampColor(-1,1)}"></i>−1 · |r⟩</span><span>节点颜色 ∝ ⟨σᶻ⟩；0=混合/纠缠</span>`)
    else if(siteVals)bits.push(`<span><i style="background:${rampColor(vmax,vmax)}"></i>原子 ${atomMode==='delta'?'δ':'Ω'} 正</span><span><i style="background:${rampColor(-vmax,vmax)}"></i>负</span><span>节点亮度/颜色 ∝ |值| · max ${fmt(vmax,3)} MHz</span>`)
    else if(model)bits.push(`<span>原子项：未显示（上方选 δ/Ω）</span>`)
    if(showPair)bits.push(`<span><i style="background:${rampColor(pairMax,pairMax)}"></i>${pairMode}&gt;0</span><span><i style="background:${rampColor(-pairMax,pairMax)}"></i>${pairMode}&lt;0</span><span>线宽与亮度 ∝ |${pairMode}| · 最多 90 条</span>`)
    else if(model)bits.push(`<span>耦合项：未显示（上方选 J/V）</span>`)
    lg.innerHTML=bits.length?bits.join(''):`<span>节点 = 原子编号；模型载入后可按系数着色</span>`
  }
  c.style.cursor='crosshair'
  c.title='滚轮以鼠标为中心缩放 · 拖动平移 · 双击复位'
  c.onwheel=e=>{e.preventDefault();e.stopPropagation()
    const rect=c.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top
    const factor=Math.exp(-Number(e.deltaY||0)*.0015),z2=Math.max(.35,Math.min(8,geoZoom*factor))
    const mxC=rect.width/2,myC=rect.height/2
    geoPanX=(mx-mxC)-(mx-mxC-geoPanX)*(z2/geoZoom)
    geoPanY=(my-myC)-(my-myC-geoPanY)*(z2/geoZoom)
    geoZoom=z2;paintGeometry(geoRun)}
  c.ondblclick=()=>{geoZoom=1;geoPanX=0;geoPanY=0;paintGeometry(geoRun)}
  c.onmousedown=e=>{if(e.button!==0)return;e.preventDefault();geoDragging=true;geoMoved=false;geoDragStart=[e.clientX,e.clientY,geoPanX,geoPanY];c.style.cursor='grabbing'}
  c.onmousemove=e=>{
    if(geoDragging){const dx=e.clientX-geoDragStart[0],dy=e.clientY-geoDragStart[1]
      if(Math.abs(dx)+Math.abs(dy)>2)geoMoved=true
      geoPanX=geoDragStart[2]+dx;geoPanY=geoDragStart[3]+dy;paintGeometry(geoRun);return}
    const rect=c.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top
    let best=-1,bestD=64
    p.positions.forEach((point,i)=>{if(!point)return;const d=(point[0]-x)**2+(point[1]-y)**2;if(d<bestD){bestD=d;best=i}})
    const out=$('.geo-readout');if(!out)return
    if(best<0){out.textContent='';return}
    const s=sites[best]||{},dv=model&&model.delta?Number(model.delta[best])||0:null,ov=model&&model.omega?Number(model.omega[best])||0:null,
      iv=model&&model.initial_sz?Number(model.initial_sz[best]):null
    out.textContent=`#${s.site_id??best} · (${fmt(s.pos?.[0],3)}, ${fmt(s.pos?.[1],3)}) µm`+
      (dv!==null?` · δ ${fmt(dv,3)}`:'')+(ov?` · Ω ${fmt(ov,3)}`:'')+(iv!==null?` · ⟨σᶻ⟩ ${fmt(iv,3)}`:'')}
  c.onmouseup=()=>{if(!geoDragging)return;geoDragging=false;c.style.cursor='crosshair';setTimeout(()=>{geoMoved=false},0)}
  c.onmouseleave=()=>{const out=$('.geo-readout');if(out)out.textContent=''}
}
function paintModel(model,running){
  const box=$('.model-formula');if(!box)return
  currentModel=model||null
  const atomSel=$('.atom-quantity'),pairSel=$('.pair-quantity')
  for(const sel of [atomSel,pairSel])if(sel)sel.disabled=!model
  const initialOpt=atomSel?.querySelector('option[value="initial"]')
  if(initialOpt)initialOpt.disabled=!model?.initial_sz
  if(atomMode==='initial'&&!model?.initial_sz)atomMode='delta'
  if(!model){
    box.innerHTML=running?'<code>模型载入中…</code>':'<code>本次结果未携带模型数据</code>'
    paintGeometry(geoRun);return
  }
  if(atomSel)atomSel.value=atomMode
  if(pairSel)pairSel.value=pairMode
  const J=model.pairs.map(p=>p[2]),V=model.pairs.map(p=>p[3])
  const jm=Math.max(...J.map(Math.abs),1e-12),vm=Math.max(...V.map(Math.abs),0)
  const sum=J.reduce((s,x)=>s+Math.abs(x),0)
  const om=model.omega||[],omMax=Math.max(...om,0)
  const chans=[]
  if(vm>0)chans.push('Σ V_{ij} nᵢnⱼ')
  if(jm>0)chans.push('Σ J_{ij} σ⁺ᵢσ⁻ⱼ + h.c.')
  if(omMax>0)chans.push('Σ (Ωᵢ/2) σˣᵢ')
  box.innerHTML=`<code>H = Σ δᵢnᵢ${chans.length?' + '+chans.join(' + '):''}</code>`+
    `<span>δ_max ${fmt(Math.max(...model.delta.map(Math.abs)),3)} · J_max ${fmt(jm,3)} · Σ|J| ${fmt(sum,4)} MHz`+
    `${vm>0?` · V_max ${fmt(vm,3)}`:''}${omMax>0?` · Ω_max ${fmt(omMax,3)}`:''}</span>`
  paintGeometry(geoRun)
}
function paintInput(run){
  geoRun=run
  paintGeometry(run)
  const atomSel=$('.atom-quantity'),pairSel=$('.pair-quantity')
  if(atomSel)atomSel.onchange=()=>{atomMode=atomSel.value;paintGeometry(geoRun)}
  if(pairSel)pairSel.onchange=()=>{pairMode=pairSel.value;paintGeometry(geoRun)}
  const wave=$('.drive-mini');if(wave)drawLines(wave,(run.drives_preview||[]).slice(0,3).map((d,i)=>({x:d.x,y:d.values,label:d.label,color:palette[i]})),{compact:true,xLabel:'ns'})
}
function saveInput(){const panel=$('.input-panel');if(panel&&renderedId)inputState.set(renderedId,{scroll:panel.scrollTop,groups:new Map([...panel.querySelectorAll('[data-group]')].map(d=>[d.dataset.group,d.open]))})}
function restoreInput(){const panel=$('.input-panel'),old=inputState.get(selected);if(!panel||!old)return;panel.scrollTop=old.scroll;panel.querySelectorAll('[data-group]').forEach(d=>{if(old.groups.has(d.dataset.group))d.open=old.groups.get(d.dataset.group)})}
function updateNavigation(list){
  $('#run-count').textContent=records.length
  const i=list.findIndex(r=>r.id===selected);$('#position').textContent=`${Math.max(0,i+1)} / ${list.length}`;$('#prev').disabled=i<=0;$('#next').disabled=i<0||i>=list.length-1
  $('#history').innerHTML=list.map(r=>`<button data-id="${escape(r.id)}" class="${r.id===selected?'active':''}"><span class="dot ${r.status}"></span>${escape(runTitle(r))}<small>${escape(r.createdAt)} · ${escape(statusNames[r.status]||r.status)}</small></button>`).join('')
  $('#history').querySelectorAll('[data-id]').forEach(b=>b.onclick=()=>choose(b.dataset.id))
}
function choose(id){saveInput();dispose();selected=id;signature='';render()}
function move(delta){const list=filtered(),i=list.findIndex(r=>r.id===selected);if(list.length)choose(list[Math.max(0,Math.min(list.length-1,i+delta))].id)}
async function detail(record){
  const key=record.id+'/'+record.result_version
  if(results.has(key))return results.get(key)
  if(!inflight.has(key))inflight.set(key,(async()=>{const response=await fetch(`/api/runs/${encodeURIComponent(record.id)}/result`);if(!response.ok)throw Error('结果数据读取失败：HTTP '+response.status);const result=await response.json();results.set(key,result);return result})().finally(()=>inflight.delete(key)))
  return inflight.get(key)
}
// 模型本体按需取（不轮询）：卡片一出现就能看耦合/δ，不必等结果算完。
// 缓存键带 input_version：卡片刚 begin 时 build 还没跑完，那时取不到模型；
// 等 update_run 把版本号 +1 后要能重新取，不能让早先的 null 永久粘住。
async function loadModel(record){
  const key=record.id+'/'+record.input_version
  if(models.has(key)){paintModel(models.get(key),record.status==='running');return}
  try{
    const response=await fetch(`/api/runs/${encodeURIComponent(record.id)}/model`,{cache:'no-store'})
    if(!response.ok){
      // 只在 build 已经落地（版本 > 0）时才把"没有模型"当成结论
      if(record.input_version>0)models.set(key,null)
      if(selected===record.id&&record.status==='running')setTimeout(()=>{
        const fresh=records.find(r=>r.id===record.id)
        if(fresh&&selected===fresh.id)loadModel(fresh)
      },2000)
      return
    }
    const payload=await response.json()
    models.set(key,payload.model||null)
    if(selected===record.id)paintModel(payload.model||null,record.status==='running')
  }catch{/* 旧卡/无模型：保持占位，不编造 */}
}
function pending(record){return `<div class="pending"><strong>${record.status==='running'?'计算中':escape(statusNames[record.status]||record.status)}</strong><p class="stage-text">${escape(record.error||record.stage||'正在组装输入')}</p>${record.status==='running'?`<div class="progress-track"><i style="width:${Math.max(0,Math.min(100,record.progress||0))}%"></i></div><p class="progress-text">${fmt(record.progress||0,1)}% · ${escape(record.elapsed||'—')}</p>`:''}</div>`}
async function render(){
  if(page==='config')return
  const list=filtered();if(!list.some(r=>r.id===selected)){dispose();selected=list[0]?.id||null;signature=''}updateNavigation(list)
  const record=list.find(r=>r.id===selected)
  if(!record){dispose();signature='';$('#card-host').innerHTML=`<div class="empty"><b>${connected?'等待 AI 发起运行':'等待连接 EL'}</b><p>${connected?'每次运行都会在这里生成卡片。':'请从 EL 的 AI 模式打开本页。服务不可用时不会展示示例计算结果。'}</p></div>`;return}
  const key=`${record.id}|${record.status}|${record.result_version}|${record.input_version}`
  if(signature===key){const progress=$('.progress-track i');if(progress)progress.style.width=Math.max(0,Math.min(100,record.progress||0))+'%';if($('.progress-text'))$('.progress-text').textContent=fmt(record.progress||0,1)+'% · '+(record.elapsed||'—');if($('.stage-text')&&record.status==='running')$('.stage-text').textContent=record.stage;return}
  saveInput();dispose();signature=key;renderedId=record.id;const token=++renderToken
  currentModel=null;atomMode='delta';pairMode='J';geoZoom=1;geoPanX=0;geoPanY=0
  $('#card-host').innerHTML=`<article class="run-card"><header class="card-head"><span class="id">${escape(record.id)}</span><span>${escape(record.createdAt)}</span><span class="reason">${escape(record.reason||bodyName(record.run))}</span><span class="status ${escape(record.status)}">${escape(statusNames[record.status]||record.status)} · ${escape(record.elapsed||'—')}</span></header><div class="card-content"><aside class="input-panel">${inputMarkup(record.run,record)}</aside><section class="result-panel"><div class="result-header"><span class="section-label">输出 / ENGINE RESULT</span><span class="source">${escape(record.run.method||'')} · 真实运行</span></div>${record.error?`<div class="error">${escape(record.error)}</div>`:''}<div class="result-mount">${pending(record)}</div></section></div></article>`
  restoreInput();requestAnimationFrame(()=>{paintInput(record.run);if(!record.has_visualization)paintModel(null,record.status==='running')});loadModel(record)
  if(record.has_visualization){
    const mount=$('.result-mount');mount.innerHTML='<div class="pending"><strong>载入结果</strong><p>正在读取该次运行的观测数据</p></div>'
    try{const result=await detail(record);if(token!==renderToken||selected!==record.id||page!=='runs')return
      if(!result.visualization){paintModel(null,record.status==='running');mount.innerHTML='<div class="pending"><strong>没有可用的观测数据</strong><p>'+escape(record.projection_error||'计算没有返回可视化结果')+'</p></div>';return}
      $('.source').textContent=(result.visualization.backend||record.run.method||'')+' · 真实运行'
      paintModel(result.visualization.model,false)
      widget=new ResultView(mount,result.visualization,{...record.result,monitor_id:record.id})
      if(saved.has(record.id)){Object.assign(widget.state,saved.get(record.id));widget.controls();widget.paint()}
    }catch(error){if(token===renderToken)mount.innerHTML=`<div class="error">${escape(error.message)}。重新选择此卡片可重试。</div>`}
  }else if(record.projection_error){$('.result-mount').innerHTML=`<div class="error">结果已计算，但可视化映射失败：${escape(record.projection_error)}</div>`}
}
async function poll(){
  try{const response=await fetch('/api/runs',{cache:'no-store',signal:AbortSignal.timeout(8000)});if(!response.ok)throw Error('HTTP '+response.status);const payload=await response.json();if(!payload.ok||!Array.isArray(payload.runs))throw Error('无效的监控数据')
    const wasLatest=!selected||selected===records[0]?.id,newest=payload.runs[0]?.id
    records=payload.runs;connected=true;setConnection('● 运行服务已连接','online');$('#sync').textContent='同步于 '+new Date().toLocaleTimeString('zh-CN')
    if(wasLatest&&newest&&newest!==selected){saveInput();dispose();selected=newest;signature=''}
    const keys=new Set(records.map(r=>r.id+'/'+r.result_version));for(const key of results.keys())if(!keys.has(key))results.delete(key)
    const live=new Set(records.map(r=>r.id));for(const key of models.keys())if(!live.has(key))models.delete(key)
    render()
  }catch(error){setConnection('● 连接中断 · 自动重试','offline');$('#sync').textContent=records.length?'连接中断，保留最近一次真实结果':'等待 EL 运行服务';if(!records.length){connected=false;render()}}
  finally{setTimeout(poll,1200)}
}
$('#search').oninput=()=>{signature='';render()}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));signature='';render()})
$('#prev').onclick=()=>move(-1);$('#next').onclick=()=>move(1)
document.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea')||page!=='runs')return;if(['ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();move(e.key==='ArrowDown'?1:-1)}})
$('#stage').addEventListener('wheel',e=>{if(page!=='runs'||e.target.closest('.input-panel,input,select,.basis-picker,.technical'))return;if(Math.abs(e.deltaY)<28)return;e.preventDefault();if(performance.now()-lastWheel<500)return;lastWheel=performance.now();move(e.deltaY>0?1:-1)},{passive:false})
document.querySelectorAll('[data-page]').forEach(b=>b.onclick=async()=>{page=b.dataset.page;document.querySelectorAll('[data-page]').forEach(x=>x.classList.toggle('active',x===b));saveInput();dispose();signature='';++renderToken;if(page==='runs')return render();$('#card-host').innerHTML='<div class="config-page"><div class="section-label">当前配置 / 只读</div><pre>载入中…</pre></div>';try{const response=await fetch('/api/config');const payload=await response.json();if(page==='config')$('.config-page pre').textContent=JSON.stringify(payload.config,null,2)}catch{if(page==='config')$('.config-page pre').textContent='无法连接配置服务'}})
window.addEventListener('resize',()=>{const r=records.find(r=>r.id===selected);if(r)paintInput(r.run)})

window.addEventListener('mouseup',()=>{if(!geoDragging)return;geoDragging=false;const c=document.querySelector('.input-geometry');if(c)c.style.cursor='crosshair';setTimeout(()=>{geoMoved=false},0)})
render();poll()
