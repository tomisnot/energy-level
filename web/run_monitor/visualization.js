import {advancePlayback,PLAYBACK_SECONDS,finite,fmt,escape,extent,nearestFrame,valuesOf,decodeBasis,projection,bestPlane,canvasContext,color,palette,drawLines} from './plot.js'

const axisUnit={time:'ns',beta:'1/MHz',energy:'MHz',tau:''}
// 轴单位优先用**数据声明**（dataset.axis_unit），常量表只作兜底
const unitOf=ds=>ds?.axis_unit??axisUnit[ds?.axis]??''
const spatial=new Set(['sites','links','loops','spins','partition'])
const option=(id,label,selected)=>`<option value="${escape(id)}"${selected===id?' selected':''}>${escape(label)}</option>`

export class ResultView {
  constructor(root,data,summary={}) {
    this.root=root;this.data=data;this.datasets=data.datasets||[];this.summary=summary
    this.state={mode:this.datasets.some(d=>spatial.has(d.kind))?'array':this.datasets.some(d=>d.kind==='basis')?'basis':'curves',
      selected:0,edge:0,basis:0,plane:bestPlane(data.sites||[]),component:'real',fraction:1,
      zoom:1,panX:0,panY:0,
      playing:false,loop:true,threshold:0,reference:'all',
      // 色标默认按**数据实际范围**（0.48–0.52 也能看出结构）；需要时切回语义范围（占据 0–1、自旋 ±1）。
      autoScale:true}
    this.cache=new Map();this.closed=false;this.last=0;this.points=[]
    this.build()
    if(this.current()?.axis==='time' && this.state.mode==='array') {this.state.fraction=0;this.state.playing=true}
    this.resize=new ResizeObserver(()=>this.paint());this.resize.observe(this.root)
    this.animate=this.animate.bind(this);this.raf=requestAnimationFrame(this.animate)
  }
  destroy(){this.closed=true;cancelAnimationFrame(this.raf);this.resize?.disconnect();window.removeEventListener('mousemove',this._dragMove);window.removeEventListener('mouseup',this._dragEnd)}
  candidates(){
    return this.datasets.filter(d=>this.state.mode==='array'?spatial.has(d.kind):this.state.mode==='basis'?d.kind==='basis':this.state.mode==='spectrum'?d.kind==='spectrum':this.state.mode==='histogram'?d.kind==='histogram':['curve','sites','links','loops'].includes(d.kind)&&d.axis)
  }
  current(){const list=this.candidates();return list.find(d=>d.id===this.state.dataset)||list[0]}
  values(ds){const key=ds.id+'|'+(ds.complex?this.state.component:'real');if(!this.cache.has(key))this.cache.set(key,valuesOf(ds,ds.complex?this.state.component:'real'));return this.cache.get(key)}
  frame(ds){return ds?.axis?nearestFrame(ds.x,this.state.fraction):0}
  row(ds){if(!ds)return [];const v=this.values(ds);return ds.axis?v[this.frame(ds)]:v}
  limits(ds){
    if((ds.complex&&this.state.component==='phase')||ds.id==='observables.phase')return [-Math.PI,Math.PI]
    const semantic=['pops_atom','n_atom','occupations'].includes(ds.id)?[0,1]
      :['observables.sx','observables.sy','observables.sz'].includes(ds.id)?[-1,1]:null
    return this.scaleFor(this.values(ds),semantic)
  }
  /** 色标范围：自动=数据实际范围（负值对称）；否则按语义范围（占据 0–1 / 自旋 ±1）。 */
  scaleFor(values,semantic=null){
    if(!this.state.autoScale&&semantic)return semantic
    const e=extent(values)
    const key=`scale|${this.state.autoScale?'auto':'fixed'}|${semantic?semantic.join(','):'-'}|${e[0]}|${e[1]}`
    if(this.cache.has(key))return this.cache.get(key)
    let lo=Number.isFinite(e[0])?e[0]:0,hi=Number.isFinite(e[1])?e[1]:1
    if(!(hi>lo)){const c=(hi+lo)/2||0,m=Math.abs(c)*0.01||0.5;lo=c-m;hi=c+m}
    const pad=(hi-lo)*0.06
    lo-=pad;hi+=pad
    if(lo<0){const m=Math.max(-lo,hi);lo=-m;hi=m}
    const out=[lo,hi]
    this.cache.set(key,out)
    return out
  }
  build(){
    const modes=[['array','原子阵列'],['curves','曲线'],['basis','量子构型'],['spectrum','能谱'],['histogram','统计']].filter(([mode])=>{
      return this.datasets.some(d=>mode==='array'?spatial.has(d.kind):mode==='basis'?d.kind==='basis':mode==='spectrum'?d.kind==='spectrum':mode==='histogram'?d.kind==='histogram':['curve','sites','links','loops'].includes(d.kind)&&d.axis)
    })
    if(!modes.length){
      this.root.innerHTML=`<div class="metrics">${(this.data.scalars||[]).map(s=>`<span class="metric">${escape(s.label)}<b>${fmt(s.value,5)}</b><small>${s.unit==='1'?'':escape(s.unit)}</small></span>`).join('')}</div><div class="pending"><strong>没有可绘制的观测数组</strong><p>此结果仅有标量或原生态。完整张量保留在运行存档。</p>${this.data.artifact_available?`<a href="/api/runs/${encodeURIComponent(this.summary.monitor_id)}/artifact">完整结果 ↓</a>`:''}</div><details class="technical"><summary>计算诊断与数据说明</summary><pre>${escape(JSON.stringify({notes:this.data.notes,state:this.data.state,diagnostics:this.data.diagnostics},null,2))}</pre></details>`
      return
    }
    if(!modes.some(([id])=>id===this.state.mode))this.state.mode=modes[0][0]
    this.root.innerHTML=`<div class="metrics">${(this.data.scalars||[]).slice(0,8).map(s=>`<span class="metric">${escape(s.label)}<b>${fmt(s.value,5)}</b><small>${s.unit==='1'?'':escape(s.unit)}</small></span>`).join('')}</div><div class="view-tabs">${modes.map(([id,label])=>`<button data-mode="${id}" class="${id===this.state.mode?'active':''}">${label}</button>`).join('')}<button class="export" title="下载当前运行的绘图数据">绘图数据 ↓</button>${this.data.artifact_available?`<a class="artifact-link" href="/api/runs/${encodeURIComponent(this.summary.monitor_id)}/artifact">完整结果 ↓</a>`:""}</div><div class="visual-controls"></div><div class="plot-stage"><canvas class="main-canvas" role="img" aria-label="运行结果可视化，点击原子查看数值和曲线"></canvas><div class="plot-label"></div><div class="legend"></div><div class="node-readout"></div></div><div class="basis-picker hidden"></div><div class="playback"><button class="play" aria-label="播放或暂停动画">▶</button><input class="scrub" type="range" min="0" max="1000" value="1000" aria-label="选择实际采样时刻"><span class="coordinate"></span><span class="duration">8 秒 / 轮</span><label><input class="loop" type="checkbox" checked>循环</label></div><div class="trace-box"><canvas class="trace-canvas" role="img" aria-label="选中原子或观测量的真实数值曲线"></canvas><span class="trace-title"></span><span class="trace-legend"></span></div><div class="result-note"></div><details class="technical"><summary>计算诊断与数据说明</summary><pre></pre></details>`
    this.canvas=this.root.querySelector('.main-canvas');this.trace=this.root.querySelector('.trace-canvas')
    this.canvas.title='滚轮缩放 · 双击复位'
    const wheelZoom=e=>{
      if(!['array','basis'].includes(this.state.mode))return
      e.preventDefault();e.stopImmediatePropagation()
      const r=this.canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top
      const old=this.state.zoom||1,z2=Math.max(.35,Math.min(8,old*Math.exp(-Number(e.deltaY||0)*.0015)))
      const cx=r.width/2,cy=Math.max(35,r.height-88)/2+34
      this.state.panX=(mx-cx)-(mx-cx-(this.state.panX||0))*(z2/old)
      this.state.panY=(my-cy)-(my-cy-(this.state.panY||0))*(z2/old)
      this.state.zoom=z2;this.paint()
    }
    this.canvas.onwheel=wheelZoom
    // 图例/标签覆盖在 canvas 上：整个 .plot-stage 都优先缩放，不能被卡片翻页截走。
    const stage=this.root.querySelector('.plot-stage');if(stage)stage.onwheel=wheelZoom
    this.canvas.ondblclick=()=>{if(!['array','basis'].includes(this.state.mode))return;this.state.zoom=1;this.state.panX=0;this.state.panY=0;this.paint()}
    this.dragging=false;this.dragMoved=false;this.dragStart=null;this.suppressClick=false
    this.canvas.onmousedown=e=>{if(!['array','basis'].includes(this.state.mode)||e.button!==0)return
      e.preventDefault();this.dragging=true;this.dragMoved=false;this.dragStart=[e.clientX,e.clientY,this.state.panX||0,this.state.panY||0];this.canvas.style.cursor='grabbing'}
    this._dragMove=e=>{if(!this.dragging)return;const dx=e.clientX-this.dragStart[0],dy=e.clientY-this.dragStart[1]
      if(Math.abs(dx)+Math.abs(dy)>2)this.dragMoved=true
      this.state.panX=this.dragStart[2]+dx;this.state.panY=this.dragStart[3]+dy;this.paint()}
    this._dragEnd=()=>{if(!this.dragging)return;this.dragging=false;this.canvas.style.cursor='';this.suppressClick=this.dragMoved
      setTimeout(()=>{this.dragMoved=false;this.suppressClick=false},0)}
    window.addEventListener('mousemove',this._dragMove);window.addEventListener('mouseup',this._dragEnd)
    this.root.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{this.state.mode=b.dataset.mode;this.state.dataset=null;this.state.fraction=1;this.state.playing=false;this.state.zoom=1;this.state.panX=0;this.state.panY=0;this.controls();this.paint()})
    this.root.querySelector('.play').onclick=()=>{if(!this.state.playing&&this.state.fraction>=1)this.state.fraction=0;this.state.playing=!this.state.playing;this.last=0;this.paint()}
    this.root.querySelector('.scrub').oninput=e=>{this.state.fraction=Number(e.target.value)/1000;this.state.playing=false;this.paint()}
    this.root.querySelector('.loop').onchange=e=>this.state.loop=e.target.checked
    this.root.querySelector('.export').onclick=()=>{const blob=new Blob([JSON.stringify({summary:this.summary,visualization:this.data},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`EL-${this.summary.monitor_id||'result'}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
    this.canvas.onclick=e=>{if(this.suppressClick){this.suppressClick=false;return}if(!['array','basis'].includes(this.state.mode))return;const r=this.canvas.getBoundingClientRect();const x=e.clientX-r.left,y=e.clientY-r.top;let best=-1,dist=Infinity;this.points.forEach((p,i)=>{if(!p)return;const d=Math.hypot(p[0]-x,p[1]-y);if(d<dist){best=i;dist=d}});if(best>=0&&dist<22){this.state.selected=best;this.state.edgePicked=false;this.paint();return}
      let chosen=null,edgeDistance=Infinity
      for(const edge of this.visibleEdges||[]){const [a,b]=edge.points,dx=b[0]-a[0],dy=b[1]-a[1],u=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy||1))),d=Math.hypot(x-a[0]-u*dx,y-a[1]-u*dy);if(d<edgeDistance){edgeDistance=d;chosen=edge}}
      if(chosen&&edgeDistance<8){this.state.edge=chosen.index;this.state.edgePicked=true;this.paint()}}
    const detail={approximation:this.data.approximation,notes:this.data.notes,state:this.data.state,diagnostics:this.data.diagnostics,artifact_id:this.data.artifact_id}
    this.root.querySelector('.technical pre').textContent=JSON.stringify(detail,null,2)
    this.controls();this.paint()
  }
  controls(){
    const ds=this.current();if(!ds)return
    this.state.dataset=ds.id
    this.root.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===this.state.mode))
    const isSpatial=['array','basis'].includes(this.state.mode)&&this.data.sites?.length>0
    this.root.querySelector('.visual-controls').innerHTML=`<select class="quantity" aria-label="选择观测量">${this.candidates().map(d=>option(d.id,d.label,ds.id)).join('')}</select>${ds.complex?`<select class="component" aria-label="复值显示方式">${[['real','实部'],['imaginary','虚部'],['magnitude','幅值'],['phase','相位']].map(([v,t])=>option(v,t,this.state.component)).join('')}</select>`:''}${isSpatial?`<select class="plane" aria-label="几何投影">${['xy','xz','yz'].map(p=>option(p,p.toUpperCase()+' 投影',this.state.plane)).join('')}</select><select class="colorscale" aria-label="色标范围">${option('auto','色标：自动范围',this.state.autoScale?'auto':'fixed')}${option('fixed','色标：语义范围（占0–1/自旋±1）',this.state.autoScale?'auto':'fixed')}</select>`:''}${ds.kind==='links'?`<select class="reference" aria-label="关联参考原子">${option('all','全部连线',this.state.reference)}${option('selected','选中原子的关联',this.state.reference)}</select><label>隐藏弱关联<input class="threshold" type="range" min="0" max="90" value="${this.state.threshold}" aria-label="关联显示阈值"></label>`:''}`
    this.root.querySelector('.quantity').onchange=e=>{this.state.dataset=e.target.value;this.state.basis=0;this.state.edgePicked=false;this.state.playing=false;this.state.fraction=1;this.controls();this.paint()}
    const bind=(cls,key,convert=v=>v)=>{const element=this.root.querySelector('.'+cls);if(element)element.onchange=e=>{this.state[key]=convert(e.target.value);this.cache.clear();this.paint()}}
    bind('component','component');bind('plane','plane');bind('reference','reference');bind('threshold','threshold',Number)
    bind('colorscale','autoScale',v=>v==='auto')
    const dynamic=ds.axis&&ds.x?.length>1
    this.root.querySelector('.playback').classList.toggle('hidden',!dynamic)
    if(!dynamic)this.state.playing=false
    this.root.querySelector('.duration').textContent=PLAYBACK_SECONDS+' 秒 / 轮'
    this.root.querySelector('.trace-box').classList.toggle('hidden',!isSpatial||(!ds.axis&&!this.datasets.some(d=>d.kind==='curve'&&d.axis==='time')))
    const picker=this.root.querySelector('.basis-picker');picker.classList.toggle('hidden',ds.kind!=='basis')
    if(ds.kind==='basis'){
      const vals=this.row(ds)||[]
      picker.innerHTML=(ds.codes||[]).map((code,i)=>`<button data-basis="${i}" class="${i===this.state.basis?'active':''}">${this.data.sites?.length>0&&this.data.sites.length<=12?'|'+decodeBasis(code,this.data.sites.length).map(b=>b?'r':'g').join('')+'⟩':escape(ds.labels?.[i]||'构型 '+(i+1))}<small>${fmt(vals[i],5)}${ds.phases?' · φ '+fmt(ds.phases[i],2):''}</small></button>`).join('')
      picker.querySelectorAll('[data-basis]').forEach(b=>b.onclick=()=>{this.state.basis=Number(b.dataset.basis);this.paint()})
    }
  }
  animate(now){
    if(this.closed)return
    if(this.state.playing&&document.visibilityState==='visible'){
      if(this.last)Object.assign(this.state,advancePlayback(this.state.fraction,now-this.last,this.state.loop))
      this.paint()
    }
    this.last=now;this.raf=requestAnimationFrame(this.animate)
  }
  paint(){
    const ds=this.current();if(!ds||!this.canvas)return
    if(this.state.mode==='array'||(this.state.mode==='basis'&&this.data.sites?.length))this.drawArray(ds)
    else this.drawChart(ds)
    this.drawTrace(ds)
    const f=this.frame(ds),x=ds.x?.[f]
    this.root.querySelector('.play').textContent=this.state.playing?'Ⅱ':'▶'
    this.root.querySelector('.scrub').value=String(Math.round(this.state.fraction*1000))
    this.root.querySelector('.coordinate').textContent=ds.axis?`${ds.axis==='time'?'t':ds.axis==='beta'?'β':ds.axis==='energy'?'E':'τ'} = ${fmt(x,4)} ${unitOf(ds)}`:''
    this.root.querySelector('.result-note').textContent=[ds.sampling,ds.scope,this.data.approximation?`近似：${this.data.approximation}`:'',this.data.notes?.[0]].filter(Boolean).join(' · ')
    if(ds.kind==='basis'){
      const vals=this.row(ds)||[]
      this.root.querySelectorAll('[data-basis]').forEach(b=>{const i=Number(b.dataset.basis);b.classList.toggle('active',i===this.state.basis);b.querySelector('small').textContent=fmt(vals[i],5)+(ds.phases?' · φ '+fmt(ds.phases[i],2):'')})
    }
  }
  drawArray(ds){
    const box=canvasContext(this.canvas);if(!box)return
    const {ctx,w,h}=box,sites=this.data.sites||[],n=sites.length
    if(!n||sites.some(s=>!s.pos)){
      this.points=[];ctx.fillStyle='#64748b';ctx.textAlign='center';ctx.fillText('没有返回原子坐标；下方显示真实曲线',w/2,h/2);return
    }
    const vh=Math.max(35,h-88),z=this.state.zoom||1,px=this.state.panX||0,py=this.state.panY||0,cx=w/2,cy=vh/2+34
    const p=projection(sites,this.state.plane,w,vh,1)
    p.positions=p.positions.map(point=>point?[(point[0]-cx)*z+cx+px,(point[1]+34-cy)*z+cy+py]:null)
    p.scale*=z
    this.points=p.positions
    let vals=this.row(ds)||[],occupancy=null,vectors=null
    if(ds.kind==='spins'){vectors=vals;vals=vectors.map(v=>v?.[2])}
    if(ds.kind==='partition')vals=sites.map((_,i)=>ds.part_a.includes(i)?0:1)
    if(ds.kind==='basis'){try{vals=decodeBasis(ds.codes[this.state.basis],n)}catch{vals=[]}}
    const background=this.datasets.find(d=>['pops_atom','n_atom','occupations'].includes(d.id))
    if(['links','loops'].includes(ds.kind)&&background){const all=this.values(background);let frame=all.length-1;if(ds.axis&&background.axis===ds.axis){const target=ds.x[this.frame(ds)];frame=background.x.reduce((best,v,i)=>Math.abs(v-target)<Math.abs(background.x[best]-target)?i:best,0)}occupancy=background.axis?all[frame]:all}
    const [lo,hi]=['basis','partition'].includes(ds.kind)?[0,1]:ds.kind==='spins'?[-1,1]:this.limits(ds)
    const phase=(ds.complex&&this.state.component==='phase')||ds.id==='observables.phase',signed=lo<0&&!phase
    // A calibrated scale bar avoids suggesting independent x/y stretching.
    const span=Math.max(p.bounds[1]-p.bounds[0],p.bounds[3]-p.bounds[2])
    if(span>0){
      const raw=Math.min(span/3,95/p.scale),power=10**Math.floor(Math.log10(raw)),length=[1,2,5,10].map(v=>v*power).filter(v=>v<=raw).at(-1)||raw
      ctx.strokeStyle='#64748b';ctx.lineWidth=1.3;ctx.beginPath();ctx.moveTo(w-20-length*p.scale,23);ctx.lineTo(w-20,23);ctx.moveTo(w-20-length*p.scale,20);ctx.lineTo(w-20-length*p.scale,26);ctx.moveTo(w-20,20);ctx.lineTo(w-20,26);ctx.stroke()
      ctx.fillStyle='#475569';ctx.font='11px "Segoe UI", sans-serif';ctx.textAlign='right';ctx.fillText(`${fmt(length,3)} μm`,w-20,15)
    }
    const max=Math.max(Math.abs(lo),Math.abs(hi),1e-12)
    let cut=max*this.state.threshold/100
    const magKey=ds.id+'|magnitude-max'
    if(phase&&ds.complex&&!this.cache.has(magKey))this.cache.set(magKey,Math.max(1e-12,extent(valuesOf(ds,'magnitude'))[1]))
    if(phase&&ds.complex)cut=this.cache.get(magKey)*this.state.threshold/100
    this.visibleEdges=[]
    if(ds.kind==='links'){
      (ds.edges||[]).forEach(([a,b],i)=>{if(this.state.reference==='selected'&&a!==this.state.selected&&b!==this.state.selected)return
        const v=vals[i],aa=p.positions[a],bb=p.positions[b];if(!aa||!bb||!finite(v)||(!phase&&Math.abs(v)<cut))return
        let strength=Math.abs(v)/max
        if(ds.complex&&this.state.component==='phase') {const re=ds.axis?ds.values[this.frame(ds)][i]:ds.values[i],im=ds.axis?ds.imaginary[this.frame(ds)][i]:ds.imaginary[i];const magnitude=Math.hypot(re||0,im||0);if(magnitude<cut)return;strength=magnitude/this.cache.get(magKey)}
        this.visibleEdges.push({index:i,points:[aa,bb]})
        if(this.state.edgePicked&&this.state.edge===i){ctx.strokeStyle='#0f766e';ctx.globalAlpha=.3;ctx.lineWidth=8;ctx.beginPath();ctx.moveTo(...aa);ctx.lineTo(...bb);ctx.stroke()}
        ctx.globalAlpha=.25+.65*Math.min(1,strength);ctx.lineWidth=.6+5*Math.min(1,strength);ctx.strokeStyle=color(v,lo,hi,phase,signed);ctx.beginPath();ctx.moveTo(...aa);ctx.lineTo(...bb);ctx.stroke();ctx.globalAlpha=1
      })
    }
    if(ds.kind==='loops'){
      (ds.loops||[]).forEach((loop,i)=>{const value=vals[i];if(!finite(value))return;ctx.strokeStyle=color(value,lo,hi,phase,signed);ctx.fillStyle=color(value,lo,hi,phase,signed);ctx.lineWidth=3.5;ctx.beginPath();let cx=0,cy=0;loop.forEach((site,j)=>{const point=p.positions[site];if(!point)return;j?ctx.lineTo(...point):ctx.moveTo(...point);cx+=point[0];cy+=point[1]});ctx.closePath();ctx.globalAlpha=.14;ctx.fill();ctx.globalAlpha=.9;ctx.stroke();ctx.globalAlpha=1;ctx.textAlign='center';ctx.fillText(fmt(value,3),cx/loop.length,cy/loop.length)})
    }
    let minDistance=Infinity
    for(let i=0;i<Math.min(n,180);i++)for(let j=i+1;j<Math.min(n,180);j++)if(p.positions[i]&&p.positions[j]){const d=Math.hypot(p.positions[i][0]-p.positions[j][0],p.positions[i][1]-p.positions[j][1]);if(d>0)minDistance=Math.min(minDistance,d)}
    const radius=Math.max(3,Math.min(17,finite(minDistance)?minDistance*.23:15))
    p.positions.forEach((point,i)=>{if(!point)return;const value=occupancy?occupancy[i]:['links','loops'].includes(ds.kind)?null:vals[i]
      const [nodeLo,nodeHi]=occupancy?this.scaleFor(occupancy,[0,1]):[lo,hi]
      const actual=finite(value)?value:null,selected=i===this.state.selected
      const r=radius*(ds.kind==='basis'?(actual===1?1:.58):.8)
      ctx.beginPath();ctx.arc(...point,r,0,Math.PI*2);ctx.fillStyle=ds.kind==='partition'?palette[actual]:actual===null?'#cbd5e1':color(actual,nodeLo,nodeHi,!occupancy&&phase,!occupancy&&signed);ctx.fill();ctx.lineWidth=selected?2.3:1;ctx.strokeStyle=selected?'#0f766e':'#94a3b8aa';ctx.stroke()
      if(n<=48&&radius>=7&&!vectors&&finite(actual)&&!['links','loops'].includes(ds.kind)){ctx.fillStyle=ds.kind==='partition'||phase||Math.abs(actual)>.65?'#ffffff':'#1f2937';ctx.font='10px sans-serif';ctx.textAlign='center';ctx.fillText(ds.kind==='partition'?(actual?'B':'A'):ds.kind==='basis'?(actual?'r':'g'):fmt(actual,2),point[0],point[1]+3);ctx.font='11px sans-serif'}
      if(selected){ctx.beginPath();ctx.arc(...point,r+5,0,Math.PI*2);ctx.strokeStyle='#0f766e88';ctx.lineWidth=1;ctx.stroke()}
      const error=ds.stderr?.[this.frame(ds)]?.[i]
      if(finite(error)&&error>0){ctx.beginPath();ctx.arc(...point,r+3+Math.min(10,error*25),0,Math.PI*2);ctx.strokeStyle='#b4530999';ctx.stroke()}
      const vector=vectors?.[i];if(vector&&finite(vector[0])&&finite(vector[1])&&Math.hypot(vector[0],vector[1])>1e-10){
        const dx=vector[0]*radius*1.6,dy=-vector[1]*radius*1.6,ex=point[0]+dx,ey=point[1]+dy,ang=Math.atan2(dy,dx),head=Math.min(5,Math.hypot(dx,dy)*.45);ctx.strokeStyle='#0f172a';ctx.lineWidth=1.8;ctx.beginPath();ctx.moveTo(...point);ctx.lineTo(ex,ey);ctx.lineTo(ex-head*Math.cos(ang-.5),ey-head*Math.sin(ang-.5));ctx.moveTo(ex,ey);ctx.lineTo(ex-head*Math.cos(ang+.5),ey-head*Math.sin(ang+.5));ctx.stroke()
      }
      if(n<=96||selected){ctx.fillStyle=selected?'#1f2937':'#475569';ctx.textAlign='center';ctx.fillText(String(sites[i].site_id??i),point[0],point[1]+r+15)}
    })
    const quantity=ds.kind==='partition'?`二分纠缠熵 S = ${fmt(this.row(ds),5)} nat`:ds.kind==='basis'?`构型概率 ${fmt((this.row(ds)||[])[this.state.basis],5)}`:ds.kind==='links'?`${ds.label}${ds.temporal?'':' · 仅末态'}`:ds.label
    const singleCoordinate=ds.axis&&ds.x?.length===1?` · ${ds.axis} = ${fmt(ds.x[0],4)} ${unitOf(ds)}`:''
    this.root.querySelector('.plot-label').innerHTML=`<b>${escape(quantity)}</b><br>${n} 个原子 · ${this.state.plane.toUpperCase()} 等比例投影${ds.kind==='links'?` · |值| ≥ ${fmt(cut,4)}`:''}${singleCoordinate}`
    const legend=this.root.querySelector('.legend');legend.innerHTML=`<span>${fmt(lo,3)}</span><span class="ramp ${phase?'phase':signed?'diverging':''}"></span><span>${fmt(hi,3)}</span><span>${phase?(ds.kind==='links'?'rad · 粗度＝关联幅值':'rad'):ds.kind==='links'?'连线：颜色＝值 · 粗度＝强度；节点＝占据':ds.kind==='spins'?'颜色 σᶻ · 箭头 (σˣ,σʸ)':(this.state.autoScale?'节点颜色＝值 · 色标自动':'节点颜色＝值')}</span>`
    if(ds.kind==='partition')legend.innerHTML=`<span style="color:${palette[0]}">● 子系统 A</span><span style="color:${palette[1]}">● 子系统 B</span><span>按求解器实际切分</span>`
    const selected=this.state.selected,position=sites[selected]?.pos
    let detail=`原子 ${escape(sites[selected]?.site_id??selected)} · (${position?.map(v=>fmt(v,2)).join(', ')}) μm`
    if(ds.kind==='links'){
      const index=this.state.edgePicked?this.state.edge:(ds.edges||[]).findIndex(e=>e.includes(selected));if(index>=0)detail+=`<br>C(${ds.edges[index].join(', ')}) = ${fmt(vals[index],5)}`
    }else if(ds.kind==='partition')detail+=`<br>属于子系统 ${vals[selected]?'B':'A'}`
    else if(!['loops'].includes(ds.kind))detail+=`<br>${ds.kind==='basis'?'r/g 占据':ds.kind==='spins'?'σᶻ':escape(ds.label)} = ${fmt(vals[selected],5)}${finite(ds.stderr?.[this.frame(ds)]?.[selected])?' ± '+fmt(ds.stderr[this.frame(ds)][selected],5)+'（标准误）':''}`
    this.root.querySelector('.node-readout').innerHTML=detail
  }
  curves(ds){
    const ys=this.values(ds),x=ds.x||[]
    if(['curve','partition'].includes(ds.kind))return [{x,y:Array.isArray(ys)?ys:[],label:ds.label,error:ds.stderr}]
    if(ds.kind==='spectrum'){
      if(ds.occupied)return [true,false].map(occupied=>({x:ys.map((_,i)=>i+1),y:ys.map((v,i)=>ds.occupied.includes(i)===occupied?v:null),label:occupied?'已占据轨道':'未占据轨道',color:occupied?palette[0]:palette[1]}))
      return [{x:ys.map((_,i)=>i+1),y:ys,label:ds.label}]
    }
    if(ds.kind==='histogram')return [{x:ds.x,y:ys,label:ds.label}]
    if(ds.kind==='basis'){
      if(!ds.axis)return [{x:ys.map((_,i)=>i+1),y:ys,label:'构型概率'}]
      const indices=this.data.sites?.length?[this.state.basis]:[...new Set([this.state.basis,...ys[0].map((_,i)=>i)])].slice(0,6)
      return indices.map((i,k)=>({x,y:ys.map(row=>row[i]),label:ds.labels?.[i]||'构型 '+(i+1),color:palette[k]}))
    }
    if(ds.axis&&Array.isArray(ys[0])){
      const width=ys[0].length,indices=ds.kind==='sites'?[Math.min(this.state.selected,width-1)]:ds.kind==='links'?[this.state.edgePicked?Math.min(this.state.edge,width-1):Math.max(0,(ds.edges||[]).findIndex(e=>e.includes(this.state.selected)))]:Array.from({length:Math.min(width,6)},(_,i)=>i)
      return indices.map((i,k)=>({x,y:ys.map(row=>row[i]),label:ds.kind==='sites'?`原子 ${this.data.sites?.[i]?.site_id??i}`:ds.kind==='links'?(ds.edges?.[i]||[]).join(' ↔ '):`环 ${i+1}`,error:ds.stderr?.map(row=>row[i]),color:palette[k]}))
    }
    return []
  }
  drawChart(ds){
    this.points=[]
    drawLines(this.canvas,this.curves(ds),{xLabel:ds.kind==='spectrum'?'能级序号':ds.kind==='histogram'?'跳跃次数':ds.kind==='basis'&&!ds.axis?'构型序号':unitOf(ds),spectrum:ds.kind==='spectrum',histogram:ds.kind==='histogram'||ds.kind==='basis'&&!ds.axis,cursor:ds.axis?ds.x[this.frame(ds)]:null})
    this.root.querySelector('.plot-label').innerHTML=`<b>${escape(ds.label)}</b> ${ds.unit==='1'?'':escape(ds.unit)}`
    this.root.querySelector('.legend').innerHTML=ds.kind==='spectrum'&&ds.occupied?`<span style="color:${palette[0]}">已占据轨道</span> · <span style="color:${palette[1]}">未占据轨道</span>`:ds.stderr?'阴影表示标准误差':'使用引擎返回的真实坐标与数值'
    if(ds.kind==='basis'&&ds.axis)this.root.querySelector('.legend').innerHTML=this.curves(ds).map((c,i)=>`<span style="color:${palette[i]}">${escape(c.label)}</span>`).join(' · ')
    this.root.querySelector('.node-readout').textContent=ds.kind==='curve'?`${ds.label} = ${fmt(this.row(ds),5)} ${ds.unit==='1'?'':ds.unit}`:''
  }
  drawTrace(ds){
    if(this.root.querySelector('.trace-box').classList.contains('hidden'))return
    let selected=ds,curves=this.curves(ds),title=ds.label
    if(!ds.axis){selected=this.datasets.find(d=>d.kind==='curve'&&d.axis==='time');curves=selected?this.curves(selected):[];title=selected?.label||'此观测量仅返回末态，没有含时序列'}
    if(ds.kind==='spins'&&ds.axis){const i=this.state.selected;curves=[0,1,2].map((k)=>({x:ds.x,y:ds.values.map(row=>row[i]?.[k]),label:['σˣ','σʸ','σᶻ'][k],color:palette[k]}))}
    drawLines(this.trace,curves,{compact:true,cursor:selected?.axis?selected.x[this.frame(selected)]:null,xLabel:unitOf(selected)})
    this.root.querySelector('.trace-title').textContent=title
    this.root.querySelector('.trace-legend').innerHTML=curves.map((c,i)=>`<span style="color:${palette[i%palette.length]}">${escape(c.label)}</span>`).join(' · ')
  }
}
