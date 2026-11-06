export const PLAYBACK_SECONDS=8
export function advancePlayback(fraction,elapsedMs,loop=true){
  const next=Math.max(0,fraction)+Math.max(0,elapsedMs)/(PLAYBACK_SECONDS*1000)
  return {fraction:next>=1?(loop?next%1:1):next,playing:next<1||loop}
}
export const finite = v => typeof v === 'number' && Number.isFinite(v)
export const fmt = (v, digits=3) => !finite(v) ? '—' : (v !== 0 && (Math.abs(v)<.001 || Math.abs(v)>=1e7)) ? v.toExponential(2) : Number(v.toFixed(digits)).toLocaleString('en-US')
export const escape = v => String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))
export function extent(values, fallback=[0,1]) {
  let lo=Infinity, hi=-Infinity
  const stack=[values]
  while(stack.length) { const a=stack.pop(); if(Array.isArray(a)) { for(const x of a) stack.push(x) } else if(finite(a)) {lo=Math.min(lo,a); hi=Math.max(hi,a)} }
  return lo===Infinity ? [...fallback] : [lo,hi]
}
export function nearestFrame(xs, fraction) {
  if(!xs?.length) return 0
  const target=xs[0]+Math.max(0,Math.min(1,fraction))*(xs.at(-1)-xs[0])
  let best=0
  for(let i=1;i<xs.length;i++) if(Math.abs(xs[i]-target)<Math.abs(xs[best]-target)) best=i
  return best
}
export function component(real, imag, mode='real') {
  if(!finite(real)) return null
  if(imag===undefined) return real
  if(!finite(imag)) return null
  if(mode==='imaginary') return imag
  if(mode==='magnitude') return Math.hypot(real,imag)
  if(mode==='phase') return Math.hypot(real,imag)<1e-14 ? null : Math.atan2(imag,real)
  return real
}
export function valuesOf(ds,mode='real') {
  const walk=(r,i)=>Array.isArray(r)?r.map((v,k)=>walk(v,i?.[k])):component(r,i,mode)
  return walk(ds.values,ds.imaginary)
}
export function decodeBasis(code,n) { const x=BigInt(code); return Array.from({length:n},(_,i)=>Number((x>>BigInt(n-1-i))&1n)) }
export function projection(sites,plane,w,h,zoom=1) {
  const axes={xy:[0,1],xz:[0,2],yz:[1,2]}[plane]||[0,1]
  const positions=sites.map(s=>s.pos && axes.every(i=>finite(s.pos[i])) ? axes.map(i=>s.pos[i]) : null)
  const valid=positions.filter(Boolean)
  const [xmin,xmax]=extent(valid.map(p=>p[0])), [ymin,ymax]=extent(valid.map(p=>p[1]))
  const dx=xmax-xmin,dy=ymax-ymin,pad=Math.min(58,Math.max(12,Math.min(w,h)*.13))
  const scale=Math.min(dx>0?(w-2*pad)/dx:Infinity,dy>0?(h-2*pad)/dy:Infinity)
  const factor=(Number.isFinite(scale)&&scale>0?scale:1)*zoom
  return {positions:positions.map(p=>p?[w/2+(p[0]-(xmin+xmax)/2)*factor,h/2-(p[1]-(ymin+ymax)/2)*factor]:null),bounds:[xmin,xmax,ymin,ymax],scale:factor,axes}
}
export function bestPlane(sites) {
  const spans=[0,1,2].map(i=>{const e=extent(sites.map(s=>s.pos?.[i]));return e[1]-e[0]})
  return ['xy','xz','yz'].sort((a,b)=>{const score=p=>{const [i,j]={xy:[0,1],xz:[0,2],yz:[1,2]}[p]; return spans[i]*spans[j]+.000001*(spans[i]+spans[j])};return score(b)-score(a)})[0]
}
export function canvasContext(canvas) {
  const {width:w,height:h}=canvas.getBoundingClientRect(); if(w<2||h<2)return null
  const dpr=Math.min(window.devicePixelRatio||1,2)
  const width=Math.round(w*dpr),height=Math.round(h*dpr)
  if(canvas.width!==width||canvas.height!==height) {canvas.width=width;canvas.height=height}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h)
  ctx.font='12px "Segoe UI", "Microsoft YaHei", sans-serif';ctx.lineCap='round';ctx.lineJoin='round'
  return {ctx,w,h}
}
export const palette=['#58d6c7','#86aaff','#efc577','#f291b7','#ba9cef','#7bcbf3']
export function color(value,lo,hi,phase=false,signed=false) {
  if(!finite(value))return '#536477'
  if(phase) return `hsl(${(value+Math.PI)/(2*Math.PI)*360} 67% 66%)`
  const u=Math.max(0,Math.min(1,(value-lo)/(hi-lo||1)))
  const mix=(a,b,t)=>`rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`
  if(signed) return u<.5?mix([255,137,128],[42,58,76],u*2):mix([42,58,76],[89,220,210],(u-.5)*2)
  return u<.65?mix([35,60,85],[65,191,183],u/.65):mix([65,191,183],[245,213,129],(u-.65)/.35)
}
export function drawLines(canvas, curves, options={}) {
  const box=canvasContext(canvas);if(!box)return
  const {ctx,w,h}=box, left=53,right=19,top=options.compact?25:43,bottom=options.compact?29:52
  if(!curves.some(c=>(c.y||[]).some(finite))){ctx.textAlign='center';ctx.fillStyle='#94a9bf';ctx.fillText('没有返回此观测量的含时序列',w/2,h/2);return}
  const all=curves.flatMap(c=>(c.y||[]).flatMap((y,i)=>finite(y)?[y-(c.error?.[i]||0),y+(c.error?.[i]||0)]:[]))
  let [lo,hi]=extent(all); if(options.yRange)[lo,hi]=options.yRange
  if(options.histogram)lo=0
  if(lo===hi){const d=Math.max(.05,Math.abs(lo)*.1);lo-=d;hi+=d}else{const d=(hi-lo)*.08;lo-=d;hi+=d}
  if(options.histogram)lo=0
  const [xlo,xhi]=extent(curves.flatMap(c=>c.x||[])), x=v=>xhi===xlo?(left+w-right)/2:left+(w-left-right)*(v-xlo)/(xhi-xlo), y=v=>h-bottom-(h-top-bottom)*(v-lo)/(hi-lo||1)
  ctx.textAlign='right';ctx.fillStyle='#93a9c0'
  for(let i=0;i<=3;i++){const v=lo+(hi-lo)*i/3,yy=y(v);ctx.strokeStyle='#2d405444';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(w-right,yy);ctx.stroke();ctx.fillText(fmt(v,3),left-8,yy+4)}
  ctx.textAlign='center'
  for(let i=0;i<=(xlo===xhi?0:4);i++){const v=xlo+(xhi-xlo)*i/4;ctx.fillText(fmt(v,2),x(v),h-bottom+18)}
  ctx.textAlign='right';ctx.font='10px sans-serif';ctx.fillText(options.xLabel||'',w-right,h-bottom+28);ctx.font='12px sans-serif'
  curves.forEach((curve,k)=>{const stroke=curve.color||palette[k%palette.length];ctx.strokeStyle=stroke;ctx.lineWidth=1.7
    if(curve.error){let segment=[];const fill=()=>{if(!segment.length)return;ctx.beginPath();segment.forEach((i,j)=>{const xx=x(curve.x[i]),yy=y(curve.y[i]+curve.error[i]);j?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)});[...segment].reverse().forEach(i=>ctx.lineTo(x(curve.x[i]),y(curve.y[i]-curve.error[i])));ctx.closePath();ctx.fillStyle=stroke+'25';ctx.fill();segment=[]};curve.y.forEach((v,i)=>{if(finite(v)&&finite(curve.error[i]))segment.push(i);else fill()});fill()}
    ctx.beginPath();let open=false
    curve.y.forEach((v,i)=>{if(!finite(v)||!finite(curve.x[i])){open=false;return}const xx=x(curve.x[i]),yy=y(v);if(options.spectrum){ctx.moveTo(xx-7,yy);ctx.lineTo(xx+7,yy)}else if(options.histogram){ctx.fillStyle=stroke;ctx.fillRect(xx-4,yy,8,Math.max(1,y(0)-yy))}else{open?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);open=true}});ctx.stroke()
    if(!options.spectrum&&!options.histogram&&curve.y.filter(finite).length===1){const i=curve.y.findIndex(finite);ctx.beginPath();ctx.arc(x(curve.x[i]),y(curve.y[i]),3,0,Math.PI*2);ctx.fillStyle=stroke;ctx.fill()}
  })
  if(finite(options.cursor)){ctx.setLineDash([3,4]);ctx.strokeStyle='#d5e9f0a0';ctx.beginPath();ctx.moveTo(x(options.cursor),top);ctx.lineTo(x(options.cursor),h-bottom);ctx.stroke();ctx.setLineDash([])}
  if(!curves.length){ctx.textAlign='center';ctx.fillStyle='#94a9bf';ctx.fillText('没有可用的曲线数据',w/2,h/2)}
}
