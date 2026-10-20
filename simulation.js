/* Inspect saved service scheduling, communication dependencies and resource waits. */
'use strict';
const serviceOwners=new Map();
const localServiceIndices=[];
Object.values(window.CASCADE_DATA.operators).forEach(op=>{
  op.service_indices.forEach(i=>serviceOwners.set(i,op.id));
  if(!op.messages.length)localServiceIndices.push(...op.service_indices);
});
const [chipColumn,startColumn,endColumn]=['chiplet_id','start_s','end_s'].map(key=>window.CASCADE_DATA.schedule.service_fields.indexOf(key));
let cachedSimulationView=null;
const serviceCache=new Map();
const commandRows=new Map((window.CASCADE_DATA.schedule.commands||[]).map(row=>[row[0],row]));
const commandCache=new Map(),tensorRecords=new Map();
Object.values(window.CASCADE_DATA.operators).forEach(op=>{
  op.instances.forEach(i=>[...i.inputs,...i.outputs].forEach(t=>tensorRecords.set(t.tensor_id,t)));
  op.messages.forEach(m=>[...m.input_tensors,...(m.output_tensor?[m.output_tensor]:[])].forEach(t=>tensorRecords.set(t.tensor_id,t)));
});
function serviceCommand(s){
  if(!commandCache.has(s.command_id))commandCache.set(s.command_id,Object.fromEntries(
    D.schedule.command_fields.map((key,i)=>[key,commandRows.get(s.command_id)[i]])));
  return commandCache.get(s.command_id);
}
function commandOperands(s){
  const command=serviceCommand(s),tensor=i=>tensorRecords.get(D.schedule.tensor_ids[i]);
  return {command,inputs:command.input_tensor_indices.map(tensor),
    output:command.output_tensor_index===null?null:tensor(command.output_tensor_index)};
}
function serviceExpert(s){
  const {command,inputs,output}=commandOperands(s);
  if(command.expert_id!==null)return command.expert_id;
  // DMA commands omit expert_id. Match their exact tensors to a saved owner;
  // operations that combine several experts must retain an unspecified expert.
  const ids=[...inputs,...(output?[output]:[])].map(t=>t.tensor_id);
  const owners=D.operators[s.owner].instances.filter(i=>i.chiplet_id===s.chiplet_id&&ids.length&&
    ids.every(id=>[...i.inputs,...i.outputs].some(t=>t.tensor_id===id)));
  const experts=new Set(owners.map(i=>i.expert_id));
  return experts.size===1?[...experts][0]:null;
}
const serviceColors={tensor:'#137464',simd:'#ac7d25',memory:'#477ab4',network:'#8254a4',host:'#795347'};
function scheduledService(index){
  if(!serviceCache.has(index)){
    const row=D.schedule.services[index];
    serviceCache.set(index,{...Object.fromEntries(D.schedule.service_fields.map((key,i)=>[key,row[i]])),
      index,id:row[0]+'.s'+row[1],owner:serviceOwners.get(index)});
  }
  return serviceCache.get(index);
}
function serviceResources(s){return s.resource_indices.map(i=>D.schedule.resources[i]);}
function serviceReservations(s){return s.reservation_resource_indices.map(i=>D.schedule.resources[i]);}
function serviceLocation(s){
  const resources=serviceResources(s),link=resources.find(r=>r[0]==='D2D');
  if(link)return `C${link[1]} → C${link[2]}`;
  const owner=resources.find(r=>r[0]==='HBM')||resources.find(r=>['TENSOR','SIMD','NOC'].includes(r[0]));
  return 'C'+(owner?owner[1]:s.chiplet_id);
}
function serviceLabel(s){
  if(s.name.startsWith('link_'))return s.name.replace(/^link_(\d+)_(\d+)$/,'D2D C$1 → C$2');
  return {tensor:'Tensor matrix multiply',simd:'SIMD vector / reduction',dma_load:'Local HBM read',
    dma_store:'Local HBM write',send_read:'Source HBM read',recv_write:'Receive HBM write',
    host_dma:'Host HBM DMA',host_in:'Host input link',host_out:'Host output link'}[s.name]||s.name;
}
function resourceLabel(r){
  if(r[0]==='D2D')return `D2D C${r[1]} → C${r[2]}`;
  if(r[0]==='HOST_LINK')return 'Host '+r[1];
  return r[0]+` C${r[1]}`+(r.length===3?` / ${r[2]===-1?'communication DMA':'PE '+r[2]}`:'');
}
function serviceColor(s){return serviceColors[s.name==='tensor'?'tensor':s.name==='simd'?'simd':
  s.name.startsWith('link_')?'network':s.name.startsWith('host_')&&s.name!=='host_dma'?'host':'memory'];}
// Category and operator choose the focus. Full communication views add actual
// local activity in a bounded window; transport-only views retain its links.
function simulationOperators(){
  return operators().filter(op=>Boolean(op.messages.length)===(state.simCategory==='communication'));
}
function simulationOperator(){
  const choices=simulationOperators();
  return choices.find(op=>op.key===state.key)||choices[0];
}
function isCommunicationService(s){
  return s.name.startsWith('link_')||['host_in','host_out'].includes(s.name);
}
function simulationView(){
  const op=simulationOperator(),key=`${state.stage}:${state.layer}:${op?.id}:${state.simPart}`;
  if(cachedSimulationView?.key===key)return cachedSimulationView.value;
  const own=op?op.service_indices.map(scheduledService).filter(s=>
    !(op.messages.length&&state.simPart==='communication')||isCommunicationService(s)):[];
  if(!own.length)return {services:[],operatorServices:[],begin:0,end:0,localContext:false};
  let begin=Math.min(...own.map(s=>s.ready_s)),end=Math.max(...own.map(s=>s.end_s));
  const localContext=Boolean(op.messages.length&&state.simPart==='all');
  let services=own;
  if(localContext){
    // This is a display window, never a new schedule. Start at transfer ready,
    // extend through the earliest directly consuming local operator, and show
    // every saved local service overlapping it on the communication chips.
    const consumer=Object.values(D.operators).filter(o=>!o.messages.length&&o.predecessors.includes(op.id))
      .sort((a,b)=>a.start_s-b.start_s||a.id.localeCompare(b.id))[0];
    if(consumer)end=Math.max(end,consumer.end_s);
    const chips=new Set(own.flatMap(serviceResources).filter(r=>r[0]!=='HOST_LINK').flatMap(r=>r[0]==='D2D'?r.slice(1):[r[1]]));
    const local=localServiceIndices.filter(i=>{
      const row=D.schedule.services[i];return chips.has(row[chipColumn])&&row[endColumn]>=begin&&row[startColumn]<=end;
    }).map(scheduledService);
    services=[...own,...local];
  }
  const value={services,operatorServices:own,begin,end,localContext};
  cachedSimulationView={key,value};return value;
}
function simulationServices(){return simulationView().services;}
function simulationService(){
  const {services,operatorServices}=simulationView();
  return services.find(s=>s.index===state.simService)||
    operatorServices.find(s=>isCommunicationService(s)&&s.resource_waits.length)||
    operatorServices.find(isCommunicationService)||operatorServices.find(s=>['tensor','simd'].includes(s.name))||operatorServices[0];
}
function simulationMessage(){
  const s=simulationService();
  return s?D.operators[s.owner].messages.find(m=>m.command_id===s.command_id):null;
}
function schedulingSummary(s){
  const duration=s.end_s-s.start_s,wait=s.start_s-s.ready_s;
  return `<dl class="schedule-facts"><div><dt>Prerequisites complete / ready</dt><dd>${time(s.ready_s)}</dd></div><div><dt>Resource wait after ready</dt><dd>${time(wait)}</dd></div><div><dt>Scheduled start → finish</dt><dd>${time(s.start_s)} → ${time(s.end_s)}</dd></div><div><dt>Service duration</dt><dd>${time(duration)}</dd></div></dl>`;
}
function pricingExplanation(s){
  const h=D.config.hardware;
  if(s.name==='tensor'||s.name==='simd'){
    const rate=s.name==='tensor'?h.tensor_flops_per_chiplet/h.pes_per_chiplet:h.simd_ops_per_pe_per_s;
    return `${num(s.repeat)} × compute launch ${time(h.compute_launch_s)} + ${num(s.operations)} ops / ${num(rate)} ops/s = ${time(s.end_s-s.start_s)}`;
  }
  if(s.name.startsWith('link_'))return `D2D latency ${time(h.d2d_latency_s)} + ${size(s.traffic_bytes)} / ${num(h.d2d_bandwidth_Bps/1e9)} GB/s D2D port = ${time(s.end_s-s.start_s)}; the hop uses no local HBM / NoC bandwidth`;
  if(['dma_load','dma_store','send_read','recv_write','host_dma'].includes(s.name)){
    const repeat=['dma_load','dma_store'].includes(s.name)?s.repeat:1;
    return `${num(repeat)} × DMA launch ${time(h.dma_launch_s)} + max(${size(s.traffic_bytes)} / ${num(h.hbm_bandwidth_Bps/1e9)} GB/s HBM, ${size(s.traffic_bytes)} / ${num(h.noc_bandwidth_Bps/1e9)} GB/s NoC) = ${time(s.end_s-s.start_s)}`;
  }
  return `Host link: latency + max(bytes / host bandwidth, bytes / NoC bandwidth) = ${time(s.end_s-s.start_s)}`;
}
function serviceOperandsMarkup(s){
  const op=D.operators[s.owner],{command,inputs,output}=commandOperands(s),example=operatorExample(op,s);
  const roles=new Map([...example.guide.inputs,...example.guide.outputs].map(row=>[row.tensor.tensor_id,row.label]));
  const list=(tensors)=>tensors.map(t=>`<div class="service-tensor"><strong>${esc(roles.get(t.tensor_id)||tensorLabel(t))}</strong><span>${esc(shape(t))}</span></div>`).join('');
  const payload=inputs.length?inputs:output?[output]:[];
  let source=list(inputs),destination=output?list([output]):'',action=example.guide.summary;
  if(s.name==='dma_load'){
    destination='<p>Same logical operands in PE scratch.</p>';action='Read the local inputs needed by this operation from HBM into PE scratch.';
  }else if(s.name==='dma_store'){
    source='<p>Computed result in PE scratch.</p>'+list(output?[output]:inputs);action=op.key.endsWith('_CACHE')?'Write the new token into the newest local cache position.':'Store this PE’s result slice in the local output tensor.';
  }else if(s.name==='send_read'){
    destination='<p>Outgoing communication buffer.</p>'+list(payload);action='Read the source tensor into the outgoing buffer; no addition occurs.';
  }else if(s.name.startsWith('link_')){
    const [,from,to]=s.name.match(/^link_(\d+)_(\d+)$/);
    source=`<p>C${from} outgoing buffer.</p>`+list(payload);destination=`<p>C${to} incoming buffer.</p>`+list(payload);action='Copy the message payload over this directed physical D2D link.';
  }else if(s.name==='recv_write'){
    source='<p>Received communication buffer.</p>'+list(payload);action='Write the received payload into destination HBM. A TP sum is a separate operation.';
  }else if(s.name==='host_in'){
    source='<p>Host activation.</p>'+list(payload);destination='<p>Chiplet communication buffer.</p>'+list(payload);action='Copy the activation over the host input link.';
  }else if(s.name==='host_out'){
    source='<p>Chiplet communication buffer.</p>'+list(payload);destination='<p>Host logits.</p>'+list(payload);action='Copy the complete logits over the host output link.';
  }else if(s.name==='host_dma'){
    source=command.kind==='host_load'?'<p>Host-link receive buffer.</p>'+list(payload):list(payload);
    destination=command.kind==='host_load'?list(payload):'<p>Host-link send buffer.</p>'+list(payload);action=command.kind==='host_load'?'Store the incoming activation in local HBM.':'Read the complete logits from local HBM for the host transfer.';
  }
  const matrix=command.matrix_shape;
  const calculation=matrix?`[${matrix[0]} × ${matrix[2]}] × [${matrix[2]} × ${matrix[1]}] → [${matrix[0]} × ${matrix[1]}]` : '';
  const slice=command.output_stop>command.output_start?`Saved output ${op.key.endsWith('_CACHE')?'token positions':'range'} [${command.output_start}, ${command.output_stop})`:'';
  return `<section class="service-basics"><div class="service-operand-grid"><div><h4>Input</h4>${source||'<p>Implicit local scratch operand.</p>'}</div><div><h4>Output</h4>${destination||'<p>Local scratch state.</p>'}</div></div><p class="service-operation"><strong>Operation</strong> · ${esc(action)}</p>${calculation?`<p class="service-calculation">Local matrix: ${esc(calculation)}</p>`:''}${slice?`<p>${esc(slice)}${s.pe_id!==null?' · PE '+s.pe_id:''}</p>`:''}</section>`;
}
function renderScheduleDetail(s,all){
  if(!s)return '';
  const op=D.operators[s.owner],m=op.messages.find(m=>m.command_id===s.command_id),c=serviceCommand(s),expert=serviceExpert(s);
  const deps=s.dependencies.map(([i,reason])=>({service:scheduledService(i),reason})),waits=s.resource_waits;
  const consumers=all.filter(next=>next.dependencies.some(([i])=>i===s.index));
  const commandPhases=op.service_indices.map(scheduledService).filter(p=>p.command_id===s.command_id);
  return `<article class="schedule-detail"><h3>${esc(description(op).label)} · ${esc(serviceLabel(s))}</h3><p>${esc(serviceLocation(s))}${s.pe_id!==null?' · PE '+s.pe_id:''}${expert!==null?' · expert '+expert:''} · ${time(s.start_s)} → ${time(s.end_s)}</p>
    ${serviceOperandsMarkup(s)}${m?`<p class="schedule-route">Route: ${m.route.length?m.route.map(chip=>'C'+chip).join(' → '):m.kind==='host_load'?'Host → C'+m.chiplet_id:'C'+m.chiplet_id+' → Host'}</p><div class="schedule-chain">${commandPhases.map(p=>`<button data-schedule-service="${p.index}" aria-pressed="${p.index===s.index}">${esc(serviceLabel(p))}</button>`).join('<span>→</span>')}</div>`:''}
    <button id="open-selected-operator">View this operator’s explanation →</button>
    <details class="schedule-advanced"><summary>Timing, dependencies and resource waits</summary><p>Service ${esc(s.id)} · ${esc(c.operation)}. Shapes above are logical operands; DMA moves the modeled slices/tiles, whose physical layout is not expanded.</p>${schedulingSummary(s)}<p class="schedule-formula">${esc(pricingExplanation(s))}</p><p><strong>Uses:</strong> ${serviceResources(s).map(resourceLabel).map(esc).join(' + ')}</p><p><strong>Shared availability:</strong> ${serviceReservations(s).map(resourceLabel).map(esc).join(' + ')}</p>
    <h4>Prerequisites</h4>${deps.length?`<ul class="schedule-prerequisites">${deps.map(({service:d,reason})=>`<li><button data-schedule-service="${d.index}">${esc(serviceLabel(d))} · ${d.id}</button><span>${esc(reason)} · finished ${time(d.end_s)}</span></li>`).join('')}</ul>`:'<p>No predecessor: ready at time zero.</p>'}
    <h4>Resource blockers</h4>${waits.length?`<div class="schedule-waits">${waits.map(w=>`<div><strong>${time(w[0])} → ${time(w[1])}</strong>${w[2].map(([r,i])=>{const b=scheduledService(i);return `<p>${esc(resourceLabel(D.schedule.resources[r]))} held by <button data-schedule-service="${i}">${esc(serviceLabel(b))} · ${b.id}</button></p>`;}).join('')}</div>`).join('')}</div>`:'<p>No occupied-resource wait was recorded.</p>'}
    ${consumers.length?`<h4>Following services</h4><div class="schedule-following">${consumers.map(c=>`<button data-schedule-service="${c.index}">${esc(serviceLabel(c))} · ${serviceLocation(c)}</button>`).join('')}</div>`:''}</details></article>`;
}
function renderScheduleChart(all,selected){
  const chart=$('system-timeline');
  if(!all.length){chart.innerHTML='';$('schedule-window').textContent='No services in this range.';return;}
  const {begin,end,localContext,operatorServices}=simulationView();
  const margin=168,width=1300,plot=width-margin-24,scale=plot/(end-begin||1),x=t=>margin+(t-begin)*scale;
  const rows=[],rowMap=new Map();
  const addRow=(id,label)=>{rowMap.set(id,rows.length);rows.push({id,label});};
  const participants=all.flatMap(serviceResources);
  const chips=[...new Set(participants.filter(r=>r[0]!=='HOST_LINK').flatMap(r=>r[0]==='D2D'?r.slice(1):[r[1]]))].sort((a,b)=>a-b);
  for(const chip of chips){
    addRow('MEMORY:'+chip,`C${chip} · HBM / NoC`);
    // Both resources share one display lane; the Set in lanes() removes duplicates.
    rowMap.set('HBM:'+chip,rows.length-1);rowMap.set('NOC:'+chip,rows.length-1);
    const pes=localContext?Array.from({length:D.config.hardware.pes_per_chiplet},(_,pe)=>pe):
      [...new Set(participants.filter(r=>['TENSOR','SIMD','SCRATCH'].includes(r[0])&&r[1]===chip&&r[2]>=0).map(r=>r[2]))].sort((a,b)=>a-b);
    pes.forEach(pe=>addRow(`PE:${chip}:${pe}`,`C${chip} · PE ${pe}`));
  }
  const links=new Map();
  all.forEach(s=>serviceResources(s).filter(r=>r[0]==='D2D'||r[0]==='HOST_LINK').forEach(r=>links.set(r.join(':'),r)));
  links.forEach((r,id)=>addRow(id,resourceLabel(r)));
  const y=row=>55+row*22,height=80+rows.length*22;
  const lanes=s=>[...new Set(serviceResources(s).flatMap(r=>{
    const id=['TENSOR','SIMD','SCRATCH'].includes(r[0])?`PE:${r[1]}:${r[2]}`:r.join(':');
    return rowMap.has(id)?[rowMap.get(id)]:[];
  }))];
  const primary=s=>lanes(s).find(row=>rows[row].id.startsWith('D2D:')||rows[row].id.startsWith('HOST_LINK:'))??
    lanes(s).find(row=>rows[row].id.startsWith('PE:'))??lanes(s)[0];
  const related=new Set(selected?[selected.index,...selected.dependencies.map(([i])=>i),
    ...D.operators[selected.owner].service_indices.filter(i=>scheduledService(i).command_id===selected.command_id),
    ...all.filter(s=>s.dependencies.some(([i])=>i===selected.index)).map(s=>s.index)]:[]);
  let html=`<defs><pattern id="wait-hatch" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0 6L6 0" stroke="#cb802f" stroke-width="1"/></pattern><marker id="schedule-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#4c645b"/></marker></defs>`;
  for(let tick=0;tick<=5;tick++){
    const t=begin+(end-begin)*tick/5;
    html+=`<line x1="${x(t)}" y1="35" x2="${x(t)}" y2="${height-20}" stroke="#e1e8e3"/><text x="${x(t)}" y="20" text-anchor="${tick===5?'end':'middle'}">${((t-begin)*1e6).toFixed(3)} µs</text>`;
  }
  rows.forEach((row,i)=>{html+=`<rect x="0" y="${y(i)-10}" width="${width}" height="22" fill="${i%2?'#f6f9f6':'#fff'}" fill-opacity=".65"/><text x="12" y="${y(i)+3}">${esc(row.label)}</text><line x1="${margin}" y1="${y(i)+9}" x2="${width-24}" y2="${y(i)+9}" stroke="#e8ede9"/>`;});
  all.filter(s=>s.end_s>=begin&&s.start_s<=end).forEach(s=>lanes(s).forEach(row=>{
    html+=`<rect class="schedule-bar" role="button" tabindex="0" data-service="${s.index}" data-operator-id="${s.owner}" data-context="${s.owner!==simulationOperator().id}" data-name="${s.name}" x="${x(Math.max(begin,s.start_s))}" y="${y(row)-6}" width="${Math.max(.9,(Math.min(end,s.end_s)-Math.max(begin,s.start_s))*scale)}" height="12" fill="${serviceColor(s)}" fill-opacity="${related.has(s.index)?1:.6}" stroke="${selected?.index===s.index?'#152e2b':'none'}" stroke-width="2"><title>${esc(serviceLabel(s))} · ${esc(description(D.operators[s.owner]).label)} · ${s.id} · ${time(s.start_s)} → ${time(s.end_s)}</title></rect>`;
  }));
  if(selected){
    const row=primary(selected);
    if(row!==undefined)selected.resource_waits.forEach(w=>{
      if(w[1]>begin)html+=`<rect class="schedule-wait" x="${x(Math.max(begin,w[0]))}" y="${y(row)-7}" width="${Math.max(.7,(w[1]-Math.max(begin,w[0]))*scale)}" height="14" fill="url(#wait-hatch)" stroke="#c17c30"><title>Ready but blocked: ${time(w[0])} → ${time(w[1])}</title></rect>`;
    });
    const displayed=new Set(all.map(s=>s.index));
    all.filter(s=>related.has(s.index)).forEach(s=>s.dependencies.forEach(([i])=>{
      if(!related.has(i)||!displayed.has(i))return;
      const prev=scheduledService(i),a=primary(prev),b=primary(s);if(a===undefined||b===undefined)return;
      if(s.start_s>=begin)html+=`<path class="schedule-dependency" d="M${x(Math.max(begin,prev.end_s))} ${y(a)} L${x(s.start_s)} ${y(b)}" fill="none" stroke="#4c645b" stroke-width="1.4" marker-end="url(#schedule-arrow)"><title>${prev.id} completes before ${s.id} starts</title></path>`;
    }));
  }
  chart.setAttribute('viewBox',`0 0 ${width} ${height}`);chart.innerHTML=html;
  chart.querySelectorAll('[data-service]').forEach(el=>{
    const activate=()=>selectScheduledService(+el.dataset.service);
    el.onclick=activate;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};
  });
  const op=simulationOperator();
  $('schedule-window').textContent=`Layer ${state.layer+1} · ${description(op).label} · ${operatorServices.length} operator services${localContext?' + '+(all.length-operatorServices.length)+' local activity services':''} · global origin ${time(begin)} · window ${time(end-begin)}. Bars are clipped to this window; details retain full saved intervals.`;
}
function selectScheduledService(index){
  const service=scheduledService(index),target=D.operators[service.owner];
  // Inspect visible PE context without replacing the communication focus.
  if(simulationServices().some(s=>s.index===index)){state.simService=index;state.route=null;render();return;}
  if(!operators().some(op=>op.id===target.id)){
    if(target.layer_id!==null){
      state.stage=D.stages.find(s=>s.layers.includes(target.layer_id)).index;state.layer=target.layer_id;
    }else{
      const owningStage=D.stages.find(s=>s.incoming.includes(target.id)||s.outgoing.includes(target.id));
      state.stage=owningStage.index;state.layer=owningStage.outgoing.includes(target.id)?owningStage.layers.at(-1):owningStage.layers[0];
    }
  }
  state.simCategory=target.messages.length?'communication':'local';
  if(!isCommunicationService(service))state.simPart='all';
  state.expert=serviceExpert(service);state.chip=service.chiplet_id;
  state.message=Math.max(0,target.messages.findIndex(m=>m.command_id===service.command_id));state.phase=0;
  state.key=target.key;state.simService=index;state.route=null;render();
}
function renderSimulation(){
  const panel=$('simulation-panel');panel.hidden=state.step!==2;if(state.step!==2)return;
  const scroll=$('system-timeline')?.parentElement;
  const scrollTop=scroll?.scrollTop||0,scrollLeft=scroll?.scrollLeft||0;
  const selectedOp=simulationOperator(),all=simulationServices().sort((a,b)=>a.start_s-b.start_s||a.index-b.index),selected=simulationService();
  if(selectedOp)state.key=selectedOp.key;
  if(state.simCategory==='local')state.simPart='all';
  if(selected)state.simService=selected.index;
  const choices=simulationOperators(),hasCommunication=Boolean(selectedOp?.messages.length);
  const serviceChoice=s=>{const expert=serviceExpert(s);return `${serviceLabel(s)} · ${serviceLocation(s)}${s.pe_id!==null?' / PE '+s.pe_id:''}${expert!==null?' / expert '+expert:''} · ${time(s.start_s)}`;};
  panel.innerHTML=`<div class="section-title"><h2>Compute, memory and D2D schedule</h2><p><a href="#layer-graph-panel">Operator graph ↑</a> · Select a category, one operator and the part to inspect. Its services share the saved global time axis.</p></div><div class="schedule-controls"><label class="schedule-control" for="schedule-category">View<select id="schedule-category"><option value="local" ${state.simCategory==='local'?'selected':''}>Local operators</option><option value="communication" ${state.simCategory==='communication'?'selected':''}>Communication operators</option></select></label><label class="schedule-control schedule-operator-control" for="schedule-operator">Operator<select id="schedule-operator" ${choices.length?'':'disabled'}>${choices.map(op=>`<option value="${op.id}" ${op.id===selectedOp?.id?'selected':''}>${esc(description(op).label)}</option>`).join('')}</select></label><label class="schedule-control" for="schedule-part">Part<select id="schedule-part" ${hasCommunication?'':'disabled'}><option value="all" ${state.simPart==='all'?'selected':''}>All activity (including local)</option><option value="communication" ${state.simPart==='communication'?'selected':''} ${hasCommunication?'':'disabled'}>Communication only</option></select></label></div>
    <p class="timeline-caption">${hasCommunication?'All activity shows this transfer plus every local PE / HBM service in its time window, including receiving computation. Communication only keeps its D2D / Host links.':'Local operators show their HBM read, PE computation and HBM write services.'}</p>
    <div class="schedule-legend"><span><i style="background:${serviceColors.tensor}"></i>Tensor · matrix multiply</span><span><i style="background:${serviceColors.simd}"></i>SIMD · vector / reduction</span><span><i style="background:${serviceColors.memory}"></i>HBM / NoC</span><span><i style="background:${serviceColors.network}"></i>Directed D2D hop</span><span><i style="background:${serviceColors.host}"></i>Host link</span><span class="wait-key">▨ Ready but blocked</span><span>→ Dependency</span></div><p id="schedule-window" class="timeline-caption"></p><div class="schedule-scroll"><svg id="system-timeline" role="group" aria-label="Selected operator compute memory and directed D2D service schedule"></svg></div><p class="timeline-caption">HBM / NoC share a lane. D2D hops occupy only their port lanes, so they can overlap local HBM / NoC work and other ports. Repeated bars are the same service; hatched intervals show recorded resource waits.</p>
    ${all.length?`<label class="schedule-service-choice" for="schedule-service">Inspect service<select id="schedule-service">${all.map(s=>`<option value="${s.index}" ${s.index===selected.index?'selected':''}>${esc(description(D.operators[s.owner]).label)} · ${esc(serviceChoice(s))}</option>`).join('')}</select></label>${renderScheduleDetail(selected,all)}`:'<p>No operator in this category at the selected layer.</p>'}`;
  $('schedule-category').onchange=e=>{state.simCategory=e.target.value;state.simPart='all';state.expert=null;resetExecution();render();};
  $('schedule-operator').onchange=e=>{state.key=D.operators[e.target.value].key;state.expert=null;resetExecution();render();};
  $('schedule-part').onchange=e=>{state.simPart=e.target.value;state.simService=null;render();};
  if($('schedule-service'))$('schedule-service').onchange=e=>selectScheduledService(+e.target.value);
  if($('open-selected-operator'))$('open-selected-operator').onclick=()=>{
    const owner=D.operators[selected.owner];state.key=owner.key;state.chip=selected.chiplet_id;state.expert=serviceExpert(selected);
    if(owner.layer_id!==null){state.stage=D.stages.find(s=>s.layers.includes(owner.layer_id)).index;state.layer=owner.layer_id;}
    const i=owner.messages.findIndex(m=>m.command_id===selected.command_id);state.message=Math.max(0,i);state.phase=0;selectStep(1);
  };
  panel.querySelectorAll('[data-schedule-service]').forEach(el=>{
    const activate=()=>selectScheduledService(+el.dataset.scheduleService);
    el.onclick=activate;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};
  });
  renderScheduleChart(all,selected);
  $('system-timeline').parentElement.scrollTop=scrollTop;
  $('system-timeline').parentElement.scrollLeft=scrollLeft;
}
