'use strict';
const D = window.CASCADE_DATA;
if (!D || D.schema_version !== 5) throw new Error('Export the current CASCADE operator dataset first.');
const colors = ['#127464', '#3f6ca8', '#9c6627', '#795b9d'];
const state = {step:0, stage:0, layer:0, chip:4, key:'Q.raw', expert:null, message:0, phase:0, route:null, graphExpanded:false, placementExpanded:false, simCategory:'local', simPart:'all', simService:null};
const $ = id => document.getElementById(id);
const esc = x => String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = x => new Intl.NumberFormat('en-US').format(x);
const time = x => x >= .001 ? (x*1000).toFixed(3)+' ms' : (x*1e6).toFixed(3)+' µs';
const size = x => x >= 1e9 ? (x/1e9).toFixed(2)+' GB' : x >= 1e6 ? (x/1e6).toFixed(2)+' MB' : x >= 1000 ? (x/1000).toFixed(2)+' KB' : x+' B';
const shape = tensor => '['+tensor.shape.join(' × ')+']';
const stage = () => D.stages[state.stage];
const block = () => stage().blocks.find(b=>b.index===state.layer);
const stageOf = chip => D.stages.find(s=>s.chips.includes(chip)).index;
const pos = chip => [160+(chip%4)*113,63+Math.floor(chip/4)*116];
const labels = ['Architecture','Operators','Simulation'];

function operators(){
  const ids=[...(state.layer===stage().layers[0]?stage().incoming:[]), ...block().operator_ids,
    ...(state.layer===stage().layers.at(-1)?stage().outgoing:[])];
  return ids.map(id=>D.operators[id]);
}
const operator = () => operators().find(o=>o.key===state.key)||operators()[0];
function boundaryOf(op){
  if(state.layer===stage().layers[0]&&stage().incoming.includes(op.id))return 'input';
  if(state.layer===stage().layers.at(-1)&&stage().outgoing.includes(op.id))return 'output';
  return null;
}
const description = op => describeOperator(op||operator(),D.config.model,block(),boundaryOf(op||operator()));
const instances = () => operator().instances.filter(i=>state.expert===null||i.expert_id===state.expert);
const message = () => operator().messages[state.message];
const isOperatorView = () => state.step===1;
function phases(){
  if(message())return message().service_indices.map(scheduledService).map(s=>({...s,type:'network',
    resources:serviceResources(s),label:
    s.name==='send_read'?'Source HBM read':s.name==='recv_write'?'Destination HBM write':
    s.name==='host_in'?'Host → chiplet link':s.name==='host_out'?'Chiplet → host link':
    s.name==='host_dma'?'Chiplet HBM DMA':s.name.replace(/^link_(\d+)_(\d+)$/,'D2D C$1 → C$2')}));
  return ['read','compute','write'].filter(name=>instances().some(i=>i.phases.some(p=>p.name===name))).map(name=>({
    name,type:'local',label:name==='read'?'Local HBM read':name==='write'?'Local HBM write':
      instances().some(i=>i.engines.includes('tensor_mma'))?
        (instances().some(i=>i.engines.includes('simd'))?'Tensor + SIMD compute':'Tensor compute'):'SIMD compute'
  }));
}
const phase = () => phases()[Math.min(state.phase,phases().length-1)];
function resetExecution(){state.message=0;state.phase=0;state.route=null;state.simService=null;}
function selectStep(step){state.step=step;state.route=null;if(step===2){state.simCategory=operator().messages.length?'communication':'local';if(state.simCategory==='local')state.simPart='all';}render();}
function selectOperator(id){
  const op=D.operators[id];state.key=op.key;state.expert=null;
  // Graph navigation can cross categories; align the filter before rendering
  // so simulationOperator() retains the clicked operation instead of falling back.
  if(state.step===2){state.simCategory=op.messages.length?'communication':'local';if(state.simCategory==='local')state.simPart='all';}
  resetExecution();render();
  if(state.step===2)$('simulation-panel').scrollIntoView({block:'start'});
}
function selectStage(index,chip){
  state.stage=index;state.layer=stage().layers[0];state.chip=chip??stage().chips[0];
  state.expert=null;resetExecution();render();
}
function selectMessage(index){
  state.message=Math.max(0,Math.min(index,operator().messages.length-1));state.phase=0;state.route=null;render();
}
function selectPhase(index){state.phase=Math.max(0,Math.min(index,phases().length-1));state.route=null;render();}
function selectedPaths(){
  if(state.route)return [state.route];
  if(state.step===2)return simulationMessage()?.route.length?[simulationMessage().route]:[];
  if(!isOperatorView())return [];
  return message()?.route.length?[message().route]:[];
}
function chipActivity(chip){
  if(state.step===2){
    const s=simulationService(),resources=s?serviceResources(s).filter(r=>r[0]!=='HOST_LINK'&&(r[0]==='D2D'?r.slice(1).includes(chip):r[1]===chip)):[];
    return {active:resources.length>0,label:'',slice:resources.map(resourceLabel).join(' + ')};
  }
  if(!isOperatorView())return {active:false,label:'',slice:''};
  const desc=description(),local=instances().filter(i=>i.chiplet_id===chip);
  if(local.length){
    const rank=stage().chips.indexOf(chip);
    return {active:true,label:'',slice:rankLayout(desc,rank,D.config.model)};
  }
  const m=message(),p=phase();
  if(m&&p){
    const link=p.name.match(/^link_(\d+)_(\d+)$/);
    const owners=link?[+link[1],+link[2]]:p.name==='send_read'?[m.route[0]]:
      p.name==='recv_write'?[m.route.at(-1)]:[m.chiplet_id];
    if(owners.includes(chip))return {active:true,label:'',slice:p.label+' · selected message'};
  }
  return {active:false,label:'',slice:''};
}

function renderTopology(){
  const paths=selectedPaths(),p=state.step===2?simulationService():phase();
  let svg=`<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="${colors[state.stage]}"/></marker></defs>`;
  D.stages.forEach(s=>{
    const points=s.chips.map(pos),x=Math.min(...points.map(v=>v[0]))-49,y=Math.min(...points.map(v=>v[1]))-53;
    svg+=`<rect x="${x}" y="${y}" width="211" height="215" rx="12" fill="${colors[s.index]}" fill-opacity="${s.index===state.stage?'.075':'.025'}" stroke="${colors[s.index]}" stroke-opacity="${s.index===state.stage?'.5':'.13'}" stroke-dasharray="5 4"/>`;
  });
  D.topology.physical_d2d_links.filter(([a,b])=>a<b).forEach(([a,b])=>{
    const [x1,y1]=pos(a),[x2,y2]=pos(b);
    svg+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#b9cac0" stroke-width="3"/>`;
  });
  svg+=`<rect x="11" y="172" width="64" height="126" rx="8" fill="#e8efea" stroke="#bdcdc1"/><text x="43" y="222" text-anchor="middle" style="font-size:12px;font-weight:600">HOST</text><text x="43" y="241" text-anchor="middle" style="font-size:8px">PCIe 6 ×16</text><text x="43" y="257" text-anchor="middle" style="font-size:8px">128 GB/s/dir</text>`;
  D.topology.host_chiplets.forEach(c=>{
    const [x,y]=pos(c),m=state.step===2?simulationMessage():message(),active=(isOperatorView()||state.step===2)&&m?.kind.startsWith('host_')&&m.chiplet_id===c;
    svg+=`<path d="M75 ${c===4?199:271} L98 ${y} L${x-38} ${y}" fill="none" stroke="${active?colors[state.stage]:'#b3c4b9'}" stroke-width="${active?5:2}"/>`;
  });
  paths.forEach(path=>path.slice(1).forEach((to,i)=>{
    const from=path[i],a=pos(from),b=pos(to),dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy),r=43;
    const active=state.route||p?.name===`link_${from}_${to}`;
    svg+=`<line class="transfer-hop" data-hop="${from}-${to}" x1="${a[0]+dx*r/len}" y1="${a[1]+dy*r/len}" x2="${b[0]-dx*r/len}" y2="${b[1]-dy*r/len}" stroke="${colors[state.stage]}" stroke-width="${active?5:3}" stroke-opacity="${active?1:.28}" marker-end="url(#arrow)"/>`;
  }));
  for(let chip=0;chip<16;chip++){
    const [x,y]=pos(chip),s=stageOf(chip),rank=D.stages[s].chips.indexOf(chip),chosen=chip===state.chip,a=chipActivity(chip);
    const rankLabel=`TP rank ${rank}`;
    svg+=`<g role="button" tabindex="0" data-chip="${chip}" aria-label="Chiplet ${chip}, pipeline stage ${s+1}, tensor rank ${rank}${a.label?', '+esc(a.label):''}"><title>C${chip}: ${esc(a.slice||a.label||`Stage ${s+1}, TP rank ${rank}`)}</title><rect x="${x-38}" y="${y-34}" width="76" height="75" rx="7" fill="${a.active?'#e3f0e7':'#fff'}" stroke="${colors[s]}" stroke-opacity="${chosen||a.active?1:.38}" stroke-width="${chosen?3:a.active?2:1}"/><text class="chip-id" x="${x}" y="${y-9}" text-anchor="middle">C${chip}${D.topology.host_chiplets.includes(chip)?' ★':''}</text><text class="chip-small" x="${x}" y="${y+8}" text-anchor="middle">${esc(rankLabel)}</text>${a.label?`<text class="chip-activity" x="${x}" y="${y+24}" text-anchor="middle">${esc(a.label)}</text>`:''}</g>`;
  }
  svg+='<text x="322" y="503" text-anchor="middle" class="chip-small">D2D · 256 GB/s per port, per direction</text>';
  $('topology').innerHTML=svg;
  $('topology').querySelectorAll('[data-chip]').forEach(el=>{
    const activate=()=>{
      const chip=+el.dataset.chip;
      if(stageOf(chip)!==state.stage)selectStage(stageOf(chip),chip);
      else {state.chip=chip;state.route=null;render();}
    };
    el.onclick=activate;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};
  });
  const activity=chipActivity(state.chip),rank=stage().chips.indexOf(state.chip);
  $('selected-chip').innerHTML=`<strong>C${state.chip} · Stage ${state.stage+1} · TP rank ${rank}</strong><br>${isOperatorView()?esc(activity.slice||'No local work for the selected operation'):`Layer ${state.layer+1}`}`;
  $('map-activity').hidden=state.step===0;
  $('map-activity').textContent=state.step===2?(simulationService()?serviceLabel(simulationService())+' · '+simulationService().id:'No service in this view'):isOperatorView()?description().label+' · '+(phase()?.label||''):'';
}

function renderLayerGraph(){
  const panel=$('layer-graph-panel'),mount=$(state.step===2?'simulation-graph-slot':'operator-graph-slot');
  // Both views use one graph and selection state, including actual PP boundaries.
  if(panel.parentElement!==mount)mount.appendChild(panel);
  panel.hidden=!(isOperatorView()||state.step===2);
  if(panel.hidden)return;
  const all=operators(),fused=all.some(op=>op.key==='flash_attention');
  const layout=(fused?layerGroups.filter(g=>!['softmax','context'].includes(g.id)).map(g=>
    g.id==='scores'?{...g,id:'flash_attention',label:'FlashAttention + sink'}:g):layerGroups).filter(g=>!['input','output'].includes(g.id));
  const incoming=all.filter(op=>boundaryOf(op)==='input'),outgoing=all.filter(op=>boundaryOf(op)==='output');
  const ids=new Set(all.map(op=>op.id));
  const externalInput=!incoming.length||all.some(op=>op.predecessors.some(id=>!ids.has(id)));
  // Stage input is a composite node with individually selectable transfer steps;
  // choosing only its first operator would hide TP broadcast from the graph.
  const nodeFor=op=>boundaryOf(op)==='input'?'input:staging':!boundaryOf(op)?description(op).group:
    op.key.startsWith('head.norm.')?'boundary:head.norm':'boundary:'+op.id;
  const groups=new Map(),add=(id,label,x,y)=>groups.set(id,{id,label,x,y,operators:[]});
  if(externalInput)add('input',incoming.length?'Input from previous PP stage':'Input from previous layer',150,54);
  if(incoming.length)add('input:staging','Stage input',150,98+100*(externalInput?1:0));
  const shift=incoming.length?100*(incoming.length+(externalInput?1:0)):0;
  layout.forEach(g=>add(g.id,g.label,g.x,g.y+shift));
  let tail=1054+shift;
  outgoing.forEach(op=>{
    const id=nodeFor(op);if(groups.has(id))return;
    add(id,op.key.startsWith('head.norm.')?'Final RMSNorm':description(op).label,420,tail);tail+=100;
  });
  if(!outgoing.length)add('output','Output to next layer',420,tail);
  all.forEach(op=>groups.get(nodeFor(op)).operators.push(op));
  const byId=new Map(all.map(op=>[op.id,nodeFor(op)])),selected=byId.get(operator().id);
  const edges=new Map();
  all.forEach(op=>op.predecessors.forEach(pred=>{
    const a=byId.get(pred)||'input',b=byId.get(op.id);
    if(a!==b)edges.set(a+':'+b,[a,b]);
  }));
  if(!outgoing.length)edges.set('residual2:output',['residual2','output']);
  let svg='<defs><marker id="tensor-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#91a79a"/></marker><marker id="tensor-arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#146954"/></marker></defs>';
  edges.forEach(([a,b])=>{
    const from=groups.get(a),to=groups.get(b),horizontal=from.y===to.y,active=a===selected||b===selected;
    const sign=Math.sign(to.x-from.x)||1,skip=Math.abs(from.y-to.y)>110;
    const x1=horizontal?from.x+sign*117:from.x,y1=horizontal?from.y:from.y+(from.id==='input:staging'?84:36);
    const x2=horizontal?to.x-sign*117:to.x,y2=horizontal?to.y:to.y-(to.id==='input:staging'?84:36);
    const right=from.x>=690||to.x>=690,rail=right?826:14,side=right?1:-1;
    const route=skip?`M${from.x+side*117} ${from.y} L${rail} ${from.y} L${rail} ${to.y} L${to.x+side*117} ${to.y}`:
      horizontal?`M${x1} ${y1} L${x2} ${y2}`:`M${x1} ${y1} C${x1} ${(y1+y2)/2} ${x2} ${(y1+y2)/2} ${x2} ${y2}`;
    svg+=`<path class="tensor-edge" data-edge="${a}:${b}" d="${route}" fill="none" stroke="${active?'#146954':'#bdccc2'}" stroke-width="${active?2.5:1}" stroke-dasharray="${Math.abs(from.y-to.y)>65?'4 3':''}" marker-end="url(#tensor-arrow${active?'-active':''})"><title>${esc(from.label)} → ${esc(to.label)}</title></path>`;
  });
  groups.forEach(g=>{
    const active=g.id===selected,clickable=g.operators.length>0;
    const messages=g.operators.flatMap(op=>op.messages),d2d=messages.filter(m=>m.kind==='d2d_send').length,host=messages.length-d2d;
    const badge=[d2d?'D2D ×'+d2d:'',host?'Host ×'+host:''].filter(Boolean).join(' · ');
    if(g.id==='input:staging'){
      svg+=`<g data-group="${g.id}" data-operator-ids="${esc(g.operators.map(op=>op.id).join(','))}" data-communication="true" role="group" aria-label="Stage input: individually selectable input and broadcast steps"><rect x="${g.x-117}" y="${g.y-84}" width="234" height="168" rx="9" fill="#f1ebfa" stroke="#8254a4" stroke-width="2"/><text x="${g.x}" y="${g.y-58}" text-anchor="middle" fill="#254137">Stage input</text>`;
      g.operators.forEach((op,i)=>{
        const top=g.y-40+i*64,chosen=op.id===operator().id,mark=communicationLabel(op);
        if(i&&op.predecessors.includes(g.operators[i-1].id))svg+=`<path class="tensor-edge" d="M${g.x} ${top-20} L${g.x} ${top}" stroke="#8254a4" marker-end="url(#tensor-arrow)"/>`;
        svg+=`<g data-graph-operator="${esc(op.id)}" role="button" tabindex="0" class="${chosen?'selected':''}" aria-label="${esc(description(op).label)}"><title>${esc(description(op).label)} · ${esc(mark)}</title><rect x="${g.x-110}" y="${top}" width="220" height="44" rx="6" fill="${chosen?'#146954':'#fff'}" stroke="${chosen?'#146954':'#8254a4'}"/><text x="${g.x}" y="${top+18}" text-anchor="middle" fill="${chosen?'#fff':'#254137'}">${esc(description(op).label)}</text><text class="communication-badge" x="${g.x}" y="${top+36}" text-anchor="middle">↔ ${esc(mark)}</text></g>`;
      });svg+='</g>';return;
    }
    const words=g.label.split(' '),lines=[''];
    words.forEach(word=>{const last=lines.length-1;if((lines[last]+' '+word).trim().length>22)lines.push(word);else lines[last]=(lines[last]+' '+word).trim();});
    svg+=`<g data-group="${g.id}" data-operator-ids="${esc(g.operators.map(op=>op.id).join(','))}" data-communication="${badge?'true':'false'}" class="${active?'selected':''} ${badge?'communication-node':''}" ${clickable?'role="button" tabindex="0"':''} aria-label="${esc(g.label)}${badge?', '+badge:''}${clickable?', '+g.operators.length+' operations':''}"><title>${esc(g.label)}${badge?' · '+badge:''}${clickable?' · '+g.operators.length+' selectable operations':' · local tensor boundary'}</title><rect x="${g.x-117}" y="${g.y-36}" width="234" height="72" rx="9" fill="${active?'#146954':badge?'#f1ebfa':clickable?'#fff':'#eef2ee'}" stroke="${active?'#146954':badge?'#8254a4':'#cddad1'}" stroke-width="${badge?2:1}" stroke-dasharray="${clickable?'':'3 3'}"/>${lines.map((line,i)=>`<text x="${g.x}" y="${g.y+(badge?-10:lines.length===1?5:-4)+i*18}" text-anchor="middle" fill="${active?'#fff':'#254137'}">${esc(line)}</text>`).join('')}${badge?`<text class="communication-badge" x="${g.x}" y="${g.y+27}" text-anchor="middle">↔ ${badge}</text>`:''}</g>`;
  });
  $('layer-flow').setAttribute('viewBox',`0 0 840 ${Math.max(...[...groups.values()].map(g=>g.y))+54}`);
  $('layer-flow').innerHTML=svg;
  $('layer-flow').querySelectorAll('[data-group][role="button"]').forEach(el=>{
    const activate=()=>selectOperator(groups.get(el.dataset.group).operators[0].id);
    el.onclick=activate;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};
  });
  $('layer-flow').querySelectorAll('[data-graph-operator]').forEach(el=>{
    const activate=()=>selectOperator(el.dataset.graphOperator);
    el.onclick=activate;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};
  });
  const activeExperts=[...new Set(D.operators[`l${state.layer}:expert.dispatch`].instances.map(i=>i.expert_id))];
  $('layer-graph-caption').textContent=`Layer ${state.layer+1} · ${block().attention_tokens===D.config.context_tokens+1?'full':'sliding-window'} attention · ${num(block().attention_tokens)} cached tokens · active experts ${activeExperts.join(', ')}. Stage input contains individually selectable input and TP broadcast steps.`;
  const scroll=$('layer-flow').parentElement;
  scroll.classList.toggle('expanded',state.graphExpanded);
  $('toggle-layer-graph').textContent=state.graphExpanded?'Compact graph':'Show complete graph';
  $('toggle-layer-graph').setAttribute('aria-expanded',state.graphExpanded);
  $('toggle-layer-graph').onclick=()=>{state.graphExpanded=!state.graphExpanded;renderLayerGraph();};
  if(!state.graphExpanded){
    const selectedGroup=groups.get(selected),scale=$('layer-flow').getBoundingClientRect().width/840;
    const y=selectedGroup.y*scale,x=selectedGroup.x*scale;
    if(y<scroll.scrollTop+20||y>scroll.scrollTop+scroll.clientHeight-20)scroll.scrollTop=y-scroll.clientHeight/2;
    if(x<scroll.scrollLeft+80||x>scroll.scrollLeft+scroll.clientWidth-80)scroll.scrollLeft=x-scroll.clientWidth/2;
  }
}

function operatorExample(op,service=null){
  const selectedMessage=service?op.messages.find(m=>m.command_id===service.command_id):op.messages[state.message]||op.messages[0];
  const expert=service?serviceExpert(service):state.expert;
  const chip=service?service.chiplet_id:state.chip;
  const local=op.instances.find(i=>i.chiplet_id===chip&&(expert===null||i.expert_id===expert))||
    op.instances.find(i=>i.chiplet_id===chip)||op.instances.find(i=>expert===null||i.expert_id===expert)||op.instances[0];
  const sample=selectedMessage?{inputs:selectedMessage.input_tensors.length?selectedMessage.input_tensors:[selectedMessage.output_tensor],
    outputs:[selectedMessage.output_tensor||selectedMessage.input_tensors[0]]}:local;
  const owner=selectedMessage?selectedMessage.route.length?selectedMessage.route.map(c=>'C'+c).join(' → '):
    selectedMessage.kind==='host_load'?'Host → C'+selectedMessage.chiplet_id:'C'+selectedMessage.chiplet_id+' → Host':
    `C${local.chiplet_id}${local.expert_id!==null?' · expert '+local.expert_id:''}${local.chiplet_id!==chip?' (no instance on C'+chip+')':''}`;
  const ownerBlock=op.layer_id===null?block():D.stages.flatMap(s=>s.blocks).find(b=>b.index===op.layer_id);
  return {sample,owner,guide:operatorTensorGuide(op,sample,D.config.model,ownerBlock,D.config)};
}
function operatorFormulaMarkup(op,service=null){
  const {owner,guide}=operatorExample(op,service),formula=operatorFormula(op,D.config.model,block(),D.config);
  const tensors=(title,rows)=>`<div class="tensor-guide-group"><h5>${title}</h5>${rows.map(row=>`<div class="tensor-guide-row"><strong>${esc(row.label)}</strong><span class="tensor-guide-shape">${esc(shape(row.tensor))}</span></div>`).join('')}</div>`;
  const dimensions=[...guide.inputs,...guide.outputs].map(row=>`<li><strong>${esc(row.label)}</strong>: ${row.tensor.shape.map((n,i)=>`${esc(row.axes[i])} = ${num(n)}`).join(' · ')}</li>`).join('');
  return `<section class="operator-math" data-formula-key="${esc(op.key)}"><p class="tensor-guide-owner">Shape example · ${esc(owner)}</p>${tensors('Inputs',guide.inputs)}<p class="tensor-guide-operation"><strong>Operation</strong> · ${esc(guide.summary)}</p>${tensors('Output',guide.outputs)}<details class="operator-explanation"><summary>Dimensions and computation steps</summary><ul class="tensor-guide-dimensions">${dimensions}</ul><div class="tensor-guide-flow">${esc(guide.flow)}</div><ol class="tensor-guide-steps">${guide.steps.map(step=>`<li>${esc(step)}</li>`).join('')}</ol></details><details class="operator-formula-details"><summary>Operator formula</summary><div class="operator-math-lines">${formula.lines.map(line=>`<div>${esc(line)}</div>`).join('')}</div><p class="formula-symbols">${esc(formula.symbols)}</p></details></section>`;
}
function operatorControls(){
  const all=operators(),op=operator(),index=all.indexOf(op),desc=description();
  const experts=[...new Set(op.instances.map(i=>i.expert_id).filter(e=>e!==null))].sort((a,b)=>a-b);
  return `<div class="operator-picker"><label for="operator-choice">Layer operation</label><select id="operator-choice">${all.map((o,i)=>`<option value="${o.id}" ${o.id===op.id?'selected':''}>${String(i+1).padStart(2,'0')} · ${esc(description(o).label)}${communicationLabel(o)?' · ↔ '+communicationLabel(o):''}</option>`).join('')}</select><div class="operator-navigation"><button id="previous-operator" ${index===0?'disabled':''}>← Previous operation</button><span>${index+1} / ${all.length}</span><button id="next-operator" ${index===all.length-1?'disabled':''}>Next operation →</button></div><button id="next-communication" class="communication-jump">↔ Next communication operation</button></div>${experts.length&&op.key!=='expert.accum'?`<div class="expert-choice"><label for="expert-choice">Active expert</label><select id="expert-choice"><option value="all">All ${experts.length} active experts</option>${experts.map(e=>`<option value="${e}" ${state.expert===e?'selected':''}>Expert ${e}</option>`).join('')}</select><span>Every selected expert uses all four TP ranks.</span></div>`:''}<div class="operator-summary"><h3>${esc(desc.label)}</h3>${operatorFormulaMarkup(op)}</div>`;
}
function wireOperatorControls(){
  $('operator-choice').onchange=e=>selectOperator(e.target.value);
  const all=operators(),index=all.indexOf(operator());
  $('previous-operator').onclick=()=>selectOperator(all[index-1].id);
  $('next-operator').onclick=()=>selectOperator(all[index+1].id);
  $('next-communication').onclick=()=>selectOperator((all.slice(index+1).find(o=>o.messages.length)||all.find(o=>o.messages.length)).id);
  if($('expert-choice')){
    $('expert-choice').value=state.expert===null?'all':state.expert;
    $('expert-choice').onchange=e=>{state.expert=e.target.value==='all'?null:+e.target.value;resetExecution();render();};
  }
}
function tensorLabel(t){
  const key=t.tensor_id.replace(/^mb0\.l\d+\.c\d+\./,'').replace(/^l\d+\.c\d+\./,'').replace(/^e\d+\./,'');
  const names={K_CACHE:'K cache',V_CACHE:'V cache',OP_PARTIAL:'O partial',RES1:'Attention residual',RES2:'Layer output',Q_ROPE:'Rotated Q',K_ROPE:'Rotated K'};
  return names[key]||key;
}
function renderMapping(){
  const desc=description(),local=instances();
  const partition=tensorPartitionSummary(operator(),D.config.model,block());
  return `${partition?`<div class="partition-summary"><strong>Global → TP shards</strong><p>${esc(partition)}</p></div>`:''}<div class="owner-grid">${stage().chips.map((chip,rank)=>{
    const records=local.filter(i=>i.chiplet_id===chip),sample=records[0];
    const experts=operator().key==='expert.accum'?[]:[...new Set(records.map(i=>i.expert_id).filter(e=>e!==null))];
    return `<article class="owner-card ${chip===state.chip?'chosen':''}" data-owner="${chip}"><h4>C${chip} · TP ${rank}</h4><p class="rank-layout">${esc(rankLayout(desc,rank,D.config.model))}</p>${sample?`<dl><dt>Inputs in local HBM</dt><dd>${sample.inputs.map(t=>`${esc(tensorLabel(t))} <code>${shape(t)}</code>`).join('<br>')}</dd><dt>Output in local HBM</dt><dd>${sample.outputs.map(t=>`${esc(tensorLabel(t))} <code>${shape(t)}</code>`).join('<br>')}</dd><dt>Compute placement</dt><dd>${sample.engines.map(k=>k==='tensor_mma'?'Tensor': 'SIMD').join(' + ')||'DMA only'} · ${new Set(sample.pe_slices.map(p=>p.pe_id)).size||1} PE${experts.length?' · experts '+experts.join(', '):''}</dd></dl>`:'<p>No command for this operation on this rank.</p>'}</article>`;
  }).join('')}</div><p class="note">Shapes are local to one chiplet${local.some(i=>i.expert_id!==null)?' and one active expert':''}. PEs split the local output; Tensor/SIMD timing remains coarse. Read/write bytes include the descriptor’s actual tile reloads.</p>`;
}
function messageTable(){
  return `<div class="message-list">${operator().messages.map((m,i)=>`<div class="${state.message===i?'current':''}"><span>${m.route.length?m.route.map(c=>'C'+c).join(' → '):m.kind==='host_load'?'Host → C'+m.chiplet_id:'C'+m.chiplet_id+' → Host'} · ${size(m.bytes)}</span><code>${time(m.end_s-m.start_s)}</code></div>`).join('')}</div>`;
}
function renderCommunication(){
  const all=phases(),p=phase(),op=operator(),m=message();
  let html='';
  if(m)html+=`<div class="exchange-controls"><button id="previous-exchange" ${state.message===0?'disabled':''} aria-label="Previous exchange">← Previous message</button><button id="next-exchange" ${state.message===op.messages.length-1?'disabled':''} aria-label="Next exchange">Next message →</button><label for="exchange">Message</label><input id="exchange" type="range" min="0" max="${op.messages.length-1}" value="${state.message}" ${op.messages.length<2?'disabled':''} aria-label="Communication exchange"><span id="exchange-count" class="tag" aria-live="polite">${state.message+1} / ${op.messages.length}</span></div>`;
  else html+='<p class="local-only">Local operation · no chiplet-to-chiplet transfer.</p>';
  html+=`<div class="phase-controls" aria-label="Execution phases">${all.map((item,i)=>`<button data-phase="${i}" aria-pressed="${i===state.phase}"><b>${i+1}</b>${esc(item.label)}</button>`).join('<span>→</span>')}</div><div class="phase-navigation"><button id="previous-phase" ${state.phase===0?'disabled':''}>← Previous phase</button><span>${state.phase+1} / ${all.length}</span><button id="next-phase" ${state.phase>=all.length-1?'disabled':''}>Next phase →</button></div>`;
  if(p?.type==='network'){
    const resources=p.resources.map(r=>r.join(' · ')).join('<br>');
    html+=`<div class="phase-detail"><h3>${esc(p.label)}</h3><p>${m.route.length?m.route.map(c=>'C'+c).join(' → '):m.kind==='host_load'?'Host → C'+m.chiplet_id:'C'+m.chiplet_id+' → Host'}</p><details class="phase-statistics"><summary>Message shape and timing</summary><dl class="phase-facts"><div><dt>Message tensor</dt><dd>${shape(m.output_tensor||m.input_tensors[0])} · ${size(m.bytes)}</dd></div><div><dt>Phase traffic</dt><dd>${size(p.traffic_bytes)}</dd></div><div><dt>Saved global interval</dt><dd>${time(p.start_s)} → ${time(p.end_s)}</dd></div></dl><p class="resource-line">${esc(resources).replaceAll('&lt;br&gt;','<br>')}</p></details></div>`;
    html+=`<details class="phase-statistics"><summary>All ${op.messages.length} messages</summary>${messageTable()}</details>`;
  }else if(p){
    const selected=instances().map(i=>({...i,selected:i.phases.find(item=>item.name===p.name)})).filter(i=>i.selected);
    const start=Math.min(...selected.map(i=>i.selected.start_s)),end=Math.max(...selected.map(i=>i.selected.end_s));
    html+=`<div class="phase-detail"><h3>${esc(p.label)}</h3><p>${p.name==='read'?'The DMA services read the operation’s local inputs and weights from HBM into local scratch.':p.name==='write'?'The result slices are stored back into the same chiplet’s HBM, ready for their consumers.':'Each active PE computes its assigned local output slice. Tensor operations use matrix tiles; SIMD operations handle elementwise work and local reductions.'}</p><details class="phase-statistics"><summary>Phase timing and totals</summary><dl class="phase-facts"><div><dt>Traffic across selected owners</dt><dd>${size(selected.reduce((sum,i)=>sum+i.selected.traffic_bytes,0))}</dd></div><div><dt>Saved global interval</dt><dd>${time(start)} → ${time(end)}</dd></div></dl><table class="mapping-table"><thead><tr><th>OWNER</th><th>TRAFFIC / WORK</th><th>RESOURCES</th></tr></thead><tbody>${stage().chips.map(chip=>{
      const active=selected.filter(i=>i.chiplet_id===chip);if(!active.length)return '';
      const bytes=active.reduce((sum,i)=>sum+i.selected.traffic_bytes,0),ops=active.reduce((sum,i)=>sum+i.selected.operations,0);
      return `<tr><td>C${chip}</td><td>${ops?num(ops)+' arithmetic ops':size(bytes)}</td><td>${esc(active[0].selected.resources.join(' + '))}</td></tr>`;
    }).join('')}</tbody></table></details></div>`;
  }
  return html;
}
function wireCommunication(){
  if($('exchange')){
    $('exchange').oninput=e=>selectMessage(+e.target.value);
    $('previous-exchange').onclick=()=>selectMessage(state.message-1);
    $('next-exchange').onclick=()=>selectMessage(state.message+1);
  }
  document.querySelectorAll('[data-phase]').forEach(el=>el.onclick=()=>selectPhase(+el.dataset.phase));
  $('previous-phase').onclick=()=>selectPhase(state.phase-1);
  $('next-phase').onclick=()=>selectPhase(state.phase+1);
}

function renderLesson(){
  let html;
  if(state.step===0){
    html='<p class="eyebrow">01 / ARCHITECTURE</p><h2>Chiplet architecture</h2><p>The physical D2D network connects 16 cHBM chiplets. Four TP chiplets cooperate within each of four PP stages.</p>';
    html+=`<div class="fact-row"><div class="fact"><strong>16</strong>cHBM chiplets</div><div class="fact"><strong>576 GB</strong>total HBM capacity</div><div class="fact"><strong>44 TB/s</strong>aggregate HBM bandwidth</div></div><h3>Local chiplet resources</h3><div class="chip-inside"><div class="hbm-block"><b>HBM4</b>36 GB · 2.75 TB/s</div><div class="pes">${[0,1,2,3].map(i=>`<div>PE ${i}<small>DMA · Tensor · SIMD</small></div>`).join('')}</div></div><p>Four PEs share local memory and NoC: each has one HBM / NoC port with a quarter of the chiplet bandwidth, so their transfers run concurrently. Each PE overlaps a tile group's DMA, Tensor and SIMD work; as in FuseMax, the Tensor array also runs softmax's max, exp and sum. The three D2D ports transmit concurrently and do not consume local HBM / NoC bandwidth. Tensor throughput is 22 TFLOPS per chiplet, split across its PEs in the coarse descriptor model.</p><p class="note">C4 and C8 connect to the host. The physical ports stay fixed as logical TP/PP groups are explored.</p>`;
    $('lesson').innerHTML=html;return;
  }
  if(!isOperatorView())return;
  $('lesson').innerHTML=`<p class="eyebrow">02 / OPERATORS</p><h2>Operator mapping and execution</h2><p>Select a layer and operation. Follow its tensor shapes, TP placement and local execution or physical communication phases.</p>`+
    operatorControls()+(operator().messages.length?'':`<details id="operator-placement" class="operator-placement" ${state.placementExpanded?'open':''}><summary>TP placement across four chiplets</summary>${renderMapping()}</details>`)+
    `<section class="operator-execution"><h3>Execution and data movement</h3>${renderCommunication()}</section>`;
  wireOperatorControls();wireCommunication();
  if($('operator-placement'))$('operator-placement').ontoggle=e=>{state.placementExpanded=e.target.open;};
  document.querySelectorAll('[data-owner]').forEach(el=>el.onclick=()=>{state.chip=+el.dataset.owner;state.route=null;render();});
}
function render(){
  const focusId=document.activeElement?.id;
  document.querySelectorAll('.steps [data-step]').forEach(el=>{
    el.classList.toggle('selected',+el.dataset.step===state.step);el.setAttribute('aria-current',+el.dataset.step===state.step?'step':'false');
  });
  $('stage-controls').innerHTML=D.stages.map(s=>`<button data-stage="${s.index}" style="--stage:${colors[s.index]}" aria-pressed="${s.index===state.stage}">Stage ${s.index+1}<span>Layers ${s.layers[0]+1}–${s.layers.at(-1)+1}</span></button>`).join('');
  document.querySelectorAll('[data-stage]').forEach(el=>el.onclick=()=>selectStage(+el.dataset.stage));
  $('layer-controls').innerHTML=`<label for="layer-choice">Inspect layer</label><select id="layer-choice">${stage().blocks.map(b=>`<option value="${b.index}" ${b.index===state.layer?'selected':''}>Layer ${b.index+1} · ${b.attention_tokens===D.config.context_tokens+1?'Full':'Window 128'} attention</option>`).join('')}</select>`;
  $('layer-choice').onchange=e=>{state.layer=+e.target.value;state.expert=null;resetExecution();render();};
  document.querySelector('.workspace').hidden=state.step===2;
  $('simulation-workspace').hidden=state.step!==2;
  renderLesson();renderSimulation();renderTopology();renderLayerGraph();
  $('step-label').textContent=`0${state.step+1} / 0${labels.length}`;
  $('next-step').textContent=state.step===labels.length-1?'Back to architecture ↺':'Next: '+labels[state.step+1]+' →';
  if(focusId&&$(focusId))$(focusId).focus({preventScroll:true});
}
function route(source,destination){
  const queue=[[source]],seen=new Set([source]);
  for(let i=0;i<queue.length;i++){
    const p=queue[i],u=p.at(-1);if(u===destination)return p;
    for(const [from,to] of D.topology.physical_d2d_links)if(from===u&&!seen.has(to)){seen.add(to);queue.push([...p,to]);}
  }
  return null;
}
$('latency').textContent=time(D.summary.latency_s);
$('artifact-hash').textContent='Full run SHA-256: '+D.provenance.artifact_sha256;
for(const id of ['route-from','route-to'])$(id).innerHTML=Array.from({length:16},(_,i)=>`<option value="${i}">C${i}</option>`).join('');
$('route-from').value=5;$('route-to').value=9;
$('check-route').onclick=()=>{
  const a=+$('route-from').value,b=+$('route-to').value;state.route=route(a,b);renderTopology();
  $('route-result').textContent=a===b?'Same chiplet: no D2D transfer required.':state.route?`${state.route.map(c=>'C'+c).join(' → ')} · ${state.route.length-1} physical hop(s). ${state.route.length>2?'No direct physical link; route through intermediate chiplets.':'Direct physical link exists.'}`:'No legal physical route.';
};
document.querySelectorAll('[data-step]').forEach(el=>el.onclick=()=>selectStep(+el.dataset.step));
$('next-step').onclick=()=>selectStep((state.step+1)%labels.length);
render();
