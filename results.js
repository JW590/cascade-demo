'use strict';

// Results: saved steady-state comparisons exported beside the runs (runs.js).
// The page reads them as recorded; it never reprices or extrapolates a point.
const resultSeries = [
  {id:'4x4', label:'CASCADE 4×4 · TP4 / PP4', color:'#2a78d6'},
  {id:'6x6', label:'CASCADE 6×6 · TP9 / PP4', color:'#eb6834'},
  {id:'paper', label:'CAL Fig. 5 reading (4×4)', color:'#1baf7a', dashed:true}
];
const ms = x => x===null||x===undefined ? '—' : (x*1e3).toFixed(3);
const ratio = x => x===null||x===undefined ? '—' : x.toFixed(2)+'×';

function resultPoint(results,array,batch,context){
  return results.rows.find(r=>r.array===array&&r.batch===batch&&r.context_tokens===context);
}
function resultValue(results,series,batch,context){
  if(series.id==='paper'){
    const row=results.rows.find(r=>r.array===results.paper_array&&r.batch===batch&&r.context_tokens===context);
    return row?.paper_speedup??null;
  }
  return resultPoint(results,series.id,batch,context)?.speedup??null;
}

function resultsChart(results,arrays,batches){
  const series=resultSeries.filter(s=>s.id==='paper'?batches.some(b=>resultValue(results,s,b,50000)!==null):arrays.includes(s.id));
  const values=series.flatMap(s=>batches.map(b=>resultValue(results,s,b,50000))).filter(v=>v!==null);
  const width=760,height=330,left=56,right=150,top=18,bottom=46,plotW=width-left-right,plotH=height-top-bottom;
  const ymax=Math.max(1.5,Math.ceil(Math.max(...values)*1.1/.5)*.5);
  const x=i=>left+(batches.length===1?plotW/2:i*plotW/(batches.length-1)),y=v=>top+plotH*(1-v/ymax);
  let svg=`<title>Steady-state speedup over one Rubin GPU versus batch, 50K past tokens</title>`;
  for(let v=0;v<=ymax+1e-9;v+=.5){
    svg+=`<line x1="${left}" x2="${left+plotW}" y1="${y(v)}" y2="${y(v)}" stroke="#e3e9e5" stroke-width="1"/><text class="axis-label" x="${left-8}" y="${y(v)+4}" text-anchor="end">${v.toFixed(1)}×</text>`;
  }
  svg+=`<line x1="${left}" x2="${left+plotW}" y1="${y(1)}" y2="${y(1)}" stroke="#7d8f88" stroke-width="1.5" stroke-dasharray="5 4"/><text class="axis-label" x="${left+plotW-4}" y="${y(1)+16}" text-anchor="end">GPU parity (1×)</text>`;
  batches.forEach((b,i)=>{svg+=`<text class="axis-label" x="${x(i)}" y="${top+plotH+20}" text-anchor="middle">${b}</text>`;});
  svg+=`<text class="axis-title" x="${left+plotW/2}" y="${height-6}" text-anchor="middle">Batch (50K past tokens)</text>`;
  const ends=[];
  series.forEach(s=>{
    const points=batches.map((b,i)=>({b,i,v:resultValue(results,s,b,50000)})).filter(p=>p.v!==null);
    if(!points.length)return;
    svg+=`<path d="${points.map((p,k)=>`${k?'L':'M'}${x(p.i)} ${y(p.v)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" ${s.dashed?'stroke-dasharray="6 4"':''}/>`;
    points.forEach(p=>{
      svg+=`<circle cx="${x(p.i)}" cy="${y(p.v)}" r="4.5" fill="${s.dashed?'#fff':s.color}" stroke="${s.dashed?s.color:'#fff'}" stroke-width="2"/>`;
      svg+=`<circle class="result-hit" data-series="${s.id}" data-batch="${p.b}" cx="${x(p.i)}" cy="${y(p.v)}" r="13" fill="transparent" tabindex="0" role="img" aria-label="${esc(s.label)}, batch ${p.b}: ${ratio(p.v)}"/>`;
    });
    const last=points.at(-1);ends.push({s,x:x(last.i),y:y(last.v)});
  });
  // Direct labels at line ends; nudge apart so none overlap.
  ends.sort((a,b)=>a.y-b.y).forEach((e,i,all)=>{if(i&&e.y-all[i-1].y<15)e.y=all[i-1].y+15;});
  ends.forEach(e=>{svg+=`<text class="series-label" x="${e.x+12}" y="${e.y+4}">${esc(e.s.label.replace('CASCADE ','').replace(' (4×4)',''))}</text>`;});
  const legend=series.map(s=>`<span><svg width="26" height="10" aria-hidden="true"><line x1="1" x2="25" y1="5" y2="5" stroke="${s.color}" stroke-width="2" ${s.dashed?'stroke-dasharray="6 4"':''}/><circle cx="13" cy="5" r="3.5" fill="${s.dashed?'#fff':s.color}" stroke="${s.color}" stroke-width="1.5"/></svg>${esc(s.label)}</span>`).join('');
  return `<figure class="results-chart"><div class="results-legend">${legend}</div><div class="chart-frame"><svg id="results-svg" viewBox="0 0 ${width} ${height}" role="group" aria-label="Steady-state speedup over one GPU versus batch">${svg}</svg><div id="results-tooltip" class="results-tooltip" hidden></div></div><figcaption>Speedup = GPU period / CASCADE steady-state period. Hover or focus a point for its saved values; the tables below list every point.</figcaption></figure>`;
}

function resultsTable(results,arrays,points,context){
  const head=`<tr><th>${context?'Past tokens':'Batch'}</th><th>GPU (ms)</th>${arrays.map(a=>`<th>${a.replace('x','×')} step (ms)</th><th>${a.replace('x','×')} period (ms)</th><th>${a.replace('x','×')} speedup</th>`).join('')}<th>CAL ${results.paper_array.replace('x','×')} speedup</th>${arrays.map(a=>`<th>${a.replace('x','×')} tokens/J</th>`).join('')}<th>GPU tokens/J</th></tr>`;
  const body=points.map(([batch,ctx])=>{
    const rows=arrays.map(a=>resultPoint(results,a,batch,ctx)),any=rows.find(Boolean);
    const paper=results.rows.find(r=>r.array===results.paper_array&&r.batch===batch&&r.context_tokens===ctx);
    const gpu=any?(any.gpu_latency_s===null?'OOM':ms(any.gpu_latency_s)):'—';
    const cells=rows.map(r=>r?`<td>${ms(r.step_latency_s)}</td><td>${ms(r.period_s)}</td><td><strong>${r.speedup===null?'GPU OOM':ratio(r.speedup)}</strong></td>`:'<td>—</td><td>—</td><td>—</td>').join('');
    const cal=paper?.paper_rubin_oom?'Rubin OOM':ratio(paper?.paper_speedup);
    const energy=rows.map(r=>`<td>${r?r.cascade_tokens_per_joule.toFixed(2):'—'}</td>`).join('');
    return `<tr><td>${context?num(ctx):batch}</td><td>${gpu}</td>${cells}<td>${cal}</td>${energy}<td>${any?.gpu_tokens_per_joule?any.gpu_tokens_per_joule.toFixed(2):'—'}</td></tr>`;
  }).join('');
  return `<div class="results-table-wrap"><table class="results-table"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

function renderResults(){
  const mount=$('results-workspace');
  if(state.step!==3)return;
  const results=window.CASCADE_INDEX?.results;
  if(!results||!results.rows.length){
    mount.innerHTML='<div class="results-panel"><p class="eyebrow">04 / RESULTS</p><h2>Results</h2><p>No comparison was exported with this demo. Export with <code>--comparison ARRAY=comparison.json</code>.</p></div>';
    return;
  }
  const arrays=resultSeries.map(s=>s.id).filter(id=>results.rows.some(r=>r.array===id));
  const batches=[...new Set(results.rows.filter(r=>r.context_tokens===50000).map(r=>r.batch))].sort((a,b)=>a-b);
  const contexts=[...new Set(results.rows.filter(r=>r.batch===32).map(r=>r.context_tokens))].sort((a,b)=>a-b);
  const tdp=[...new Map(results.rows.map(r=>[r.array,r.cascade_tdp_watts])).entries()].map(([a,w])=>`${num(Math.round(w))} W for ${a.replace('x','×')}`).join(', ');
  const missing=arrays.filter(a=>results.rows.filter(r=>r.array===a).length<results.rows.filter(r=>r.array===arrays[0]).length);
  mount.innerHTML=`<div class="results-panel"><p class="eyebrow">04 / RESULTS · CAL FIG. 5(A)–(D)</p><h2>Steady-state throughput versus one ${esc(results.gpu.replace(/^./,c=>c.toUpperCase()))} GPU</h2>
    <p class="results-lede">Every point is a saved simulator run of GPT-OSS-120B decode with FP16 weights and KV on both sides. CASCADE overlaps consecutive decode steps across its PP stages; one GPU runs steps in sequence, so its period is its step latency.</p>
    ${resultsChart(results,arrays,batches)}
    <h3>Batch sweep · 50K past tokens</h3>${resultsTable(results,arrays,batches.map(b=>[b,50000]),false)}
    <h3>Context sweep · batch 32</h3>${resultsTable(results,arrays,contexts.map(c=>[32,c]),true)}
    ${missing.length?`<p class="note">— marks a point without a saved ${missing.map(a=>a.replace('x','×')).join(', ')} run in this export; re-export after those runs finish.</p>`:''}
    <details class="results-definitions"><summary>How these numbers are defined</summary>
      <p><strong>Steady-state period.</strong> Evaluation schedules three back-to-back decode steps. A microbatch re-enters after its host output (host sampling and embedding take no time), and every PE keeps its scratch order across steps. The period is the interval between the last two step completions, so one step's PP drain overlaps the next step's fill. Every step attends to the same context.</p>
      <p><strong>Step latency.</strong> One isolated decode step from an idle device, as shown in the Simulation view.</p>
      <p><strong>Tokens/J.</strong> Batch / (period × TDP), with ${tdp} (CAL's 1,241 W for 16 chiplets, scaled per chiplet) and ${num(2300)} W for the GPU. Energy is not simulated.</p>
      <p><strong>CAL readings.</strong> Speedups read from the paper's Fig. 5 for its 4×4 device. The paper's absolute latencies sit below the FP16 bandwidth bound, so it likely stores MXFP4 experts; this simulator stores FP16 on both sides.</p>
      <p><strong>Scope.</strong> Both sides are uncalibrated analytical estimates. Routing is synthetic and balanced. GPU OOM marks a GPU HBM capacity overflow.</p>
    </details></div>`;
  const tooltip=$('results-tooltip'),frame=tooltip.parentElement;
  const show=el=>{
    const s=resultSeries.find(x=>x.id===el.dataset.series),batch=+el.dataset.batch;
    const row=s.id==='paper'?null:resultPoint(results,s.id,batch,50000),gpu=(row||resultPoint(results,arrays[0],batch,50000))?.gpu_latency_s;
    tooltip.innerHTML=`<strong>${esc(s.label)}</strong><br>Batch ${batch} · ${ratio(resultValue(results,s,batch,50000))}`+
      (row?`<br>Period ${ms(row.period_s)} ms · GPU ${ms(gpu)} ms<br>Step latency ${ms(row.step_latency_s)} ms`:'<br>Read from the paper figure');
    const box=el.getBoundingClientRect(),outer=frame.getBoundingClientRect();
    tooltip.hidden=false;
    tooltip.style.left=Math.min(box.left-outer.left+18,outer.width-tooltip.offsetWidth-4)+'px';
    tooltip.style.top=Math.max(0,box.top-outer.top-tooltip.offsetHeight-6)+'px';
  };
  mount.querySelectorAll('.result-hit').forEach(el=>{
    el.onmouseenter=()=>show(el);el.onfocus=()=>show(el);
    el.onmouseleave=()=>{tooltip.hidden=true;};el.onblur=()=>{tooltip.hidden=true;};
  });
}
