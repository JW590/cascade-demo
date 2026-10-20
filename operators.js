'use strict';

// Groups describe the readable block diagram. The exported tensor dependencies
// supply its edges; these positions do not prescribe execution order or timing.
const layerGroups = [
  ['input','Layer input',0,0], ['norm1','Attention RMSNorm',1,0],
  ['q','Q projection + bias',0,1], ['k','K projection + bias',1,1], ['v','V projection + bias',2,1],
  ['q_rope','Q YaRN RoPE',0,2], ['k_rope','K YaRN RoPE',1,2], ['v_cache','V cache append',2,2],
  ['k_cache','K cache append',0,3], ['scores','Q × Kᵀ',1,3], ['softmax','Attention softmax',2,3],
  ['context','Attention context',0,4], ['o','O projection',1,4], ['attention_reduce','Attention TP sum',2,4],
  ['o_bias','O bias',0,5], ['residual1','Attention residual',1,5], ['norm2','MoE RMSNorm',2,5],
  ['router','Expert router',0,6], ['dispatch','Local expert dispatch',1,6],
  ['up','Expert up + bias',0,7], ['gate','Expert gate + bias',1,7], ['swiglu','Clipped SwiGLU',2,7],
  ['down','Expert down',0,8], ['down_bias','Down bias · TP0',1,8], ['weight','Routing weight',2,8],
  ['combine','Local expert sum',0,9], ['ffn_reduce','MoE TP sum',1,9], ['residual2','MoE residual',2,9],
  ['output','Layer output',1,10]
].map(([id,label,col,row])=>({id,label,x:150+270*col,y:54+100*row}));

// TP shares come from the exported run: contiguous ranges that differ by at most one.
// With more ranks than KV heads (TP9 over eight heads) some ranks own no attention head.
const tpRanks = () => CASCADE_DATA.stages[0].chips.length;
const tpWord = () => ({4:'four',9:'nine'})[tpRanks()]||String(tpRanks());
const shareCounts = kind => CASCADE_DATA.rank_shares.map(s=>s[kind]?s[kind][1]-s[kind][0]:0);
const maxShare = kind => Math.max(...shareCounts(kind));
function countText(kind,scale=1){
  const counts=shareCounts(kind).filter(n=>n>0).map(n=>n*scale),low=Math.min(...counts),high=Math.max(...counts);
  return low===high?String(high):`${low}–${high}`;
}
const idleRanks = () => shareCounts('kv_heads').filter(n=>n===0).length;
const idleNote = () => idleRanks()?` ${idleRanks()} of ${tpRanks()} ranks own no KV head; they skip attention and only relay TP sums.`:'';
const isTreeReduce = key => /^(attention_reduce|ffn_reduce)\.(up|tree_sum|down)$/.test(key);

// Mark actual saved messages; local dispatch and local TP adds have no badge.
function communicationLabel(op){
  const d2d=op.messages.filter(m=>m.kind==='d2d_send').length;
  const host=op.messages.length-d2d;
  return [d2d?`D2D ×${d2d}`:'',host?`Host ×${host}`:''].filter(Boolean).join(' · ');
}

function describeOperator(op, model, block, boundary){
  const d=model.hidden_size, g=model.num_attention_heads/model.num_key_value_heads, kv=maxShare('kv_heads'), q=kv*g;
  const h=model.head_dim, s=maxShare('expert_columns'), n=block.attention_tokens, T=tpRanks(), ranks=tpWord();
  const item=(label,group,mode,equation,explanation)=>({label,group,mode,equation,explanation});
  const key=op.key;
  if(boundary){
    const entry=boundary==='input';
    if(key==='input.host')return item('Host activation input','input','host',`Host → C${op.messages[0].chiplet_id}: [1, ${d}]`,'The host supplies one activation vector. Embedding computation is outside this saved run.');
    if(key==='input.broadcast'||key==='stage.broadcast')return item('TP input broadcast',entry?'input':'output','broadcast',`One [1, ${d}] vector → ${T} identical TP inputs`,`The incoming chiplet forwards the activation along the physical tree. This happens once at a stage entry; layers within a stage retain the ${ranks} local copies.`);
    if(key==='stage.relay')return item(entry?'Incoming PP relay':'Outgoing PP relay',entry?'input':'output','relay',`[1, ${d}] → next PP stage`,`Only the final layer of a PP stage relays its activation. The next stage broadcasts the received vector to its ${ranks} TP ranks.`);
    if(key==='head.projection')return item('LM head projection','output','vocab',`[1, ${d}] × [${d}, ${countText('vocabulary')}] → one vocabulary shard`,`Each rank owns a contiguous 1/${T} share of the vocabulary columns. All ranks consume the same normalized final activation.`);
    if(key==='head.gather')return item('Gather vocabulary shards','output','gather',`${T} vocabulary shards → host-connected chiplet`,`${T-1} remote shards travel to the host-connected chiplet; its own shard remains local. These transfers are gather operations, so their values are concatenated, not summed.`);
    if(key==='head.concatenate')return item('Concatenate logits','output','master',`${T} shards × ${countText('vocabulary')} columns → ${model.vocab_size} logits`,`The host-connected chiplet concatenates the ${ranks} vocabulary slices in TP rank order.`);
    if(key==='head.host')return item('Host logits output','output','host',`${model.vocab_size} logits → host`,'The completed logits leave the host-connected chiplet. Sampling is outside this saved run.');
  }
  const norm=key.match(/^(norm[12]|head\.norm)\.(square|mean|scale|output)$/);
  if(norm){
    const group=boundary?'output':norm[1];
    const specs={
      square:['Square activations','x²','Square each feature locally.'],
      mean:['Mean of squares','mean(x²)','Reduce all hidden features within this chiplet.'],
      scale:['RMS normalization scale','1 / √(mean(x²) + ε)','Compute the local normalization scale.'],
      output:['Apply RMSNorm','x · scale · weight','Apply the scale and replicated normalization weight.']
    };
    const [label,equation,explanation]=specs[norm[2]];
    return item(label,group,'replicated',equation,explanation+` Each rank has the complete ${d}-feature input, so RMSNorm needs no cross-chiplet reduction.`);
  }
  for(const name of ['Q','K','V']){
    const heads=name==='Q'?q:kv, width=heads*h, mode=name==='Q'?'q':'kv';
    if(key===name+'.raw')return item(name+' projection',name.toLowerCase(),mode,`X[1, ${d}] × W${name}[${d}, ${width}] → ${heads} local heads`,`The full input is replicated. Weight columns and output heads are split across ${ranks} chiplets; each chiplet computes complete local heads.${idleNote()}`);
    if(key===name)return item(name+' projection bias',name.toLowerCase(),mode,`${name} + B${name}[${width}]`,'Add the bias slice for these local heads. The head partition is unchanged.');
    if(key===name+'_ROPE')return item(name+' YaRN RoPE',name.toLowerCase()+'_rope',mode,`YaRN(${name}, absolute position ${CASCADE_DATA.config.context_tokens})`,'Rotate each complete local head at the decode position. No values from another TP rank are needed.');
  }
  if(key==='K_CACHE'||key==='V_CACHE')return item(key[0]+' cache append',key[0].toLowerCase()+'_cache','kv',`${kv} heads × ${h} features → cache position ${n-1}`,'Read the new local token from HBM, then write its cache row on the same chiplet. This is a local DMA operation, with no Tensor/SIMD compute or D2D transfer.');
  const attention={
    flash_attention:['Sink-aware FlashAttention','flash_attention','online softmax(QKᵀ / √head_dim, sink) × V','Fuse score calculation, sink-aware online softmax and value aggregation. Only the final context is written to HBM; score and probability blocks stay in PE scratch. As in FuseMax, the Tensor array computes QKᵀ, PV and each score\'s max, exp (six MACs) and sum; SIMD rescales the running state per block and divides once, and both work concurrently. With one token per microbatch, each KV group\'s sequence is split over its PEs and SIMD merges the partial states, applying the sink once.'],
    scores:['Attention scores','scores',`Q[${q} heads] × Kᵀ[${kv} KV heads, ${n} tokens]`,'Each rank computes its own query heads; every query uses a complete local key head. Eight query heads share each KV head.'],
    scaled:['Scale attention scores','softmax','scores / √head_dim','Apply the attention scale independently to each local head.'],
    maximum:['Maximum with attention sink','softmax','max(scores, learned sink)','Find the stable-softmax maximum, including this head’s learned sink.'],
    exp:['Exponentiate shifted scores','softmax','exp(scores − maximum)','Exponentiate local scores after subtracting the maximum.'],
    denominator:['Softmax denominator with sink','softmax','Σ exp(scores − maximum) + exp(sink − maximum)','The sink contributes to the denominator; it has no value vector in the context product.'],
    av_raw:['Attention value product','context',`exp(scores)[${q}, ${n}] × V[${kv}, ${n}, ${h}]`,'Multiply unnormalized attention weights by the locally cached value heads.'],
    av:['Normalize attention context','context','context / denominator','Divide each local head’s context by its sink-aware denominator.']
  };
  if(attention[key]){
    const [label,group,equation,explanation]=attention[key];
    return item(label,group,'q',equation,explanation+' Heads remain local throughout attention; there is no attention-head exchange.');
  }
  if(key==='OP_PARTIAL')return item('O projection partial','o','partial',`local context[1, ${q*h}] × WO[${q*h}, ${d}] → partial[1, ${d}]`,`O weight rows follow the query-head partition. Each chiplet produces a full-width partial result; the ${ranks} partials must be summed.${idleNote()}`);
  const reduction=key.match(/^(attention_reduce|ffn_reduce)\.r([01])\.(exchange|sum)$/);
  if(reduction){
    const [,group,round,action]=reduction, first=round==='0';
    return item(`${group==='attention_reduce'?'Attention':'MoE'} TP round ${+round+1}: ${action==='exchange'?'exchange':'local add'}`,group,action==='exchange'?'exchange':first?'pair_sum':'replicated',
      action==='exchange'?`4 directed messages × [1, ${d}]`:first?'local partial + peer partial → pair sum':'two pair sums → sum of all four TP ranks',
      (first?'The first round pairs horizontal neighbors. ':'The second round pairs vertical neighbors. ')+
      (action==='exchange'?'Each rank sends its current vector to its partner. Inspect one actual message and its read/link/write services at a time; the visual steps do not serialize the simulated exchanges.':first?'Each chiplet adds the received partial to its own. This is still only a two-rank sum.':'Each chiplet adds the received pair sum. All four chiplets now hold the same complete result.'));
  }
  const tree=key.match(/^(attention_reduce|ffn_reduce)\.(up|tree_sum|down)$/);
  if(tree){
    const [,group,part]=tree,name=group==='attention_reduce'?'Attention':'MoE';
    if(part==='up')return item(`${name} TP sum: send partials up`,group,'exchange',`${op.messages.length} directed messages × [1, ${d}]`,
      `Each rank sends its current partial one step toward the stage's most central chiplet, along a shortest-path tree inside the ${Math.sqrt(T)} × ${Math.sqrt(T)} group; deeper ranks send first.${group==='attention_reduce'&&idleRanks()?' A rank without KV heads has no attention partial, so a head-less leaf sends nothing.':''} Inspect one actual message and its read/link/write services at a time.`);
    if(part==='tree_sum')return item(`${name} TP sum: local add`,group,'tree_sum','own partial + each received child partial',
      `A parent adds every received child partial to its own on SIMD. When the root has added its children, it holds the sum of all ${ranks} ranks.`);
    return item(`${name} TP sum: broadcast total`,group,'broadcast',`${op.messages.length} directed messages × [1, ${d}]`,
      `The root copies the complete sum back down the same tree, so all ${ranks} ranks hold equal complete results.`);
  }
  if(key==='O_BIASED')return item('O output bias','o_bias','replicated',`complete O result + BO[${d}]`,'Add the O bias after the TP sum, once to each replicated complete result. Adding it separately to every partial before summing would multiply the bias.');
  if(key==='RES1')return item('Attention residual add','residual1','replicated','layer input + complete attention output','Both operands are full-width local replicas. The residual addition runs on SIMD and requires no D2D transfer.');
  if(key==='router_logits.raw')return item('Expert router projection','router','replicated',`normalized input[1, ${d}] × router[${d}, ${model.num_local_experts}]`,'The small router is replicated. Each rank evaluates the same complete expert logits; expert weights are still tensor-partitioned.');
  if(key==='router_logits')return item('Expert router bias','router','replicated',`router logits + bias[${model.num_local_experts}]`,'Add the replicated router bias locally.');
  if(key==='routing_weights')return item('Select and normalize expert weights','router','replicated',`${model.num_experts_per_tok} selected experts of ${model.num_local_experts}`,'The command applies selected-expert softmax. Active expert IDs in this run come from deterministic synthetic assignments, not learned router decisions.');
  const experts={
    dispatch:['Expert dispatch','dispatch','replicated',`normalized input → active expert input[1, ${d}]`,'Every expert is TP-split across this same group. Dispatch is a local copy; this mapping has no expert-parallel all-to-all.'],
    'up.raw':['Expert up projection','up','expert_width',`input[1, ${d}] × Wup[${d}, ${s}] → up[1, ${s}]`,`Each rank owns a contiguous 1/${T} share of each active expert’s intermediate columns.`],
    up:['Expert up bias','up','expert_width',`up + Bup[${s}]`,'Add the matching intermediate-column bias slice.'],
    'gate.raw':['Expert gate projection','gate','expert_width',`input[1, ${d}] × Wgate[${d}, ${s}] → gate[1, ${s}]`,'Gate and up use the same intermediate-column partition.'],
    gate:['Expert gate bias','gate','expert_width',`gate + Bgate[${s}]`,'Add the gate bias for the local intermediate columns.'],
    glu:['Clipped SwiGLU','swiglu','expert_width',`clipped SwiGLU(gate, up) → [1, ${s}]`,'Clamp and combine matching local gate/up columns. All operands remain on the owning chiplet.'],
    down:['Expert down projection','down','partial',`local activation[1, ${s}] × Wdown[${s}, ${d}] → partial[1, ${d}]`,'Down weight rows follow the intermediate partition. Every chiplet produces a full-width partial expert result.'],
    down_biased:['Expert down bias','down_bias','rank_zero',`TP rank 0 partial + Bdown[${d}]`,'Only TP rank 0 adds this bias before the TP sum. This includes the bias exactly once in the final expert result.'],
    weighted:['Apply routing weight','weight','partial','expert partial × token’s expert weight',`Weight each active expert’s local partial before summing. The same routing weight is used by all ${ranks} TP ranks.`],
    accum:['Combine local expert partials','combine','partial',`Σ ${model.num_experts_per_tok} weighted expert partials`,`Add the active experts on each chiplet first. One TP all-reduce then sums the ${ranks} combined partials; there is no separate all-reduce for every expert.`]
  };
  if(key.startsWith('expert.')&&experts[key.slice(7)])return item(...experts[key.slice(7)]);
  if(key==='RES2')return item('MoE residual add','residual2','replicated','attention residual + complete MoE output','The complete TP-reduced MoE result is added to the residual on every rank. The next layer within this stage consumes these local copies.');
  throw new Error('Missing operator explanation: '+key);
}

// Walk through one saved chiplet/expert instance. Shapes come from the artifact;
// intermediate attention shapes below explain the math, not additional HBM tensors.
function operatorTensorGuide(op,sample,model,block,config){
  const key=op.key,inputs=sample.inputs,outputs=sample.outputs;
  const dims=t=>'['+t.shape.join(' × ')+']';
  const last=t=>t.shape[t.shape.length-1];
  const hidden=['batch','tokens','hidden features'];
  const query=['batch','local KV groups','query heads per KV group','query tokens','head features'];
  const kv=['batch','local KV heads','tokens','head features'];
  const scores=['batch','local KV groups','query heads per KV group','query tokens','cached positions'];
  const scalar=['batch','tokens','one scalar'];
  const rows=(tensors,labels,axes)=>tensors.map((t,i)=>({tensor:t,label:labels[i],axes:axes[i]}));
  const guide=(summary,steps,inputLabels,outputLabels,inputAxes,outputAxes,flow)=>({
    summary,steps,flow,inputs:rows(inputs,inputLabels,inputAxes),outputs:rows(outputs,outputLabels,outputAxes)});
  const same=(summary,steps,labels,axes=hidden)=>guide(summary,steps,labels,['Result'],inputs.map(()=>axes),[axes],
    inputs.map(dims).join(' with ')+' → '+outputs.map(dims).join(', '));
  const norm=key.match(/^(norm[12]|head\.norm)\.(square|mean|scale|output)$/);
  if(norm){
    const n=norm[2];
    if(n==='square')return same('Square every hidden feature independently.',[
      `Each token has ${last(inputs[0])} hidden features. Replace each value x with x × x.`,
      'No axis is removed; the squared vector has the same shape.'],['Activation']);
    if(n==='mean')return guide('Reduce the already squared vector to one mean per token.',[
      `Sum the ${last(inputs[0])} squared features, then divide by ${last(inputs[0])}.`,
      'Only the hidden-feature axis shrinks to 1; batch and token axes remain.'],['Squared activation'],['Mean square'],[hidden],[scalar],`${dims(inputs[0])} → ${dims(outputs[0])}`);
    if(n==='scale')return same('Turn the mean square into one normalization scale per token.',[
      `Add ε = ${config.rms_norm_epsilon}, take the square root, then take its reciprocal.`,
      'The scale is a scalar for each token and will be shared by all its hidden features.'],['Mean square'],scalar);
    return guide('Scale each feature and apply its learned normalization weight.',[
      'Broadcast the one scalar scale across the hidden-feature axis.',
      `Multiply each of the ${last(outputs[0])} original features by that scale and its own learned weight. Shape stays unchanged.`],
    ['Original activation','Normalization scale','Learned feature weights'],['Normalized activation'],[hidden,scalar,['hidden features']],[hidden],`${dims(inputs[0])} × scalar × ${dims(inputs[2])} → ${dims(outputs[0])}`);
  }
  const linear=['Q.raw','K.raw','V.raw','OP_PARTIAL','router_logits.raw','expert.up.raw','expert.gate.raw','expert.down','head.projection'];
  if(linear.includes(key)){
    const [m,n,k]=sample.pe_slices[0].matrix_shape;
    const axis=key==='Q.raw'?query:['K.raw','V.raw'].includes(key)?kv:
      ['expert.up.raw','expert.gate.raw'].includes(key)?['expert token group','tokens','local intermediate features']:
      key==='router_logits.raw'?['batch','tokens','expert scores']:key==='head.projection'?['batch','tokens','local vocabulary columns']:hidden;
    const partial=['OP_PARTIAL','expert.down'].includes(key);
    const steps=[`View the activation as ${m} row(s) of ${k} features. The local weight has ${k} input rows and ${n} output columns.`,
      `For each output column, sum ${k} activation × weight products. This produces ${m} × ${n} values.`];
    if(key==='Q.raw')steps.push(`Store the ${n} features as ${n/model.head_dim} complete query heads × ${model.head_dim} features. The two head axes group queries by their shared KV head; this reshape does not add computation.`);
    if(['K.raw','V.raw'].includes(key))steps.push(`Store the ${n} features as ${n/model.head_dim} complete KV heads × ${model.head_dim} features.`);
    if(partial)steps.push(`The ${n}-feature result covers the full hidden width but contains only this rank's contribution. A later TP sum adds all ${tpWord()} ranks; this projection performs no D2D transfer.`);
    else if(key==='router_logits.raw')steps.push('Each rank computes the full set of expert scores; these columns are replicated across TP ranks.');
    else steps.push('This rank owns these output columns. The stored shape preserves the batch/token grouping.');
    return guide(partial?'Project a local feature shard into a full-width partial result.':'Multiply an activation by the local projection weight.',steps,
      ['Activation','Local projection weight'],[partial?'Full-width partial result':'Projected activation'],
      [key==='OP_PARTIAL'?query:key==='expert.down'?['expert token group','tokens','local intermediate features']:key.startsWith('expert.')?['assigned token rows','singleton token axis','hidden features']:hidden,['input features','output features']],[axis],
      `[${m} × ${k}] × [${k} × ${n}] → [${m} × ${n}] → stored ${dims(outputs[0])}`);
  }
  if(['Q','K','V','O_BIASED','router_logits','expert.up','expert.gate','expert.down_biased'].includes(key)){
    const axis=key==='Q'?query:['K','V'].includes(key)?kv:['expert.up','expert.gate'].includes(key)?['expert token group','tokens','local intermediate features']:key==='router_logits'?['batch','tokens','expert scores']:hidden;
    return guide(key==='expert.down_biased'?'Add the learned down-projection bias on TP rank 0, so the TP sum includes it once.':'Add one learned bias to each matching output feature.',[
      'Broadcast the bias across token rows; for Q/K/V, its flat features match the stored head layout.',
      key==='expert.down_biased'?'Only TP rank 0 has this command. Other ranks keep their unbiased partial, so the later TP sum includes the bias once.':'Add corresponding values. Batch, head/token and feature counts remain unchanged.'],
    ['Projection result','Bias'],['Biased result'],[axis,['matching output features']],[axis],`${dims(inputs[0])} + bias ${dims(inputs[1])} → ${dims(outputs[0])}`);
  }
  if(key==='Q_ROPE'||key==='K_ROPE')return guide('Rotate pairs of features inside each complete head.',[
    `Each head has ${model.head_dim} features. Pair feature j with j + ${model.head_dim/2}, giving ${model.head_dim/2} pairs.`,
    `Use the absolute decode position ${config.context_tokens} and the YaRN frequency for each pair to rotate its two values. The optional formula gives the frequency and scale.`,
    'Every pair still produces two values. No head, token or feature axis changes, and all work stays local.'],
  [key==='Q_ROPE'?'Query heads':'Key heads'],['Rotated heads'],[key==='Q_ROPE'?query:kv],[key==='Q_ROPE'?query:kv],`${dims(inputs[0])} → pairwise rotation → ${dims(outputs[0])}`);
  if(key==='K_CACHE'||key==='V_CACHE')return guide('Store the new token in the local KV cache.',[
    `The input supplies one ${model.head_dim}-feature vector per local KV head.`,
    `Write it into the newest position of the ${outputs[0].shape[2]} visible cached positions. Earlier positions were already resident; this is not a broadcast of the new vector to all positions.`,
    'The output shape names the visible cache after the update. This operation has no chiplet-to-chiplet transfer.'],
  ['New token heads'],['Visible local cache'],[kv],[['batch','local KV heads','visible cached positions','head features']],`${dims(inputs[0])} → update newest cache position → ${dims(outputs[0])}`);
  if(key==='flash_attention'){
    // Split-sequence groups (one token per microbatch) add FP32 partial states that SIMD merges.
    const isPartial=t=>/\.partial\d+$/.test(t.tensor_id),partials=inputs.filter(isPartial);
    const core=inputs.filter(t=>!isPartial(t)),finals=outputs.filter(t=>!isPartial(t));
    const n=core[1].shape[2],f=model.head_dim,heads=core[0].shape[1]*core[0].shape[2];
    const merge=partials.length?[`Here each KV group's ${n} positions are split over ${partials.length} PEs. Each PE writes an FP32 partial state per head (weighted value, denominator, running maximum); SIMD merges the ${partials.length} states and applies the sink once.`]:[];
    const view=guide('For each query head, summarize its matching cached value vectors.',[
      `One query [1 × ${f}] compares with its KV group's keys [${n} × ${f}]. Dot products and division by √${f} give ${n} scores: [1 × ${n}].`,
      `Normalize those scores together with one learned sink logit. There are ${n} real-token weights plus one sink weight; the sink contributes no value vector.`,
      `Multiply the real-token weights [1 × ${n}] by values [${n} × ${f}] to get one context [1 × ${f}]. Repeat for the ${heads} local query heads; every ${inputs[0].shape[2]} query heads share one KV head.`,
      `FlashAttention processes at most ${config.flash_attention_block_tokens} cached positions per block and retains running state in scratch. The [1 × ${n}] score/weight shapes explain the equivalent calculation; complete score/weight tensors are not stored in HBM.`,...merge],
    ['Query heads','Cached keys','Cached values','Learned sink logits'],['Context per query head'],[query,['batch','local KV heads','visible cached positions','head features'],['batch','local KV heads','visible cached positions','head features'],['batch','local KV groups','query heads per KV group','query tokens','one sink logit']],[query],
    `Per head: [1 × ${f}] × [${f} × ${n}] → scores [1 × ${n}] → weights × [${n} × ${f}] → context [1 × ${f}]`);
    const axes=[query,['batch','local KV heads','visible cached positions','head features'],['batch','local KV heads','visible cached positions','head features'],['batch','local KV groups','query heads per KV group','query tokens','one sink logit']];
    return {...view,inputs:rows(core,['Query heads','Cached keys','Cached values','Learned sink logits'],axes),outputs:rows(finals,['Context per query head'],[query])};
  }
  const attention={
    scores:['Compare each query with every visible key.',['Dot products reduce the head-feature axis and create one score per cached position.','Each query uses the KV head of its GQA group.'],['Query heads','Cached keys'],[query,kv],scores],
    scaled:['Scale attention scores.',[`Divide each score by √${model.head_dim}. The cached-position axis is unchanged.`],['Attention scores'],[scores],scores],
    maximum:['Find a stable softmax maximum per query.',['Reduce the cached-position axis, including the learned sink logit, to one maximum.'],['Scaled scores','Learned sink logits'],[scores,query.slice(0,4).concat('one sink logit')],query.slice(0,4).concat('one maximum')],
    exp:['Compute unnormalized attention weights.',['Broadcast the maximum over cached positions, subtract it, then exponentiate each score.'],['Scaled scores','Maximum per query'],[scores,query.slice(0,4).concat('one maximum')],scores],
    denominator:['Sum real-token and sink weights.',['Sum the cached-position axis and add the sink exponential once; retain one denominator per query.'],['Unnormalized weights','Maximum per query','Learned sink logits'],[scores,query.slice(0,4).concat('one maximum'),query.slice(0,4).concat('one sink logit')],query.slice(0,4).concat('one denominator')],
    av_raw:['Sum cached values using unnormalized weights.',['For each query, [1 × cached positions] multiplies [cached positions × head features].','The cached-position axis is reduced; one vector of head features remains.'],['Unnormalized weights','Cached values'],[scores,kv],query],
    av:['Normalize the weighted context.',['Broadcast one denominator per query over its head features, then divide each context value.'],['Unnormalized context','Denominator per query'],[query,query.slice(0,4).concat('one denominator')],query]
  };
  if(attention[key]){
    const [summary,steps,labels,axes,outAxis]=attention[key];
    return guide(summary,steps,labels,['Result'],axes,[outAxis],`${inputs.map(dims).join(' with ')} → ${dims(outputs[0])}`);
  }
  if(/^(attention_reduce|ffn_reduce)\.r[01]\.exchange$/.test(key))return same('Copy a TP partial to its partner rank.',[
    'Transfer the same-shaped vector along the saved physical route: source HBM read → D2D hop(s) → destination HBM write.',
    'Receiving preserves the feature values and shape. The separate local sum operation adds the received vector.'],['Source TP partial']);
  if(/^(attention_reduce|ffn_reduce)\.r[01]\.sum$/.test(key))return same('Add the received TP partial to the local partial.',[
    `Add corresponding features of the two ${last(outputs[0])}-feature vectors after the receive has completed. This step is local SIMD.`,
    key.includes('.r0.')?'The result includes two ranks. A second exchange and sum will combine all four ranks.':'Each output is now a complete four-rank result. The four chiplets retain equal-shaped replicas.'],['Local partial','Received partial']);
  if(/^(attention_reduce|ffn_reduce)\.(up|down)$/.test(key))return same(key.endsWith('.up')?'Copy a TP partial one step toward the tree root.':'Copy the complete TP sum one step down the tree.',[
    'Transfer the same-shaped vector along the saved physical route: source HBM read → D2D hop(s) → destination HBM write.',
    key.endsWith('.up')?'Receiving preserves the feature values and shape. The parent adds it in the separate local sum operation.':'Every rank receives the same complete result; copying does not change any feature values.'],[key.endsWith('.up')?'Child TP partial':'Complete TP sum']);
  if(/^(attention_reduce|ffn_reduce)\.tree_sum$/.test(key))return same('Add a received child partial to the parent’s partial.',[
    `Add corresponding features of the two ${last(outputs[0])}-feature vectors after the receive has completed. This step is local SIMD.`,
    `A parent repeats this for each child. After the root's last add, its vector is the sum of all ${tpWord()} ranks.`],['Parent partial','Received child partial']);
  if(key==='routing_weights')return guide('Normalize the fixed selected experts’ weights in a full expert-width vector.',[
    `The input has ${last(inputs[0])} expert scores per token. This saved performance run uses ${config.expert_routes?'provided':'synthetic balanced'} expert assignments, rather than computing a learned top-k decision.`,
    `Normalize only the ${model.num_experts_per_tok} selected expert scores. Store their weights in the matching expert columns and put zero in the other columns.`,
    `The expert axis keeps its full width; this is not a compact ${model.num_experts_per_tok}-column output.`],['All expert scores'],['Sparse routing weights'],[['batch','tokens','all experts']],[['batch','tokens','all experts']],`${dims(inputs[0])} → ${model.num_experts_per_tok} active entries per token → ${dims(outputs[0])}`);
  if(key==='expert.dispatch')return guide('Gather token rows assigned to this expert.',[
    'Use the saved expert assignments to choose token rows; keep each row’s full hidden-feature vector.',
    `This example groups ${outputs[0].shape[0]} assigned token row(s). With this batch-1 run, its stored tensor has shape ${dims(outputs[0])}.`,
    `The copy stays inside the owning chiplet. Experts are TP-sharded across the same ${tpWord()} ranks; this is not an EP all-to-all.`],
  ['Normalized activation','Routing weights'],['Expert token group'],[hidden,['batch','tokens','all experts']],[['assigned token rows','singleton token axis','hidden features']],`${dims(inputs[0])} → gather assigned rows → ${dims(outputs[0])}`);
  if(key==='expert.glu')return same('Combine matching gate and up features element by element.',[
    `Both branches have ${last(outputs[0])} local intermediate features. Cap gate only above at ${model.swiglu_limit}; clip up to [−${model.swiglu_limit}, ${model.swiglu_limit}].`,
    `For each feature, compute gate × sigmoid(${model.swiglu_alpha} × gate) × (up + 1).`,
    'One gate/up pair produces one feature; the intermediate width and TP partition remain unchanged.'],['Gate branch','Up branch'],['expert token group','tokens','local intermediate features']);
  if(key==='expert.weighted')return guide('Apply this expert’s token weight and restore the token’s batch slot.',[
    'Read the scalar weight at this expert’s column in the routing vector.',
    `Multiply every feature of the ${last(inputs[0])}-feature expert partial by that scalar; scatter the row back to its original batch slot. Unassigned slots are zero.`,
    'This remains a TP partial. Other experts combine locally, then a separate TP reduction combines the ranks.'],
  ['Expert partial','Routing weights'],['Weighted partial in batch slots'],[['assigned token rows','singleton token axis','hidden features'],['batch','tokens','all experts']],[hidden],`${dims(inputs[0])} × one expert weight → ${dims(outputs[0])}`);
  if(key==='expert.accum')return same('Add the next weighted expert partial to the local running sum.',[
    `Add corresponding features of two ${last(outputs[0])}-feature vectors; the output width stays the same.`,
    `Repeat for the remaining active experts on this rank. The later MoE TP sum combines all ${tpWord()} ranks.`],['Previous expert sum / first contribution','Next expert contribution']);
  if(key==='RES1'||key==='RES2')return same('Add the completed block result to its residual activation.',[
    'Both inputs already have the full hidden width and are local to the same chiplet.',
    'Add corresponding features. This preserves the activation shape for the next block or layer.'],['Residual activation','Complete block result']);
  if(key==='head.concatenate')return guide('Join the vocabulary shards along their column axis.',[
    `Each of the ${inputs.length} inputs contains ${last(inputs[0])} different vocabulary columns. They have already been gathered on the master chiplet.`,
    `Concatenate in TP rank order: ${inputs.length} × ${last(inputs[0])} = ${last(outputs[0])} vocabulary logits. Batch and token axes stay unchanged.`,
    'Columns are placed next to each other; values are not added.'],inputs.map((_,i)=>`TP ${i} vocabulary shard`),['Complete vocabulary logits'],inputs.map(()=>['batch','tokens','local vocabulary columns']),[['batch','tokens','all vocabulary columns']],`${inputs.length} × ${dims(inputs[0])} → concatenate columns → ${dims(outputs[0])}`);
  const transfers={
    'input.host':['Copy the host activation into master-chiplet HBM.','Embedding computation is outside this saved run.'],
    'input.broadcast':['Copy the incoming activation to the TP ranks.','Each destination gets the same full hidden-width activation; copying does not split or add its features.'],
    'stage.broadcast':['Copy the received PP activation to the new stage’s TP ranks.',`The ${tpWord()} ranks receive full hidden-width replicas before executing their local shards.`],
    'stage.relay':['Copy the completed layer activation to the next PP stage.',`The next stage broadcasts this vector to its ${tpWord()} TP ranks.`],
    'head.gather':['Copy one vocabulary shard to the master chiplet.',`Keep its vocabulary columns intact. A later concatenate operation joins the ${tpWord()} disjoint shards.`],
    'head.host':['Copy the complete vocabulary logits to the host.','Host sampling is outside this trace.']
  };
  if(transfers[key]){
    const axis=key==='head.gather'?['batch','tokens','local vocabulary columns']:key==='head.host'?['batch','tokens','all vocabulary columns']:hidden;
    return guide(transfers[key][0],[transfers[key][1],'The selected message preserves tensor values and shape; its route is shown above.'],
      inputs.map(()=>'Source tensor'),outputs.map(()=>'Destination tensor'),inputs.map(()=>axis),outputs.map(()=>axis),`${dims(inputs[0])} → copy → ${dims(outputs[0])}`);
  }
  throw new Error('Missing tensor walkthrough: '+key);
}

// Mathematical semantics supplement the tensor walkthrough in optional details.
// All expressions describe the saved operator, including local-only transfers
// and fixed routing assignments, rather than an unexecuted learned top-k choice.
function operatorFormula(op,model,block,config){
  const key=op.key,h=model.head_dim,d=model.hidden_size,g=model.num_attention_heads/model.num_key_value_heads;
  const result=(lines,symbols)=>({lines,symbols});
  const norm=key.match(/^(norm[12]|head\.norm)\.(square|mean|scale|output)$/);
  if(norm){
    const expressions={square:['aⱼ = xⱼ²'],mean:[`μ = (Σⱼ aⱼ) / ${d}`],
      scale:['r = 1 / √(μ + ε)'],output:['yⱼ = xⱼ · r · γⱼ']};
    return result(expressions[norm[2]],`j indexes the ${d} hidden features; a is the already squared input to mean; γ is the normalization weight; ε = ${config.rms_norm_epsilon}. Each TP rank holds the full input.`);
  }
  for(const name of ['Q','K','V']){
    if(key===name+'.raw')return result([`${name}_raw = normalized_input × local_W${name}`],'The input is replicated across TP ranks; the local weight owns this rank’s output-head columns.');
    if(key===name)return result([`${name} = ${name}_raw + local_bias`],'The bias is broadcast over tokens and has the same local head slice as the projection.');
    if(key===name+'_ROPE')return result([
      'y1 = c × (x1 × cos(angle) − x2 × sin(angle))',
      'y2 = c × (x2 × cos(angle) + x1 × sin(angle))',
      'angle = position × frequency[j]',
      'blend[j] = clip((j − low)/(high − low), 0, 1)',
      'frequency[j] = (1 − blend[j] + blend[j]/scale) / theta^(2j/f)',
      'low = (f/2) × ln(initial_context/(2π × beta_fast)) / ln(theta)',
      'high = (f/2) × ln(initial_context/(2π × beta_slow)) / ln(theta)'
    ],`x1 = x[j] and x2 = x[j + f/2] are a pair in one ${name} head; y1/y2 are the rotated pair. f = ${h}; j = 0…${h/2-1}; position = ${config.context_tokens} is absolute, including window attention. theta = ${config.rope_theta}; scale = ${model.rope_scaling_factor}; initial_context = ${model.initial_context_length}; beta_fast = ${model.rope_beta_fast}; beta_slow = ${model.rope_beta_slow}; c = 1 + 0.1 ln(scale) when scale > 1, otherwise 1.`);
  }
  if(key==='K_CACHE'||key==='V_CACHE')return result([`${key[0]}cache[b, k, n−1, :] ← ${key[0]}new[b, k, :]`],`b is the token's batch index; k is its local KV head; n = ${block.attention_tokens}. Other cache rows remain resident. This assignment reads and writes local HBM.`);
  const attentionSymbols=`For one query head: query is its ${h}-feature vector; key[t] and value[t] belong to its matching GQA KV head; ${g} query heads share one KV head. t indexes ${block.attention_tokens} visible tokens; f = ${h}; sink is the learned sink logit, with zero value and no division by √f. Sum/max over t visit all visible tokens.`;
  const attention={scores:['score[t] = dot(query, key[t])'],scaled:['z[t] = score[t] / √f'],
    maximum:['m = max(sink, max over t of z[t])'],exp:['p[t] = exp(z[t] − m)'],
    denominator:['denom = exp(sink − m) + sum over t of p[t]'],
    av_raw:['context_sum = sum over t of p[t] × value[t]'],av:['context = context_sum / denom']};
  if(attention[key])return result(attention[key],attentionSymbols);
  if(key==='flash_attention')return result([
    'Equivalent result: z[t] = dot(query, key[t]) / √f',
    'weight[t] = exp(z[t]) / (exp(sink) + sum over j of exp(z[j]))',
    'context = sum over t of weight[t] × value[t]',
    'Stable block algorithm: initialize m = sink; denom = 1; context_sum = 0',
    'For each block: next_m = max(m, max of z in block); a = exp(m − next_m)',
    'p[t] = exp(z[t] − next_m) for tokens t in this block',
    'next_denom = a × denom + sum of p[t] in block',
    'next_context_sum = a × context_sum + sum of p[t] × value[t] in block',
    'Replace m, denom and context_sum with their next values; final context = context_sum / denom'
  ],attentionSymbols+` The first three lines express the result; the implementation uses the stable block algorithm below them. Process at most ${config.flash_attention_block_tokens} KV tokens per block, including the final tail. m is the running maximum, denom the running denominator, and context_sum a ${h}-feature weighted sum. They persist in scratch; the sink is initialized exactly once. No full HBM score/probability tensors are created.`);
  if(key==='OP_PARTIAL')return result(['partial = local_context × local_WO'],`local_context holds the local query heads; local_WO owns matching input rows. partial has the full hidden width. A separate TP reduction adds the ${tpWord()} partials.`);
  const tree=key.match(/^(attention_reduce|ffn_reduce)\.(up|tree_sum|down)$/);
  if(tree)return result(tree[2]==='up'?['parent_buffer ← child_partial']:tree[2]==='tree_sum'?['partial_parent ← partial_parent + received_child_partial']:['X_r ← root_sum  for every rank r'],
    `The tree is rooted at the group's most central chiplet. Each parent adds its children's partials (up), so the root holds Σ over all ${tpRanks()} ranks; the root then broadcasts that sum down the same tree. Sends are physical SEND/receive; additions are local SIMD.`);
  const reduction=key.match(/^(attention_reduce|ffn_reduce)\.r([01])\.(exchange|sum)$/);
  if(reduction){
    const round=+reduction[2]+1;
    return result(reduction[3]==='exchange'?['received_partial ← partner_partial']:['result = local_partial + received_partial'],`This is round ${round}. Round 1 pairs ranks (0,1) and (2,3); round 2 pairs (0,2) and (1,3). After round 2, result includes all four original partials. Exchange is a physical SEND/receive; addition is local SIMD.`);
  }
  if(key==='O_BIASED')return result(['Ybiased = Y + bO'],'Y is the complete TP-reduced projection. Add the bias once to each complete replica.');
  if(key==='RES1')return result(['Xattn = Xlayer + Ybiased'],'Both operands are full hidden-width replicas on the same chiplet.');
  if(key==='router_logits.raw')return result(['L⁰ = Xnorm Wrouter'],'Wrouter has one output column for every expert and is replicated across TP ranks.');
  if(key==='router_logits')return result(['L = L⁰ + brouter'],'L contains every expert logit for each token.');
  if(key==='routing_weights')return result([
    'selected = saved expert assignment for this token',
    'weight[e] = exp(logit[e]) / sum of exp(logit[j]) over selected experts j',
    'weight[e] = 0 for unselected experts'
  ],`selected contains ${model.num_experts_per_tok} expert IDs. This performance run uses ${config.expert_routes?'provided':'synthetic balanced'} assignments; it does not predict learned TopK. Softmax normalizes only the selected logits.`);
  const expert={
    dispatch:[['expert_input = gather(normalized_input, assigned_token_rows)'],'Gather copies token rows assigned to this expert, locally on each TP rank.'],
    'up.raw':[['up_raw = expert_input × local_up_weight'],`Each TP rank owns a contiguous 1/${tpRanks()} share of the intermediate output columns.`],
    up:[['up = up_raw + local_up_bias'],'Bias follows the same intermediate-column slice.'],
    'gate.raw':[['gate_raw = expert_input × local_gate_weight'],'Gate and up own matching intermediate columns.'],
    gate:[['gate = gate_raw + local_gate_bias'],'Bias follows the same intermediate-column slice.'],
    glu:[['g = min(gate, limit); u = clip(up, −limit, limit)',
          'result = g × sigmoid(alpha × g) × (u + 1)', 'sigmoid(z) = 1 / (1 + exp(−z))'],`limit = ${model.swiglu_limit}; alpha = ${model.swiglu_alpha}. Every multiplication is elementwise. Gate is capped only above; up is capped on both sides.`],
    down:[['partial = local_intermediate × local_down_weight'],'The weight owns the matching intermediate input rows. Each result is a full hidden-width partial.'],
    down_biased:[['biased_partial = partial + expert_down_bias (only on TP rank 0)'],'Other ranks retain their unbiased partial. The TP sum includes the down bias exactly once.'],
    weighted:[['weighted_partial = expert_partial × routing_weight_for_this_expert','batch_result = scatter(weighted_partial, original_token_rows)'],'The routing weight is one scalar per assigned token, shared by its features and TP ranks. Unassigned batch slots are zero.'],
    accum:[['next_sum = previous_sum + next_weighted_expert_partial'],`Repeat for the active experts on this rank. The separate MoE TP reduction adds all ${tpWord()} rank sums.`]
  };
  if(key.startsWith('expert.')&&expert[key.slice(7)])return result(...expert[key.slice(7)]);
  if(key==='RES2')return result(['Xout = Xattn + M'],'M is the complete TP-reduced weighted MoE result. Xout feeds the next layer or PP relay.');
  const boundary={
    'input.host':[['Xmaster ← Xhost'],'The host provides the activation; embedding computation is outside the trace.'],
    'input.broadcast':[[`Xr ← Xentry  for r = 0…${tpRanks()-1}`],'Broadcast makes identical TP input copies along the recorded physical routes.'],
    'stage.broadcast':[[`Xr ← Xentry  for r = 0…${tpRanks()-1}`],'The next stage broadcasts its received PP activation.'],
    'stage.relay':[['Xstage+1,entry ← Xstage,last'],'Relay copies the completed activation to the next PP stage.'],
    'head.projection':[['logitsr = Xfinal_norm WLM,r'],'Each rank owns one vocabulary-column shard; this is not a partial sum.'],
    'head.gather':[['master_buffer[r] ← logitsr'],'Gather preserves each vocabulary shard; it does not add shard values.'],
    'head.concatenate':[[`logits = concat(logits0, …, logits${tpRanks()-1})`],'Concatenate vocabulary columns in TP rank order.'],
    'head.host':[['logitshost ← logitsmaster'],'Return the complete logits; host sampling is outside the trace.']
  };
  if(boundary[key])return result(...boundary[key]);
  throw new Error('Missing operator formula: '+key);
}

function rankLayout(description, rank, model){
  const share=CASCADE_DATA.rank_shares[rank],g=model.num_attention_heads/model.num_key_value_heads;
  const span=(name,range,scale=1)=>range[1]-range[0]===1&&scale===1?`${name} ${range[0]}`:`${name}s ${range[0]*scale}–${range[1]*scale-1}`;
  const noHeads='No KV head · relays TP sums';
  if(description.mode==='q')return share.kv_heads?span('Q head',share.kv_heads,g):noHeads;
  if(description.mode==='kv')return share.kv_heads?span('KV head',share.kv_heads):noHeads;
  if(description.mode==='expert_width')return `Expert columns ${share.expert_columns[0]}–${share.expert_columns[1]-1}`;
  if(description.mode==='vocab')return `Vocabulary ${share.vocabulary[0]}–${share.vocabulary[1]-1}`;
  if(description.mode==='partial')return `${model.hidden_size}-feature partial sum`;
  if(description.mode==='pair_sum')return 'Sum of two TP ranks';
  if(description.mode==='tree_sum')return 'Adds received child partials';
  if(description.mode==='rank_zero')return rank===0?'Down bias added once':'No down-bias command';
  return 'Replicated input / local output';
}

function tensorPartitionSummary(op, model, block){
  const d=model.hidden_size,q=model.num_attention_heads,h=model.head_dim,kv=model.num_key_value_heads,s=model.intermediate_size;
  const T=tpRanks(),g=q/kv,idle=idleRanks()?` ${idleRanks()} ranks own none.`:'';
  if(['Q.raw','Q','Q_ROPE','K.raw','K','K_ROPE','V.raw','V'].includes(op.key)){
    const heads=op.key.startsWith('Q')?q:kv,scale=op.key.startsWith('Q')?g:1;
    return `${heads} global heads × ${h} features → ${T} ranks; an owning rank holds ${countText('kv_heads',scale)} complete heads (${countText('kv_heads',scale*h)} projection columns).${idle}`;
  }
  if(op.key==='K_CACHE'||op.key==='V_CACHE')return `Global cache [1 × ${kv} × ${block.attention_tokens} × ${h}] → local caches of ${countText('kv_heads')} KV head(s) [1 × ${countText('kv_heads')} × ${block.attention_tokens} × ${h}].${idle} The new token is appended only to the owning head slice.`;
  if(op.key==='OP_PARTIAL')return `Global WO [${q*h} × ${d}] → row shards of ${countText('kv_heads',g*h)} rows on the owning ranks.${idle} Each owner produces ${d} output features as a partial sum; TP reduction combines them.`;
  if(['expert.up.raw','expert.up','expert.gate.raw','expert.gate','expert.glu'].includes(op.key))return `Each expert has ${s} intermediate features → ${T} ranks × ${countText('expert_columns')} features. Up and gate weights split their output columns; SwiGLU keeps the same partition.`;
  if(op.key==='expert.down')return `Each expert’s Wdown [${s} × ${d}] → ${T} row shards [${countText('expert_columns')} × ${d}]. Outputs are full-width partial sums, rather than disjoint output feature slices.`;
  if(op.key==='head.projection')return `Global LM weights [${d} × ${model.vocab_size}] → ${T} column shards [${d} × ${countText('vocabulary')}]. Vocabulary outputs are concatenated, not reduced.`;
  if(['scores','scaled','maximum','exp','denominator','av_raw','av','flash_attention'].includes(op.key))return `${q} query heads and ${kv} KV heads → ${countText('kv_heads',g)} Q + ${countText('kv_heads')} KV heads per owning rank.${idle} Complete heads stay local across score calculation, softmax and value aggregation.`;
  if(op.key.startsWith('router_')||op.key==='routing_weights')return `The full router [${d} × ${model.num_local_experts}] is replicated across TP ranks. Every active expert then uses the same ${T}-rank intermediate partition.`;
  return '';
}
