'use strict';
const $=id=>document.getElementById(id), pct=x=>Number.isFinite(x)?(100*x).toFixed(1)+'%':'—';
const time=x=>Number.isFinite(x)?(x<1000?Math.round(x)+' ms':(x/1000).toFixed(2)+' s'):'—';
let worker=null, result=null, esn=null, points=[], exportUrls=[];
function config() {return {reservoirSize:+$('reservoirSize').value,gruUnits:+$('gruUnits').value,learningRate:+$('learningRate').value,ridge:+$('ridge').value,maxSeconds:+$('maxSeconds').value,seed:+$('seed').value,tolerance:+$('tolerance').value,batchSize:8,checkpointEvery:8,earlyStoppingPatience:6,maxUpdates:2000};}
function status(message,busy=false) {$('status').textContent=message;$('statusDot').classList.toggle('busy',busy);}
function running(active) {$('fields').disabled=active;$('run').disabled=active;$('stop').hidden=!active;}
function score(id, accuracy) {$(id).replaceChildren(document.createTextNode(pct(accuracy)));const small=document.createElement('small');small.textContent='test accuracy';$(id).append(small);}
function chart() {
  if(!esn&&!points.length) return;
  const w=Math.max(280,$('chart').clientWidth),h=$('chart').clientHeight||280,l=42,r=18,t=18,b=42, max=Math.max(1000,esn?.trainingMs||0,...points.map(p=>p.trainingMs))*1.06;
  const x=ms=>l+ms/max*(w-l-r), y=a=>t+(1-a)*(h-t-b);
  let svg=`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><g font-family="Arial,sans-serif" font-size="10" fill="#586c71">`;
  for(const a of [0,.25,.5,.75,1]) svg+=`<line x1="${l}" y1="${y(a)}" x2="${w-r}" y2="${y(a)}" stroke="#e3eae5" ${a===.25?'stroke-dasharray="3 4"':''}/><text x="${l-9}" y="${y(a)+4}" text-anchor="end">${a*100}%</text>`;
  for(let i=0;i<=4;i++) svg+=`<text x="${x(max*i/4)}" y="${h-23}" text-anchor="middle">${(max*i/4000).toFixed(max<5000?2:1)}</text>`;
  svg+=`<text x="${w/2}" y="${h-4}" text-anchor="middle">Active training time (seconds)</text></g>`;
  if(esn) {
    svg+=`<line x1="${x(esn.trainingMs)}" y1="${t}" x2="${x(esn.trainingMs)}" y2="${h-b}" stroke="#107c67" stroke-dasharray="4 5" opacity=".5"/><line x1="${l}" y1="${y(esn.validation.accuracy)}" x2="${w-r}" y2="${y(esn.validation.accuracy)}" stroke="#107c67" opacity=".25"/><circle cx="${x(esn.trainingMs)}" cy="${y(esn.validation.accuracy)}" r="5" fill="#107c67"><title>ESN: ${pct(esn.validation.accuracy)} at ${time(esn.trainingMs)}</title></circle>`;
  }
  if(points.length) {
    svg+=`<path d="${points.map((p,i)=>(i?'L':'M')+x(p.trainingMs)+','+y(p.accuracy)).join(' ')}" fill="none" stroke="#bd513c" stroke-width="2.5"/>`;
    for(const p of points) svg+=`<circle cx="${x(p.trainingMs)}" cy="${y(p.accuracy)}" r="3" fill="#bd513c"><title>GRU: ${pct(p.accuracy)} at ${time(p.trainingMs)} · ${p.updates} updates</title></circle>`;
  }
  $('chart').innerHTML=svg+'</svg>';
}
function renderMatrix() {
  if(!result) return;
  const kind=$('matrixModel').value;
  const modelNames={esn:'ESN',equal:'GRU · same budget',match:'GRU · first match',final:'GRU · finalised'};
  const metric=kind==='esn'?result.esn.test:kind==='equal'?result.gru.equalBudget?.test:kind==='match'?result.gru.matched?.test:result.gru.finalized?.test||result.gru.finalTest;
  if(!metric) {$('confusion').textContent=kind==='equal'?'No complete GRU checkpoint fitted within the ESN budget.':'The GRU did not reach the validation target in this run.';return;}
  const names=['100%','90%','80%','73%'];
  $('confusion').innerHTML='<table><caption class="hint">'+modelNames[kind]+' · '+pct(metric.accuracy)+' accuracy · '+metric.count+' test cycles</caption><thead><tr><th>Actual ↓ / predicted →</th>'+names.map(n=>'<th>'+n+'</th>').join('')+'</tr></thead><tbody>'+metric.confusion.map((row,i)=>'<tr><th>'+names[i]+'</th>'+row.map((v,j)=>'<td class="'+(i===j?'diag':'')+'">'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
}
function finish(r) {
  result=r;esn=r.esn;points=r.curve;chart();
  score('esnScore',r.esn.test.accuracy);$('esnTime').textContent=time(r.esn.trainingMs);$('esnVal').textContent=pct(r.esn.validation.accuracy);$('esnParams').textContent=r.esn.trainableParameters.toLocaleString();
  const equal=r.gru.equalBudget;score('gruScore',equal?.test?.accuracy);$('gruTime').textContent=time(equal?.trainingMs);$('gruVal').textContent=pct(equal?.validation?.accuracy);$('gruUpdates').textContent=equal?equal.updates:'No checkpoint';
  const matched=r.gru.matched;
  const finalized=r.gru.finalized||{trainingMs:r.gru.trainingMs,updates:r.gru.updates,validation:r.gru.finalValidation,test:r.gru.finalTest};
  score('matchScore',matched?.test?.accuracy);$('matchTime').textContent=time(matched?.trainingMs);$('matchVal').textContent=pct(matched?.validation?.accuracy);$('matchUpdates').textContent=matched?matched.updates:'Not reached';
  score('finalScore',finalized?.test?.accuracy);$('finalTime').textContent=time(r.gru.trainingMs);$('finalVal').textContent=pct(finalized?.validation?.accuracy);$('finalUpdates').textContent=finalized?.updates??'—';
  if(matched) $('match').textContent=`GRU first reached the ${pct(r.targetAccuracy)} validation target at ${time(matched.trainingMs)} (${matched.updates} updates). Training continued to ${time(r.gru.trainingMs)} and retained update ${finalized.updates}, which had the best observed validation accuracy of ${pct(finalized.validation.accuracy)}.`;
  else $('match').textContent=`GRU did not reach the ${pct(r.targetAccuracy)} validation target within the ${r.config.maxSeconds}-second cap. The finalised result uses its best observed validation checkpoint: update ${finalized.updates} at ${pct(finalized.validation.accuracy)} validation accuracy.`;
  $('budgetNote').textContent=(equal?`GRU kept ${equal.updates} complete updates inside the ${time(r.esn.trainingMs)} budget. `:'GRU setup alone exceeded the ESN budget, so no equal-budget result is available. ')+`${r.gru.reason==='early-stopping'?`Training stopped after ${r.gru.earlyStopping.patienceChecks} validation checks without improvement. `:'Training ended at the configured cap. '}Total experiment elapsed time: ${time(r.wallMs)}; evaluation and logging are excluded from the training clocks. GRU fitted parameters: ${r.gru.trainableParameters.toLocaleString()}.`;
  const filename='reservoir-lab-'+r.createdAt.replace(/[:.]/g,'-');
  setExport('downloadJson',JSON.stringify(r,null,2),filename+'.json','application/json');
  setExport('downloadCsv','training_ms,updates,epoch,validation_accuracy,loss\n'+r.curve.map(p=>[p.trainingMs,p.updates,p.epoch,p.accuracy,p.loss??''].join(',')).join('\n')+'\n',filename+'-curve.csv','text/csv');
  $('resultsDetail').hidden=false;$('progressFill').style.width='100%';$('progressLabel').textContent='Complete';status('Experiment complete');running(false);renderMatrix();worker.terminate();worker=null;
}
$('setup').addEventListener('submit',event=> {
  event.preventDefault();result=null;esn=null;points=[];exportUrls.forEach(url=>URL.revokeObjectURL(url));exportUrls=[];$('resultsDetail').hidden=true;
  for(const id of ['esnTime','esnVal','esnParams','gruTime','gruVal','gruUpdates','matchTime','matchVal','matchUpdates','finalTime','finalVal','finalUpdates']) $(id).textContent='—';
  for(const id of ['esnScore','gruScore','matchScore','finalScore']) score(id,null);
  $('chart').innerHTML='<div class="chart-empty"><p>Loading the experiment…</p><span>All training runs on your device.</span></div>';
  $('match').textContent='The ESN’s full training time becomes the GRU’s comparison budget.';$('progressFill').style.width='0';$('progressLabel').textContent='';
  running(true);status('Loading sensor data and the training engine…',true);
  if(innerWidth<=650) document.querySelector('.results').scrollIntoView({behavior:'instant',block:'start'});
  worker=new Worker('/examples/hydraulic/worker.js?v=0.3.0');
  worker.onmessage=({data:event})=> {
    if(event.type==='status') status(event.message,true);
    if(event.type==='esn') {esn=event.esn;$('esnTime').textContent=time(esn.trainingMs);$('esnVal').textContent=pct(esn.validation.accuracy);$('esnParams').textContent=esn.trainableParameters.toLocaleString();chart();status('GRU: learning through backpropagation…',true);}
    if(event.type==='point') {points.push(event.point);chart();}
    if(event.type==='progress') {$('progressFill').style.width=Math.min(100,event.trainingMs/event.maxMs*100)+'%';$('progressLabel').textContent=time(event.trainingMs)+' · '+event.updates+' updates';}
    if(event.type==='done') finish(event.result);
    if(event.type==='error') {status('Experiment failed: '+event.message);running(false);worker.terminate();worker=null;}
  };
  worker.onerror=event=>{status('Could not start the training worker: '+event.message);running(false);worker?.terminate();worker=null;};
  worker.postMessage({config:config()});
});
$('stop').addEventListener('click',()=>{worker?.terminate();worker=null;running(false);status('Stopped. Run again to start a fresh experiment.');$('progressLabel').textContent='Stopped';$('match').textContent='This run was stopped before final evaluation. No result has been saved.';});
$('matrixModel').addEventListener('change',renderMatrix);
function setExport(id,content,name,type) {const url=URL.createObjectURL(new Blob([content],{type}));exportUrls.push(url);$(id).href=url;$(id).download=name;}
addEventListener('resize',()=>{if(esn||points.length) requestAnimationFrame(chart);});
