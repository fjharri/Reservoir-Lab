'use strict';
const $=id=>document.getElementById(id);
const canvas=$('pendulumCanvas'),chartCanvas=$('errorChart');
let worker=null,timer=null,noticeTimer=null,running=false;
let state,physics,config,sampleId,totalSteps,changeStep,changed,latestForecast;
let predictions=new Map(),errors=[],recentErrors=[],errorAverage=null,errorTotal=0,errorCount=0;
let updateAverage=null,latencyAverage=null,deadlineMisses=0,responseCount=0,recoveryStreak=0,recoverySeconds=null,baselineError=null,trail=[];

const scenarios={
  load:{description:'Second mass increased × 2.2',apply:p=>{p.m2=2.2;}},
  length:{description:'Second arm shortened to 62%',apply:p=>{p.l2=.62;}},
  damping:{description:'Joint friction increased × 8',apply:p=>{p.damping=.12;}},
  combined:{description:'Mass, arm length and friction changed',apply:p=>{p.m2=1.8;p.l2=.72;p.damping=.07;}}
};

function readConfig(){
  const rate=+$('sampleRate').value,horizonMs=+$('forecastHorizon').value;
  return {rate,horizonMs,horizon:Math.max(1,Math.round(rate*horizonMs/1000)),duration:+$('duration').value,size:+$('reservoirSize').value,seed:+$('pendulumSeed').value,scenario:$('physicsChange').value,interval:1000/rate};
}
function setStatus(text,busy=false){$('pendulumStatus').textContent=text;$('pendulumStatusDot').classList.toggle('busy',busy);}
function setRunning(active){running=active;$('pendulumFields').disabled=active;$('startPendulum').disabled=active;$('stopPendulum').hidden=!active;}
function resetMetrics(){
  predictions=new Map();errors=[];recentErrors=[];errorAverage=null;errorTotal=0;errorCount=0;updateAverage=null;latencyAverage=null;deadlineMisses=0;responseCount=0;recoveryStreak=0;recoverySeconds=null;baselineError=null;trail=[];latestForecast=null;
  $('forecastError').textContent='—';$('updateTime').textContent='—';$('deadlinesMissed').textContent='—';$('recoveryTime').textContent='—';$('recoveryNote').textContent='after physics change';$('pendulumClock').textContent='0.0 s';$('pendulumProgress').style.width='0';$('changeNotice').hidden=true;
  $('runFinding').textContent='The physics will change 40% of the way through the run. Watch the error rise, then fall as the readout relearns online.';
  drawChart();
}
function begin(){
  state=[1.72,0,1.08,0];physics={m1:1,m2:1,l1:1,l2:1,g:9.81,damping:.015};sampleId=0;totalSteps=config.duration*config.rate;changeStep=Math.round(totalSteps*.4);changed=false;
  $('deadlineText').textContent=`${config.interval.toFixed(1)} ms deadline`;
  setStatus('Learning the initial dynamics…',true);setRunning(true);sendSample();drawPendulum();
  timer=setInterval(step,config.interval);
}
function step(){
  if(!running)return;
  sampleId++;
  if(sampleId===changeStep) applyPhysicsChange();
  state=PendulumCore.stepPhysics(state,physics,1/config.rate);
  const actual=PendulumCore.observation(state),prediction=predictions.get(sampleId);
  if(prediction){recordError(PendulumCore.angularErrorDegrees(prediction,actual));predictions.delete(sampleId);}
  trail.push(position(state[0],state[2]));if(trail.length>90)trail.shift();
  sendSample(actual);drawPendulum();
  const elapsed=sampleId/config.rate;$('pendulumClock').textContent=elapsed.toFixed(1)+' s';$('pendulumProgress').style.width=Math.min(100,sampleId/totalSteps*100)+'%';
  if(sampleId>=totalSteps) finish();
}
function sendSample(value=PendulumCore.observation(state)){
  worker?.postMessage({type:'sample',id:sampleId,observation:value,sentAt:performance.now()});
}
function applyPhysicsChange(){
  changed=true;baselineError=recentErrors.length?recentErrors.reduce((a,b)=>a+b,0)/recentErrors.length:null;
  const scenario=scenarios[config.scenario];scenario.apply(physics);$('changeDescription').textContent=scenario.description;$('changeNotice').hidden=false;
  clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>{$('changeNotice').hidden=true;},2800);setStatus('Physics changed — adapting online…',true);
}
function recordError(value){
  errorAverage=errorAverage===null?value:errorAverage*.88+value*.12;errorTotal+=value;errorCount++;
  $('forecastError').textContent=errorAverage.toFixed(1)+'°';
  const time=sampleId/config.rate;errors.push({time,value:errorAverage});
  if(!changed){recentErrors.push(value);if(recentErrors.length>config.rate*3)recentErrors.shift();}
  else if(recoverySeconds===null&&baselineError!==null){
    const threshold=Math.max(6,baselineError*1.5);
    recoveryStreak=errorAverage<=threshold?recoveryStreak+1:0;
    if(recoveryStreak>=config.rate){recoverySeconds=(sampleId-changeStep)/config.rate;$('recoveryTime').textContent=recoverySeconds.toFixed(1)+' s';$('recoveryNote').textContent='to sustain pre-change range';setStatus('Adapted — continuing to learn',true);}
  }
  if(errors.length%3===0)drawChart();
}
function handleWorker(message){
  if(message.type==='ready'){$('runFinding').textContent=`Reservoir ready: ${config.size} fixed neurons, ${message.featureSize} readout features, ${(message.memoryBytes/1048576).toFixed(2)} MB model state.`;begin();return;}
  if(message.type==='error'){stopExperiment('Experiment failed: '+message.message);return;}
  if(message.type!=='prediction'||!running)return;
  const latency=performance.now()-message.sentAt;responseCount++;if(latency>config.interval)deadlineMisses++;
  updateAverage=updateAverage===null?message.computeMs:updateAverage*.9+message.computeMs*.1;latencyAverage=latencyAverage===null?latency:latencyAverage*.9+latency*.1;
  $('updateTime').textContent=updateAverage<10?updateAverage.toFixed(2)+' ms':updateAverage.toFixed(1)+' ms';$('deadlinesMissed').textContent=deadlineMisses.toLocaleString();
  if(message.targetId>sampleId) predictions.set(message.targetId,message.prediction);latestForecast=message.prediction;
}
function finish(){
  clearInterval(timer);timer=null;worker?.terminate();worker=null;setRunning(false);setStatus('Experiment complete');$('pendulumProgress').style.width='100%';
  const mean=errorCount?errorTotal/errorCount:NaN,missRate=responseCount?deadlineMisses/responseCount*100:0;
  if(recoverySeconds===null){$('recoveryTime').textContent='Not yet';$('recoveryNote').textContent='within this run';}
  $('runFinding').textContent=`Across ${errorCount.toLocaleString()} scored forecasts, mean angular error was ${Number.isFinite(mean)?mean.toFixed(1)+'°':'unavailable'}. The worker missed ${deadlineMisses} of ${responseCount} sample deadlines (${missRate.toFixed(1)}%)${recoverySeconds===null?'; the error did not return to its pre-change range before the run ended':` and returned to its pre-change range in ${recoverySeconds.toFixed(1)} seconds`}.`;
  drawChart();
}
function stopExperiment(message='Stopped. Start again for a fresh stream.'){
  clearInterval(timer);timer=null;clearTimeout(noticeTimer);worker?.terminate();worker=null;setRunning(false);setStatus(message);$('changeNotice').hidden=true;
}

function sizeCanvas(element){
  const rect=element.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(rect.width*ratio)),h=Math.max(1,Math.round(rect.height*ratio));
  if(element.width!==w||element.height!==h){element.width=w;element.height=h;}const context=element.getContext('2d');context.setTransform(ratio,0,0,ratio,0,0);return {context,width:rect.width,height:rect.height};
}
function position(a,b){return {a,b};}
function points(a,b,width,height){
  const scale=Math.min(width/5,height/3.2),ox=width/2,oy=Math.max(55,height*.15),x1=ox+Math.sin(a)*physics.l1*scale,y1=oy+Math.cos(a)*physics.l1*scale;
  return {ox,oy,x1,y1,x2:x1+Math.sin(b)*physics.l2*scale,y2:y1+Math.cos(b)*physics.l2*scale};
}
function drawSystem(context,p,color,dashed=false,alpha=1){
  context.save();context.globalAlpha=alpha;context.strokeStyle=color;context.fillStyle=color;context.lineWidth=dashed?2:3;if(dashed)context.setLineDash([7,6]);
  context.beginPath();context.moveTo(p.ox,p.oy);context.lineTo(p.x1,p.y1);context.lineTo(p.x2,p.y2);context.stroke();context.setLineDash([]);
  for(const [x,y,r] of [[p.ox,p.oy,4],[p.x1,p.y1,8],[p.x2,p.y2,10]]){context.beginPath();context.arc(x,y,r,0,Math.PI*2);context.fill();}context.restore();
}
function drawPendulum(){
  const {context,width,height}=sizeCanvas(canvas);context.clearRect(0,0,width,height);
  if(trail.length>1){context.save();context.strokeStyle='#107c6724';context.lineWidth=1;context.beginPath();trail.forEach((item,index)=>{const p=points(item.a,item.b,width,height);if(index)context.lineTo(p.x2,p.y2);else context.moveTo(p.x2,p.y2);});context.stroke();context.restore();}
  if(latestForecast){const [a,b]=PendulumCore.anglesFromObservation(latestForecast);drawSystem(context,points(a,b,width,height),'#107c67',true,.55);}
  drawSystem(context,points(state?.[0]||1.72,state?.[2]||1.08,width,height),'#0d242b');
  context.fillStyle='#586c71';context.font='10px ui-monospace, monospace';context.fillText(`${config?.horizonMs||250} ms ahead`,14,22);
}
function drawChart(){
  const {context,width,height}=sizeCanvas(chartCanvas),left=34,right=10,top=10,bottom=25,maxTime=config?.duration||30,maxError=Math.max(30,...errors.map(point=>point.value))*1.1;
  context.clearRect(0,0,width,height);context.font='9px ui-monospace, monospace';context.fillStyle='#586c71';context.strokeStyle='#dfe7e2';context.lineWidth=1;
  for(let i=0;i<=3;i++){const y=top+(height-top-bottom)*i/3,value=maxError*(1-i/3);context.beginPath();context.moveTo(left,y);context.lineTo(width-right,y);context.stroke();context.fillText(Math.round(value)+'°',2,y+3);}
  const changeX=left+(width-left-right)*.4;context.strokeStyle='#bd513c';context.beginPath();context.moveTo(changeX,top);context.lineTo(changeX,height-bottom);context.stroke();
  if(errors.length){context.strokeStyle='#107c67';context.lineWidth=2;context.beginPath();errors.forEach((point,index)=>{const x=left+point.time/maxTime*(width-left-right),y=top+(1-point.value/maxError)*(height-top-bottom);if(index)context.lineTo(x,y);else context.moveTo(x,y);});context.stroke();}
  context.fillStyle='#586c71';for(let i=0;i<=3;i++)context.fillText((maxTime*i/3).toFixed(0)+'s',left+(width-left-right)*i/3-7,height-5);
}

$('pendulumSetup').addEventListener('submit',event=>{
  event.preventDefault();stopExperiment('Preparing the reservoir…');resetMetrics();config=readConfig();setRunning(true);setStatus('Preparing the fixed reservoir…',true);
  worker=new Worker('/examples/double-pendulum/esn-worker.js?v=0.4.0');worker.onmessage=({data})=>handleWorker(data);worker.onerror=event=>stopExperiment('Could not start the learning worker: '+event.message);
  worker.postMessage({type:'init',size:config.size,seed:config.seed,horizon:config.horizon});
  if(innerWidth<=650)document.querySelector('.pendulum-results').scrollIntoView({behavior:'smooth',block:'start'});
});
$('stopPendulum').addEventListener('click',()=>stopExperiment());
addEventListener('resize',()=>{drawPendulum();drawChart();});
config=readConfig();state=[1.72,0,1.08,0];physics={m1:1,m2:1,l1:1,l2:1,g:9.81,damping:.015};drawPendulum();drawChart();
