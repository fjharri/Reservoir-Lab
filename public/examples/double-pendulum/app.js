'use strict';
const $=id=>document.getElementById(id);
const canvas=$('pendulumCanvas'),chartCanvas=$('errorChart');
let worker=null,timer=null,noticeTimer=null,finishTimer=null,running=false,settling=false;
let state,physics,config,sampleId,totalSteps,changeStep,changed,latestAdaptive,latestFrozen;
let predictions=new Map(),adaptiveErrors=[],frozenErrors=[],recentErrors=[],adaptiveAverage=null,frozenAverage=null,adaptivePostTotal=0,frozenPostTotal=0,postCount=0;
let updateAverage=null,deadlineMisses=0,responseCount=0,recoveryStreak=0,recoverySeconds=null,baselineError=null,trail=[];
let scheduleStartedAt=0,scheduleEndedAt=0,sentSamples=0,droppedSamples=0,lateTicks=0,inFlight=0;
const MAX_CATCH_UP=4;

const scenarios={
  load:{description:'Second mass increased × 2.2',apply:p=>{p.m2=2.2;}},
  length:{description:'Second arm shortened to 62%',apply:p=>{p.l2=.62;}},
  damping:{description:'Joint friction increased × 8',apply:p=>{p.damping=.12;}},
  combined:{description:'Mass × 2.6, arm shortened, friction increased',apply:p=>{p.m2=2.6;p.l2=.55;p.damping=.12;}}
};

function readConfig(){
  const rate=+$('sampleRate').value,horizonMs=+$('forecastHorizon').value;
  return {rate,horizonMs,horizon:Math.max(1,Math.round(rate*horizonMs/1000)),duration:+$('duration').value,size:+$('reservoirSize').value,seed:+$('pendulumSeed').value,scenario:$('physicsChange').value,interval:1000/rate};
}
function setStatus(text,busy=false){$('pendulumStatus').textContent=text;$('pendulumStatusDot').classList.toggle('busy',busy);}
function setRunning(active){running=active;$('pendulumFields').disabled=active;$('startPendulum').disabled=active;$('stopPendulum').hidden=!active;}
function resetMetrics(){
  predictions=new Map();adaptiveErrors=[];frozenErrors=[];recentErrors=[];adaptiveAverage=null;frozenAverage=null;adaptivePostTotal=0;frozenPostTotal=0;postCount=0;updateAverage=null;deadlineMisses=0;responseCount=0;recoveryStreak=0;recoverySeconds=null;baselineError=null;trail=[];latestAdaptive=null;latestFrozen=null;
  scheduleStartedAt=0;scheduleEndedAt=0;sentSamples=0;droppedSamples=0;lateTicks=0;inFlight=0;settling=false;
  $('forecastError').textContent='—';$('frozenError').textContent='—';$('updateTime').textContent='—';$('deadlinesMissed').textContent='—';$('recoveryTime').textContent='—';$('recoveryNote').textContent='must beat frozen for 2 s';$('pendulumClock').textContent='0.0 s';$('pendulumProgress').style.width='0';$('changeNotice').hidden=true;
  $('achievedRate').textContent='—';$('lateTicks').textContent='0';$('droppedSamples').textContent='0';$('clockDrift').textContent='—';
  $('runFinding').textContent='This is an engineered parameter-switch stress test. The physics changes 40% into the run and creates a frozen no-learning counterfactual.';
  drawChart();
}
function begin(){
  state=[1.72,0,1.08,0];physics={m1:1,m2:1,l1:1,l2:1,g:9.81,damping:.015};sampleId=0;totalSteps=config.duration*config.rate;changeStep=Math.round(totalSteps*.4);changed=false;
  $('deadlineText').textContent=`${config.interval.toFixed(1)} ms deadline`;
  setStatus('Learning the initial dynamics…',true);setRunning(true);scheduleStartedAt=performance.now();sendSample(PendulumCore.observation(state),scheduleStartedAt);drawPendulum();scheduleNext();
}
function advanceSample(deliver){
  sampleId++;
  if(sampleId===changeStep) applyPhysicsChange();
  state=PendulumCore.stepPhysics(state,physics,1/config.rate);
  const actual=PendulumCore.observation(state),prediction=predictions.get(sampleId);
  if(prediction){recordError(PendulumCore.angularErrorDegrees(prediction.adaptive,actual),prediction.frozen?PendulumCore.angularErrorDegrees(prediction.frozen,actual):null);predictions.delete(sampleId);}
  trail.push(position(state[0],state[2]));if(trail.length>90)trail.shift();
  const scheduledAt=scheduleStartedAt+sampleId*config.interval;
  if(deliver)sendSample(actual,scheduledAt);else droppedSamples++;
}
function schedule(){
  if(!running||settling)return;
  const now=performance.now(),due=Math.min(totalSteps,Math.floor((now-scheduleStartedAt)/config.interval)),backlog=Math.max(0,due-sampleId),toDrop=Math.max(0,backlog-MAX_CATCH_UP);
  for(let i=0;i<toDrop;i++)advanceSample(false);
  while(sampleId<due)advanceSample(true);
  drawPendulum();updateScheduleMetrics(now);
  if(sampleId>=totalSteps){endSchedule();return;}
  scheduleNext();
}
function scheduleNext(){
  clearTimeout(timer);const nextAt=scheduleStartedAt+(sampleId+1)*config.interval;timer=setTimeout(schedule,Math.max(0,nextAt-performance.now()));
}
function sendSample(value,scheduledAt){
  const dispatchLag=performance.now()-scheduledAt;if(sampleId>0&&dispatchLag>config.interval)lateTicks++;
  sentSamples++;inFlight++;worker?.postMessage({type:'sample',id:sampleId,observation:value,scheduledAt});
}
function updateScheduleMetrics(now=performance.now()){
  const elapsed=Math.max(.001,Math.min(config.duration,(now-scheduleStartedAt)/1000)),achieved=Math.max(0,sentSamples-1)/elapsed,currentLag=Math.max(0,now-(scheduleStartedAt+sampleId*config.interval));
  $('pendulumClock').textContent=(sampleId/config.rate).toFixed(1)+' s';$('pendulumProgress').style.width=Math.min(100,sampleId/totalSteps*100)+'%';
  $('achievedRate').textContent=elapsed>.25?achieved.toFixed(1)+' Hz':'—';$('lateTicks').textContent=lateTicks.toLocaleString();$('droppedSamples').textContent=droppedSamples.toLocaleString();$('clockDrift').textContent=currentLag.toFixed(1)+' ms';
}
function applyPhysicsChange(){
  changed=true;baselineError=recentErrors.length?recentErrors.reduce((a,b)=>a+b,0)/recentErrors.length:null;
  worker?.postMessage({type:'freeze'});const scenario=scenarios[config.scenario];scenario.apply(physics);$('changeDescription').textContent=scenario.description;$('changeNotice').hidden=false;
  $('runFinding').textContent='The readout has split in two: green keeps adapting; red is frozen at the old physics. Both receive the same reservoir state.';
  clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>{$('changeNotice').hidden=true;},2800);setStatus('Physics changed — adapting online…',true);
}
function recordError(adaptiveValue,frozenValue){
  adaptiveAverage=adaptiveAverage===null?adaptiveValue:adaptiveAverage*.88+adaptiveValue*.12;
  $('forecastError').textContent=adaptiveAverage.toFixed(1)+'°';
  const time=sampleId/config.rate;adaptiveErrors.push({time,value:adaptiveAverage});
  if(!changed){recentErrors.push(adaptiveValue);if(recentErrors.length>config.rate*3)recentErrors.shift();}
  if(frozenValue!==null){frozenAverage=frozenAverage===null?frozenValue:frozenAverage*.88+frozenValue*.12;$('frozenError').textContent=frozenAverage.toFixed(1)+'°';frozenErrors.push({time,value:frozenAverage});adaptivePostTotal+=adaptiveValue;frozenPostTotal+=frozenValue;postCount++;}
  if(changed&&recoverySeconds===null&&baselineError!==null){
    const threshold=Math.max(5,baselineError*1.25),windowSteps=config.rate*2,withinPriorRange=adaptiveAverage<=threshold,beatsFrozen=frozenAverage!==null&&adaptiveAverage<=frozenAverage*.75;
    recoveryStreak=withinPriorRange&&beatsFrozen?recoveryStreak+1:0;
    if(recoveryStreak>=windowSteps){recoverySeconds=(sampleId-changeStep)/config.rate;$('recoveryTime').textContent=recoverySeconds.toFixed(1)+' s';$('recoveryNote').textContent='prior range + 25% below frozen';setStatus('Recovery confirmed — continuing to learn',true);}
  }
  if(adaptiveErrors.length%3===0)drawChart();
}
function handleWorker(message){
  if(message.type==='ready'){$('runFinding').textContent=`Reservoir ready: ${config.size} fixed neurons, ${message.featureSize} readout features, ${(message.memoryBytes/1048576).toFixed(2)} MB model state.`;begin();return;}
  if(message.type==='error'){stopExperiment('Experiment failed: '+message.message);return;}
  if(message.type==='frozen')return;
  if(message.type!=='prediction'||!running)return;
  inFlight=Math.max(0,inFlight-1);const latency=performance.now()-message.scheduledAt;responseCount++;if(latency>config.interval)deadlineMisses++;
  updateAverage=updateAverage===null?message.computeMs:updateAverage*.9+message.computeMs*.1;
  $('updateTime').textContent=updateAverage<10?updateAverage.toFixed(2)+' ms':updateAverage.toFixed(1)+' ms';$('deadlinesMissed').textContent=deadlineMisses.toLocaleString();
  if(message.targetId>sampleId) predictions.set(message.targetId,{adaptive:message.prediction,frozen:message.frozenPrediction});latestAdaptive=message.prediction;latestFrozen=message.frozenPrediction;
  if(settling&&inFlight===0)finish();
}
function endSchedule(){
  clearTimeout(timer);timer=null;scheduleEndedAt=performance.now();settling=true;updateScheduleMetrics(scheduleEndedAt);setStatus('Finishing in-flight updates…',true);
  const drift=scheduleEndedAt-(scheduleStartedAt+totalSteps*config.interval);$('clockDrift').textContent=(drift>=0?'+':'')+drift.toFixed(1)+' ms';
  if(inFlight===0)finish();else finishTimer=setTimeout(finish,1200);
}
function finish(){
  if(!running)return;clearTimeout(finishTimer);finishTimer=null;
  if(inFlight>0){deadlineMisses+=inFlight;responseCount+=inFlight;inFlight=0;$('deadlinesMissed').textContent=deadlineMisses.toLocaleString();}
  worker?.terminate();worker=null;setRunning(false);settling=false;setStatus('Experiment complete');$('pendulumProgress').style.width='100%';
  const preMean=baselineError,adaptivePost=postCount?adaptivePostTotal/postCount:NaN,frozenPost=postCount?frozenPostTotal/postCount:NaN,advantage=Number.isFinite(frozenPost)&&frozenPost>0?(1-adaptivePost/frozenPost)*100:NaN,missRate=responseCount?deadlineMisses/responseCount*100:0;
  const comparison=Number.isFinite(advantage)?advantage>=0?`${advantage.toFixed(0)}% lower`:`${Math.abs(advantage).toFixed(0)}% higher`:'';
  const wallSeconds=(scheduleEndedAt-scheduleStartedAt)/1000,achieved=Math.max(0,sentSamples-1)/Math.max(.001,wallSeconds),drift=scheduleEndedAt-(scheduleStartedAt+totalSteps*config.interval);
  $('achievedRate').textContent=achieved.toFixed(1)+' Hz';$('lateTicks').textContent=lateTicks.toLocaleString();$('droppedSamples').textContent=droppedSamples.toLocaleString();$('clockDrift').textContent=(drift>=0?'+':'')+drift.toFixed(1)+' ms';
  if(recoverySeconds===null){$('recoveryTime').textContent='Not yet';$('recoveryNote').textContent='within this run';}
  $('runFinding').textContent=`Engineered ${scenarios[config.scenario].description.toLowerCase()}: during the three seconds before the switch, mean error was ${Number.isFinite(preMean)?preMean.toFixed(1)+'°':'unavailable'}. Afterwards, adaptive averaged ${Number.isFinite(adaptivePost)?adaptivePost.toFixed(1)+'°':'unavailable'} versus ${Number.isFinite(frozenPost)?frozenPost.toFixed(1)+'°':'unavailable'} frozen${comparison?` — ${comparison}`:''}. The stream achieved ${achieved.toFixed(1)} Hz, dropped ${droppedSamples} samples and missed ${deadlineMisses} of ${responseCount} response deadlines (${missRate.toFixed(1)}%)${recoverySeconds===null?'; recovery was not confirmed':` with recovery confirmed after ${recoverySeconds.toFixed(1)} seconds`}.`;
  drawChart();
}
function stopExperiment(message='Stopped. Start again for a fresh stream.'){
  clearTimeout(timer);timer=null;clearTimeout(finishTimer);finishTimer=null;clearTimeout(noticeTimer);worker?.terminate();worker=null;settling=false;setRunning(false);setStatus(message);$('changeNotice').hidden=true;
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
  if(latestFrozen){const [a,b]=PendulumCore.anglesFromObservation(latestFrozen);drawSystem(context,points(a,b,width,height),'#bd513c',true,.45);}
  if(latestAdaptive){const [a,b]=PendulumCore.anglesFromObservation(latestAdaptive);drawSystem(context,points(a,b,width,height),'#107c67',true,.65);}
  drawSystem(context,points(state?.[0]||1.72,state?.[2]||1.08,width,height),'#0d242b');
  context.fillStyle='#586c71';context.font='10px ui-monospace, monospace';context.fillText(`${config?.horizonMs||250} ms ahead`,14,22);
}
function drawChart(){
  const {context,width,height}=sizeCanvas(chartCanvas),left=34,right=10,top=10,bottom=25,maxTime=config?.duration||30,maxError=Math.max(30,...adaptiveErrors.map(point=>point.value),...frozenErrors.map(point=>point.value))*1.1;
  context.clearRect(0,0,width,height);context.font='9px ui-monospace, monospace';context.fillStyle='#586c71';context.strokeStyle='#dfe7e2';context.lineWidth=1;
  for(let i=0;i<=3;i++){const y=top+(height-top-bottom)*i/3,value=maxError*(1-i/3);context.beginPath();context.moveTo(left,y);context.lineTo(width-right,y);context.stroke();context.fillText(Math.round(value)+'°',2,y+3);}
  const changeX=left+(width-left-right)*.4;context.strokeStyle='#bd513c';context.beginPath();context.moveTo(changeX,top);context.lineTo(changeX,height-bottom);context.stroke();
  if(frozenErrors.length){context.strokeStyle='#bd513c';context.lineWidth=1.5;context.beginPath();frozenErrors.forEach((point,index)=>{const x=left+point.time/maxTime*(width-left-right),y=top+(1-point.value/maxError)*(height-top-bottom);if(index)context.lineTo(x,y);else context.moveTo(x,y);});context.stroke();}
  if(adaptiveErrors.length){context.strokeStyle='#107c67';context.lineWidth=2;context.beginPath();adaptiveErrors.forEach((point,index)=>{const x=left+point.time/maxTime*(width-left-right),y=top+(1-point.value/maxError)*(height-top-bottom);if(index)context.lineTo(x,y);else context.moveTo(x,y);});context.stroke();}
  context.fillStyle='#586c71';for(let i=0;i<=3;i++)context.fillText((maxTime*i/3).toFixed(0)+'s',left+(width-left-right)*i/3-7,height-5);
}

$('pendulumSetup').addEventListener('submit',event=>{
  event.preventDefault();stopExperiment('Preparing the reservoir…');resetMetrics();config=readConfig();setRunning(true);setStatus('Preparing the fixed reservoir…',true);
  worker=new Worker('/examples/double-pendulum/esn-worker.js?v=0.5.0');worker.onmessage=({data})=>handleWorker(data);worker.onerror=event=>stopExperiment('Could not start the learning worker: '+event.message);
  worker.postMessage({type:'init',size:config.size,seed:config.seed,horizon:config.horizon});
  if(innerWidth<=650)document.querySelector('.pendulum-results').scrollIntoView({behavior:'smooth',block:'start'});
});
$('stopPendulum').addEventListener('click',()=>stopExperiment());
addEventListener('resize',()=>{drawPendulum();drawChart();});
config=readConfig();state=[1.72,0,1.08,0];physics={m1:1,m2:1,l1:1,l2:1,g:9.81,damping:.015};drawPendulum();drawChart();
