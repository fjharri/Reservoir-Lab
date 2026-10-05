'use strict';
importScripts('core.js?v=0.7.0');

const seeds=[1,7,42,99,123];
const scenarios={
  load:p=>{p.m2=2.2;},
  length:p=>{p.l2=.62;},
  damping:p=>{p.damping=.12;},
  combined:p=>{p.m2=2.6;p.l2=.55;p.damping=.12;}
};

function run(seed,config){
  const model=new PendulumCore.OnlineESN({size:config.size,seed}),features=[],predictions=new Map();
  const sensor=config.sensorNoise?PendulumCore.createSensorModel({rate:config.rate,seed:seed^0x51f15e}):null;
  let state=[1.72,0,1.08,0],physics={m1:1,m2:1,l1:1,l2:1,g:9.81,damping:.015},frozenWeights=null,adaptiveTotal=0,frozenTotal=0,count=0;
  const totalSteps=config.duration*config.rate,changeStep=Math.round(totalSteps*.4);
  const consume=(id,input)=>{
    while(features.length&&features[0].targetId<=id){const delayed=features.shift();if(delayed.targetId===id)model.update(delayed.feature,input);}
    const feature=model.advance(input),adaptive=model.predict(feature),frozen=frozenWeights?model.predict(feature,frozenWeights):null;
    features.push({targetId:id+config.horizon,feature});predictions.set(id+config.horizon,{adaptive,frozen});
  };
  consume(0,sensor?sensor.observe(state):PendulumCore.observation(state));
  for(let id=1;id<=totalSteps;id++){
    if(id===changeStep){frozenWeights=model.weights.slice();scenarios[config.scenario](physics);}
    state=PendulumCore.stepPhysics(state,physics,1/config.rate);
    const truth=PendulumCore.observation(state),due=predictions.get(id);
    if(due?.frozen){adaptiveTotal+=PendulumCore.angularErrorDegrees(due.adaptive,truth);frozenTotal+=PendulumCore.angularErrorDegrees(due.frozen,truth);count++;}
    predictions.delete(id);consume(id,sensor?sensor.observe(state):truth);
  }
  const adaptive=adaptiveTotal/count,frozen=frozenTotal/count;
  return {seed,adaptive,frozen,advantage:(1-adaptive/frozen)*100};
}

self.onmessage=({data})=>{
  try{
    if(data.type!=='run')return;
    const results=[];
    for(let index=0;index<seeds.length;index++){results.push(run(seeds[index],data.config));postMessage({type:'progress',complete:index+1,total:seeds.length});}
    postMessage({type:'complete',results});
  }catch(error){postMessage({type:'error',message:error.message,stack:error.stack});}
};
