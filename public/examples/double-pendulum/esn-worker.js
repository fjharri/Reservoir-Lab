'use strict';
importScripts('core.js?v=0.4.0');
let model=null,horizon=1,features=[],frozenWeights=null;
self.onmessage=({data:message})=>{
  try{
    if(message.type==='init'){
      model=new PendulumCore.OnlineESN({size:message.size,seed:message.seed});horizon=message.horizon;features=[];frozenWeights=null;
      postMessage({type:'ready',memoryBytes:model.estimatedBytes(),featureSize:model.featureSize});return;
    }
    if(message.type==='freeze'&&model){frozenWeights=model.weights.slice();postMessage({type:'frozen',updates:model.updates});return;}
    if(message.type!=='sample'||!model) return;
    const started=performance.now(),input=Float64Array.from(message.observation);
    if(features.length>=horizon) model.update(features.shift(),input);
    const feature=model.advance(input),prediction=model.predict(feature),frozenPrediction=frozenWeights?model.predict(feature,frozenWeights):null;features.push(feature);
    postMessage({type:'prediction',id:message.id,targetId:message.id+horizon,prediction:Array.from(prediction),frozenPrediction:frozenPrediction?Array.from(frozenPrediction):null,computeMs:performance.now()-started,updates:model.updates,sentAt:message.sentAt});
  }catch(error){postMessage({type:'error',message:error.message,stack:error.stack});}
};
