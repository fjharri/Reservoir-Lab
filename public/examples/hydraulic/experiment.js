/* Shared experiment protocol, run in a Web Worker in the browser. */
(function(scope) {
  'use strict';
  const C=scope.PilotCore, now=()=>performance.now(), pause=()=>new Promise(resolve=>setTimeout(resolve,0));
  function snapshot(model) { return model.getWeights().map(w=>w.clone()); }
  function disposeWeights(weights) { weights.forEach(w=>w.dispose()); }
  function budgetStateAfterSetup(trainingMs, budgetMs) {
    const budgetPassed=trainingMs>budgetMs;
    return {budgetPassed,crossingOvershootMs:budgetPassed?trainingMs-budgetMs:null};
  }
  function shouldEarlyStop(hasMatched, budgetPassed, checksWithoutImprovement, patienceChecks) {
    return hasMatched&&budgetPassed&&checksWithoutImprovement>=patienceChecks;
  }
  function createGru(tf, steps, inputs, units, seed, learningRate=.003) {
    const model=tf.sequential();
    model.add(tf.layers.gru({units,inputShape:[steps,inputs],returnSequences:true,
      kernelInitializer:tf.initializers.glorotUniform({seed}),
      recurrentInitializer:tf.initializers.glorotUniform({seed:seed+1}),
      biasInitializer:'zeros'}));
    model.add(tf.layers.globalAveragePooling1d({}));
    model.add(tf.layers.dense({units:4,activation:'softmax',kernelInitializer:tf.initializers.glorotUniform({seed:seed+2})}));
    model.compile({optimizer:tf.train.adam(learningRate),loss:'categoricalCrossentropy'});
    return model;
  }
  function evaluateGru(tf, model, values, labels, steps, inputs) {
    // Chunked inference limits memory on mobile. Does not update weights.
    const predictions=[];
    for(let offset=0;offset<labels.length;offset+=16) {
      const count=Math.min(16,labels.length-offset);
      const part=tf.tidy(()=> {
        const x=tf.tensor3d(values.subarray(offset*steps*inputs,(offset+count)*steps*inputs),[count,steps,inputs]);
        return Array.from(model.predict(x).argMax(-1).dataSync());
      });
      predictions.push(...part);
    }
    return C.metrics(predictions,labels);
  }
  async function run(tf, data, meta, config, emit=()=>{}) {
    await tf.setBackend('cpu'); await tf.ready();
    const startWall=now(), {steps}=meta, inputs=meta.channels.length, seed=config.seed;
    const earlyStoppingPatience=config.earlyStoppingPatience??6;
    const labels=Object.fromEntries(Object.entries(meta.splits).map(([k,v])=>[k,v.labels]));
    const result={protocolVersion:2,createdAt:new Date().toISOString(),config,
      environment:{tfjs:tf.version.tfjs,backend:tf.getBackend(),userAgent:typeof navigator!=='undefined'?navigator.userAgent:'Node'},
      data:{source:meta.source,splitMethod:meta.splitMethod,seed:meta.seed,steps,channels:meta.channels,
        classes:meta.classNames,counts:Object.fromEntries(Object.entries(meta.splits).map(([k,v])=>[k,v.count])),
        splitHashes:Object.fromEntries(Object.entries(meta.splits).map(([k,v])=>[k,v.sha256]))},
      curve:[],targetTolerance:config.tolerance};
    emit({type:'status',message:'ESN: running the fixed reservoir on training cycles…'}); await pause();
    let t=now();
    const reservoir=new C.Reservoir(config.reservoirSize,inputs,seed);
    const trainFeatures=reservoir.features(data.train,labels.train.length,steps);
    const featureMs=now()-t;
    emit({type:'status',message:'ESN: fitting the ridge readout…'}); await pause();
    t=now(); const readout=C.ridge(trainFeatures,labels.train,config.reservoirSize,4,config.ridge); const readoutMs=now()-t;
    const budgetMs=featureMs+readoutMs;
    t=now();
    const valFeatures=reservoir.features(data.validation,labels.validation.length,steps);
    const esnValidation=C.metrics(C.predict(readout,valFeatures),labels.validation);
    const validationMs=now()-t;
    // Test is deliberately kept untouched until all GRU training and selection finish.
    result.esn={trainingMs:budgetMs,featureMs,readoutMs,validationMs,validation:esnValidation,
      fixedParameters:config.reservoirSize*(inputs+4),trainableParameters:(config.reservoirSize+1)*4};
    result.targetAccuracy=Math.max(0,esnValidation.accuracy-config.tolerance);
    emit({type:'esn',esn:result.esn,targetAccuracy:result.targetAccuracy}); await pause();
    emit({type:'status',message:'GRU: initialising trainable recurrent weights…'});
    t=now(); const model=createGru(tf,steps,inputs,config.gruUnits,seed,config.learningRate);
    // Data tensor materialization is part of the GRU training cost.
    const xs=tf.tensor3d(data.train,[labels.train.length,steps,inputs]);
    const ys=tf.tidy(()=>tf.oneHot(tf.tensor1d(labels.train,'int32'),4));
    let trainingMs=now()-t, evaluationMs=0, update=0, epoch=0, loss=null;
    let {budgetPassed,crossingOvershootMs}=budgetStateAfterSetup(trainingMs,budgetMs);
    const parameterCount=model.countParams();
    let equalWeights=null, equalPoint=null, matchWeights=null, matchPoint=null, bestWeights=null, bestPoint=null;
    let checksWithoutImprovement=0, earlyStopped=false;
    if(trainingMs<=budgetMs) {equalWeights=snapshot(model);equalPoint={trainingMs,updates:0,epoch:0};}
    t=now(); const initial=evaluateGru(tf,model,data.validation,labels.validation,steps,inputs); evaluationMs+=now()-t;
    const initialPoint={trainingMs,updates:0,epoch:0,accuracy:initial.accuracy,loss:null};
    result.curve.push(initialPoint); emit({type:'point',point:initialPoint});
    bestPoint={...initialPoint,validation:initial};bestWeights=snapshot(model);
    if(equalPoint) equalPoint={...equalPoint,validation:initial};
    if(initial.accuracy>=result.targetAccuracy) {matchPoint={...initialPoint,validation:initial};matchWeights=snapshot(model);}
    const rng=C.random(seed+19);
    try {
      while(trainingMs<config.maxSeconds*1000 && update<config.maxUpdates && !earlyStopped) {
        const order=C.shuffle(labels.train.length,rng); epoch++;
        for(let offset=0;offset<order.length;offset+=config.batchSize) {
          t=now();
          const indices=tf.tensor1d(order.slice(offset,offset+config.batchSize),'int32');
          const xb=tf.gather(xs,indices), yb=tf.gather(ys,indices);
          loss=await model.trainOnBatch(xb,yb);
          xb.dispose();yb.dispose();indices.dispose();
          trainingMs+=now()-t; update++;
          const withinBudget=trainingMs<=budgetMs;
          if(withinBudget) {
            if(equalWeights) disposeWeights(equalWeights);
            equalWeights=snapshot(model); equalPoint={trainingMs,updates:update,epoch};
          }
          const crossesBudget=!budgetPassed&&!withinBudget;
          if(crossesBudget) {budgetPassed=true;crossingOvershootMs=trainingMs-budgetMs;}
          const checkpoint=crossesBudget || update%config.checkpointEvery===0 || offset+config.batchSize>=order.length;
          if(checkpoint) {
            t=now(); const validation=evaluateGru(tf,model,data.validation,labels.validation,steps,inputs); evaluationMs+=now()-t;
            const point={trainingMs,updates:update,epoch,accuracy:validation.accuracy,loss:Array.isArray(loss)?loss[0]:loss};
            result.curve.push(point); emit({type:'point',point});
            if(withinBudget&&equalPoint) equalPoint={...equalPoint,validation};
            if(validation.accuracy>bestPoint.accuracy) {
              disposeWeights(bestWeights);bestWeights=snapshot(model);bestPoint={...point,validation};checksWithoutImprovement=0;
            } else checksWithoutImprovement++;
            if(!matchPoint&&validation.accuracy>=result.targetAccuracy) {matchPoint={...point,validation};matchWeights=snapshot(model);}
            if(shouldEarlyStop(Boolean(matchPoint),budgetPassed,checksWithoutImprovement,earlyStoppingPatience)) earlyStopped=true;
          }
          emit({type:'progress',trainingMs,updates:update,epoch,maxMs:config.maxSeconds*1000});
          await pause();
          if(earlyStopped || trainingMs>=config.maxSeconds*1000 || update>=config.maxUpdates) break;
        }
      }
      // The final completed batch may exceed a deadline. Never use it as the equal-budget model.
      const finalTrainingMs=trainingMs;
      if(equalWeights) {
        model.setWeights(equalWeights); t=now();
        equalPoint.validation=evaluateGru(tf,model,data.validation,labels.validation,steps,inputs); evaluationMs+=now()-t;
        if(!result.curve.some(p=>p.updates===equalPoint.updates)) {
          result.curve.push({trainingMs:equalPoint.trainingMs,updates:equalPoint.updates,epoch:equalPoint.epoch,accuracy:equalPoint.validation.accuracy,loss:null,selectedEqualBudget:true});
          result.curve.sort((a,b)=>a.trainingMs-b.trainingMs);
        }
      }
      result.gru={trainableParameters:parameterCount,updates:update,epochsStarted:epoch,trainingMs:finalTrainingMs,
        equalBudget:equalPoint,matched:matchPoint,finalized:bestPoint,validationEvaluationMs:evaluationMs,
        reason:earlyStopped?'early-stopping':trainingMs>=config.maxSeconds*1000?'time-cap':'update-cap',
        earlyStopping:{metric:'validation-accuracy',patienceChecks:earlyStoppingPatience,checksWithoutImprovement},
        budgetOvershootMs:crossingOvershootMs,capOvershootMs:Math.max(0,finalTrainingMs-config.maxSeconds*1000),
        noBudgetCheckpoint:!equalPoint};
      emit({type:'status',message:'Training complete. Evaluating the held-out test cycles…'}); await pause();
      t=now(); const testFeatures=reservoir.features(data.test,labels.test.length,steps);
      result.esn.test=C.metrics(C.predict(readout,testFeatures),labels.test); result.esn.testMs=now()-t;
      if(equalWeights) {model.setWeights(equalWeights);t=now();result.gru.equalBudget.test=evaluateGru(tf,model,data.test,labels.test,steps,inputs);result.gru.equalBudget.testMs=now()-t;}
      if(matchWeights) {model.setWeights(matchWeights);t=now();result.gru.matched.test=evaluateGru(tf,model,data.test,labels.test,steps,inputs);result.gru.matched.testMs=now()-t;}
      model.setWeights(bestWeights);t=now();result.gru.finalized.validation=evaluateGru(tf,model,data.validation,labels.validation,steps,inputs);result.gru.finalized.test=evaluateGru(tf,model,data.test,labels.test,steps,inputs);result.gru.finalized.train=evaluateGru(tf,model,data.train,labels.train,steps,inputs);result.gru.finalized.evaluationMs=now()-t;
      // Compatibility aliases for earlier result readers.
      result.gru.finalValidation=result.gru.finalized.validation;result.gru.finalTest=result.gru.finalized.test;result.gru.finalTrain=result.gru.finalized.train;result.gru.finalEvaluationMs=result.gru.finalized.evaluationMs;
      result.wallMs=now()-startWall;
      result.notes=['Training clock includes model setup and training operations, excludes download, validation/test and snapshot logging. ESN training includes train reservoir extraction and ridge fit.',
        'GRU equal-budget model is the last complete batch (including setup) within the ESN budget. A batch cannot be interrupted; overshoot is excluded from this selected model. If setup alone exceeds the budget, no equal-budget checkpoint is eligible, but training can continue toward the validation target.',
        'Target is ESN validation accuracy minus the stated tolerance. Selection never uses test accuracy. First observed validation crossing is a checkpoint-based upper bound on time to target.',
        'After reaching the target, GRU training uses early stopping. The finalized GRU restores the checkpoint with the highest observed validation accuracy and stops after the configured number of validation checks without improvement, or at the training cap.',
        'Both implementations use CPU in a worker and mean pooling across all 120 steps. Sparse custom JS ESN versus TensorFlow.js GRU is an implementation-level browser comparison.',
        'A single rig and deterministic pilot split do not establish generalization to new equipment. Repeat seeds/devices; backend, width and hyperparameters affect results.'];
      return result;
    } finally {
      if(equalWeights) disposeWeights(equalWeights);if(matchWeights) disposeWeights(matchWeights);if(bestWeights) disposeWeights(bestWeights);
      xs.dispose();ys.dispose();model.optimizer.dispose();model.dispose();
    }
  }
  scope.PilotExperiment={run,createGru,evaluateGru,budgetStateAfterSetup,shouldEarlyStop};
  if(typeof module!=='undefined') module.exports=scope.PilotExperiment;
})(typeof self!=='undefined'?self:globalThis);
