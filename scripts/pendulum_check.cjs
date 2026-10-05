const assert=require('node:assert/strict');
const Core=require('../public/examples/double-pendulum/core.js');

const base={m1:1,m2:1,l1:1,l2:1,g:9.81,damping:.015};
let state=[1.72,0,1.08,0];
for(let i=0;i<6000;i++){
  state=Core.stepPhysics(state,base,1/60);
  assert.ok(state.every(Number.isFinite),'Pendulum integration must remain finite');
  assert.ok(Math.abs(state[0])<=Math.PI&&Math.abs(state[2])<=Math.PI,'Angles must remain wrapped');
}

let original=[1.3,.1,.8,-.2],changed=original.slice();
const altered={...base,m2:2.2};
for(let i=0;i<300;i++){original=Core.stepPhysics(original,base,1/60);changed=Core.stepPhysics(changed,altered,1/60);}
assert.ok(original.some((value,index)=>Math.abs(value-changed[index])>.05),'Changing the load must change the trajectory');

const first=new Core.OnlineESN({size:24,seed:7}),second=new Core.OnlineESN({size:24,seed:7});
const sample=Core.observation([1,.2,-.4,.1]);
assert.deepEqual([...first.advance(sample)],[...second.advance(sample)],'Reservoir state must be seeded and deterministic');

const sensorA=Core.createSensorModel({rate:60,seed:19}),sensorB=Core.createSensorModel({rate:60,seed:19});
const sensedA=[sensorA.observe([1,0,-.4,0]),sensorA.observe([1.01,.6,-.39,.6])],sensedB=[sensorB.observe([1,0,-.4,0]),sensorB.observe([1.01,.6,-.39,.6])];
assert.deepEqual(sensedA,sensedB,'Noisy sensor observations must be deterministic for a seed');
assert.notDeepEqual(sensedA[1],Core.observation([1.01,.6,-.39,.6]),'Sensor mode must not expose perfect simulator state');
assert.ok(Math.abs(sensedA[1][2])>0&&Math.abs(sensedA[1][5])>0,'Angular velocity must be derived from successive angle readings');

const model=new Core.OnlineESN({size:40,seed:42}),queue=[],pending=new Map(),early=[],late=[],horizon=4;
for(let step=0;step<900;step++){
  const t=step/30,target=[Math.sin(t),Math.cos(t),Math.sin(t*.37),Math.sin(t*.71),Math.cos(t*.71),Math.cos(t*.23)];
  if(queue.length>=horizon)model.update(queue.shift(),target);
  const feature=model.advance(target),prediction=model.predict(feature);queue.push(feature);pending.set(step+horizon,prediction);
  const due=pending.get(step);
  if(due){const mse=due.reduce((sum,value,index)=>sum+(value-target[index])**2,0)/target.length;if(step<260)early.push(mse);if(step>700)late.push(mse);pending.delete(step);}
}
const mean=values=>values.reduce((sum,value)=>sum+value,0)/values.length;
assert.ok(mean(late)<mean(early)*.35,`Online readout should learn the delayed stream (${mean(early)} -> ${mean(late)})`);
const probe=model.advance([0,.8,.1,.2,.9,-.1]),frozen=model.weights.slice(),frozenBefore=model.predict(probe,frozen);
model.update(probe,[1,0,1,0,1,0]);
assert.deepEqual([...model.predict(probe,frozen)],[...frozenBefore],'Frozen readout prediction must remain unchanged while the adaptive weights update');
assert.notDeepEqual([...model.predict(probe)],[...frozenBefore],'Adaptive readout must diverge from its frozen snapshot after an update');
assert.ok(model.estimatedBytes()<2_000_000,'Default-scale online model should have bounded memory');
console.log('PASS: stable double-pendulum integration, changing dynamics, deterministic reservoir and noisy sensors, online delayed learning, frozen-readout counterfactual, bounded model memory.');
