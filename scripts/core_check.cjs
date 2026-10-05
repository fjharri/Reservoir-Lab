const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const C=require('../public/esn.js'),root=path.resolve(__dirname,'..');
const E=require('../public/experiment.js');
const meta=JSON.parse(fs.readFileSync(path.join(root,'public/data/metadata.json')));
const sets=Object.values(meta.splits).map(s=>new Set(s.runIds));
for(let i=0;i<sets.length;i++)for(let j=i+1;j<sets.length;j++)assert.equal([...sets[i]].filter(x=>sets[j].has(x)).length,0,'Acquisition runs must not cross splits');
const cycles=Object.values(meta.splits).flatMap(s=>s.cycleIds);assert.equal(new Set(cycles).size,640);
const b=fs.readFileSync(path.join(root,'public/data/train.bin'));const data=new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
assert.equal(data.length,400*120*6);
for(let c=0;c<6;c++) {let mean=0,sq=0,n=0;for(let i=c;i<data.length;i+=6){mean+=data[i];sq+=data[i]**2;n++;}assert.ok(Math.abs(mean/n)<1e-5);assert.ok(Math.abs(sq/n-1)<1e-5);}
// Independent SPD solve checked against its original equations.
const a=Float64Array.from([4,1,1,3]),rhs=Float64Array.from([1,2]),sol=C.choleskySolve(a,rhs,2,1);
assert.ok(Math.abs(4*sol[0]+sol[1]-1)<1e-10);assert.ok(Math.abs(sol[0]+3*sol[1]-2)<1e-10);
const reservoir=new C.Reservoir(12,6,42),first=data.subarray(0,120*6),next=data.subarray(120*6,240*6);
const joined=new Float32Array([...first,...next]);
assert.deepEqual([...reservoir.features(joined,2,120).slice(12)],[...reservoir.features(next,1,120)],'Cycle states must reset');
assert.deepEqual([...new C.Reservoir(12,6,42).features(first,1,120)],[...reservoir.features(first,1,120)],'Seeded reservoir reproducibility');
// Both primal and dual readout paths must recover a separable toy task.
for(const width of [2,8]) {
 const features=Float32Array.from(Array.from({length:6},(_,i)=>Array.from({length:width},(_,j)=>(i<3?-1:1)*(j+1)+i*.01)).flat());
 const labels=[0,0,0,1,1,1];const fit=C.ridge(features,labels,width,2,.001);assert.deepEqual(C.predict(fit,features),labels);
}
assert.deepEqual(E.budgetStateAfterSetup(50,100),{budgetPassed:false,crossingOvershootMs:null});
assert.deepEqual(E.budgetStateAfterSetup(125,100),{budgetPassed:true,crossingOvershootMs:25});
assert.equal(E.shouldEarlyStop(false,true,6,6),false,'Early stopping must wait until the validation target is matched');
assert.equal(E.shouldEarlyStop(true,false,6,6),false,'Early stopping must preserve the full ESN comparison budget');
assert.equal(E.shouldEarlyStop(true,true,5,6),false,'Early stopping must respect patience');
assert.equal(E.shouldEarlyStop(true,true,6,6),true,'Early stopping should trigger when patience is exhausted');
console.log('PASS: split separation, train scaling, solver residuals, state reset, deterministic features, primal/dual readout predictions, setup budget detection, early-stopping gate.');
