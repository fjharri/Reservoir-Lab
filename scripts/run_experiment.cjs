// Runs exactly the browser's engine on Node's JS CPU backend. Not phone measurements.
const fs=require('node:fs'), path=require('node:path');
const root=path.resolve(__dirname,'..');
const tf=require(path.join(root,'public/vendor/tf.min.js'));
require(path.join(root,'public/examples/hydraulic/esn.js'));
const E=require(path.join(root,'public/examples/hydraulic/experiment.js'));
const meta=JSON.parse(fs.readFileSync(path.join(root,'public/examples/hydraulic/data/metadata.json')));
const data=Object.fromEntries(Object.entries(meta.splits).map(([key,split])=>{
 const b=fs.readFileSync(path.join(root,'public/examples/hydraulic/data',split.file));return [key,new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength))];
}));
const config={reservoirSize:500,gruUnits:16,ridge:.005,learningRate:.01,batchSize:8,seed:42,tolerance:0,maxSeconds:30,maxUpdates:1000,checkpointEvery:8,earlyStoppingPatience:6,...JSON.parse(process.argv[2]||'{}')};
E.run(tf,data,meta,config,event=>{
 if(['status','esn','point'].includes(event.type)) console.log(JSON.stringify(event));
}).then(result=>{
 const target=process.argv[3]||path.join(root,'experiment-result.json');
 fs.writeFileSync(target,JSON.stringify(result,null,2));
 console.log('SAVED',target,'test ESN',result.esn.test.accuracy,'GRU',result.gru.finalTest.accuracy);
}).catch(e=>{console.error(e);process.exitCode=1;});
