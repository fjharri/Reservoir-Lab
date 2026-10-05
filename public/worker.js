importScripts('vendor/tf.min.js', 'esn.js?v=0.2.0', 'experiment.js?v=0.2.0');
self.onmessage=async ({data:message})=> {
  try {
    const meta=await (await fetch('data/metadata.json')).json();
    const entries=await Promise.all(Object.entries(meta.splits).map(async ([key,split])=> {
      const response=await fetch('data/'+split.file); if(!response.ok) throw Error('Could not load '+split.file);
      return [key,new Float32Array(await response.arrayBuffer())];
    }));
    postMessage({type:'data',meta:{steps:meta.steps,channels:meta.channels,counts:Object.fromEntries(Object.entries(meta.splits).map(([k,v])=>[k,v.count]))}});
    const result=await PilotExperiment.run(tf,Object.fromEntries(entries),meta,message.config,event=>postMessage(event));
    postMessage({type:'done',result});
  } catch(error) {postMessage({type:'error',message:error.message,stack:error.stack});}
};
