(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports) module.exports=api;
  root.PendulumCore=api;
})(typeof self!=='undefined'?self:globalThis,function(){
  'use strict';
  const TAU=Math.PI*2;
  const wrap=angle=>((angle+Math.PI)%TAU+TAU)%TAU-Math.PI;
  const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));

  function derivatives(state,physics){
    const [a,w,b,v]=state,{m1,m2,l1,l2,g,damping}=physics;
    const delta=a-b,den=2*m1+m2-m2*Math.cos(2*delta);
    const aa=(-g*(2*m1+m2)*Math.sin(a)-m2*g*Math.sin(a-2*b)-2*Math.sin(delta)*m2*(v*v*l2+w*w*l1*Math.cos(delta)))/(l1*den)-damping*w;
    const ba=(2*Math.sin(delta)*(w*w*l1*(m1+m2)+g*(m1+m2)*Math.cos(a)+v*v*l2*m2*Math.cos(delta)))/(l2*den)-damping*v;
    return [w,aa,v,ba];
  }

  function rk4(state,physics,dt){
    const k1=derivatives(state,physics);
    const k2=derivatives(state.map((x,i)=>x+k1[i]*dt/2),physics);
    const k3=derivatives(state.map((x,i)=>x+k2[i]*dt/2),physics);
    const k4=derivatives(state.map((x,i)=>x+k3[i]*dt),physics);
    return state.map((x,i)=>x+dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6);
  }

  function stepPhysics(state,physics,dt,substeps=4){
    let next=state;
    const h=dt/substeps;
    for(let i=0;i<substeps;i++) next=rk4(next,physics,h);
    next[0]=wrap(next[0]);next[2]=wrap(next[2]);
    next[1]=clamp(next[1],-20,20);next[3]=clamp(next[3],-20,20);
    return next;
  }

  function observation(state){
    return [Math.sin(state[0]),Math.cos(state[0]),clamp(state[1]/8,-2.5,2.5),Math.sin(state[2]),Math.cos(state[2]),clamp(state[3]/8,-2.5,2.5)];
  }
  function anglesFromObservation(value){return [Math.atan2(value[0],value[1]),Math.atan2(value[3],value[4])];}
  function angularErrorDegrees(predicted,actual){
    const p=anglesFromObservation(predicted),a=anglesFromObservation(actual);
    return (Math.abs(wrap(p[0]-a[0]))+Math.abs(wrap(p[1]-a[1])))*90/Math.PI;
  }

  function rng(seed){
    let value=seed>>>0;
    return ()=>{value+=0x6D2B79F5;let t=value;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};
  }

  class OnlineESN{
    constructor({size=200,seed=42,inputSize=6,outputSize=6,leak=.35,forgetting=.999,initialPrecision=4}={}){
      this.size=size;this.inputSize=inputSize;this.outputSize=outputSize;this.leak=leak;this.forgetting=forgetting;
      this.featureSize=1+inputSize+size;this.state=new Float64Array(size);this.next=new Float64Array(size);
      this.inputWeights=new Float64Array(size*inputSize);this.bias=new Float64Array(size);
      this.links=new Int32Array(size*3);this.recurrent=new Float64Array(size*3);
      this.weights=new Float64Array(this.featureSize*outputSize);this.precision=new Float64Array(this.featureSize*this.featureSize);
      this.px=new Float64Array(this.featureSize);this.gain=new Float64Array(this.featureSize);this.errors=new Float64Array(outputSize);
      const random=rng(seed);
      for(let i=0;i<size;i++){
        this.bias[i]=(random()*2-1)*.18;
        for(let j=0;j<inputSize;j++) this.inputWeights[i*inputSize+j]=(random()*2-1)*.55;
        const base=i*3;this.links[base]=(i-1+size)%size;this.links[base+1]=(i+1+Math.floor(random()*(size-1)))%size;this.links[base+2]=Math.floor(random()*size);
        const raw=[random()*2-1,random()*2-1,random()*2-1],scale=.88/(Math.abs(raw[0])+Math.abs(raw[1])+Math.abs(raw[2]));
        for(let k=0;k<3;k++) this.recurrent[base+k]=raw[k]*scale;
      }
      for(let i=0;i<this.featureSize;i++) this.precision[i*this.featureSize+i]=initialPrecision;
      this.updates=0;
    }
    advance(input){
      for(let i=0;i<this.size;i++){
        let sum=this.bias[i],offset=i*this.inputSize,base=i*3;
        for(let j=0;j<this.inputSize;j++) sum+=this.inputWeights[offset+j]*input[j];
        for(let k=0;k<3;k++) sum+=this.recurrent[base+k]*this.state[this.links[base+k]];
        this.next[i]=(1-this.leak)*this.state[i]+this.leak*Math.tanh(sum);
      }
      [this.state,this.next]=[this.next,this.state];
      const feature=new Float64Array(this.featureSize);feature[0]=1;
      feature.set(input,1);feature.set(this.state,1+this.inputSize);
      return feature;
    }
    predict(feature){
      const output=new Float64Array(this.outputSize);
      for(let i=0;i<this.featureSize;i++) for(let o=0;o<this.outputSize;o++) output[o]+=feature[i]*this.weights[i*this.outputSize+o];
      return output;
    }
    update(feature,target){
      const n=this.featureSize,p=this.precision;
      for(let i=0;i<n;i++){let sum=0,offset=i*n;for(let j=0;j<n;j++) sum+=p[offset+j]*feature[j];this.px[i]=sum;}
      let denominator=this.forgetting;for(let i=0;i<n;i++) denominator+=feature[i]*this.px[i];
      for(let i=0;i<n;i++) this.gain[i]=this.px[i]/denominator;
      const prediction=this.predict(feature);
      for(let o=0;o<this.outputSize;o++) this.errors[o]=target[o]-prediction[o];
      for(let i=0;i<n;i++) for(let o=0;o<this.outputSize;o++) this.weights[i*this.outputSize+o]+=this.gain[i]*this.errors[o];
      for(let i=0;i<n;i++){const offset=i*n,g=this.gain[i];for(let j=0;j<n;j++) p[offset+j]=(p[offset+j]-g*this.px[j])/this.forgetting;}
      this.updates++;
    }
    estimatedBytes(){return this.state.byteLength+this.next.byteLength+this.inputWeights.byteLength+this.bias.byteLength+this.links.byteLength+this.recurrent.byteLength+this.weights.byteLength+this.precision.byteLength;}
  }

  return {wrap,stepPhysics,observation,anglesFromObservation,angularErrorDegrees,OnlineESN};
});
