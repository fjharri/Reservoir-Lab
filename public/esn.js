/* Sparse, fixed echo-state reservoir and ridge readout. No autograd. */
(function (scope) {
  'use strict';
  function random(seed) {
    let a = seed >>> 0;
    return () => { a += 0x6D2B79F5; let t = a; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function shuffle(n, rng) {
    const a = Array.from({length:n}, (_, i) => i);
    for (let i=n-1;i>0;i--) { const j=Math.floor(rng()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
    return a;
  }
  class Reservoir {
    constructor(size, inputs, seed) {
      this.size=size; this.inputs=inputs;
      const rng=random(seed);
      this.win=Float64Array.from({length:size*inputs}, () => (rng()*2-1)*.7);
      this.bias=Float64Array.from({length:size}, () => (rng()*2-1)*.2);
      this.links=new Int32Array(size*3); this.weights=new Float64Array(size*3);
      for(let i=0;i<size;i++) {
        this.links[i*3]=(i+size-1)%size; this.links[i*3+1]=Math.floor(rng()*size); this.links[i*3+2]=Math.floor(rng()*size);
        this.weights[i*3]=(rng()<.5?-1:1)*.8;
        this.weights[i*3+1]=(rng()<.5?-1:1)*.05; this.weights[i*3+2]=(rng()<.5?-1:1)*.05;
      }
    }
    features(data, count, steps) {
      const n=this.size, d=this.inputs, result=new Float32Array(count*n);
      let h=new Float64Array(n), next=new Float64Array(n);
      for(let cycle=0;cycle<count;cycle++) {
        h.fill(0); next.fill(0); // Independent complete cycles; no hidden-state leakage.
        for(let t=0;t<steps;t++) {
          const offset=(cycle*steps+t)*d;
          for(let i=0;i<n;i++) {
            let s=this.bias[i];
            for(let c=0;c<d;c++) s+=this.win[i*d+c]*data[offset+c];
            for(let k=0;k<3;k++) s+=this.weights[i*3+k]*h[this.links[i*3+k]];
            next[i]=.5*h[i]+.5*Math.tanh(s);
            result[cycle*n+i]+=next[i]/steps;
          }
          [h,next]=[next,h];
        }
      }
      return result;
    }
  }
  function choleskySolve(a, rhs, n, columns) {
    const l=new Float64Array(n*n), y=new Float64Array(n*columns), out=new Float64Array(n*columns);
    for(let i=0;i<n;i++) for(let j=0;j<=i;j++) {
      let v=a[i*n+j]; for(let k=0;k<j;k++) v-=l[i*n+k]*l[j*n+k];
      if(i===j) { if(!(v>0)) throw Error('Ridge matrix is not positive definite.'); l[i*n+j]=Math.sqrt(v); }
      else l[i*n+j]=v/l[j*n+j];
    }
    for(let c=0;c<columns;c++) {
      for(let i=0;i<n;i++) { let v=rhs[i*columns+c]; for(let j=0;j<i;j++) v-=l[i*n+j]*y[j*columns+c]; y[i*columns+c]=v/l[i*n+i]; }
      for(let i=n-1;i>=0;i--) { let v=y[i*columns+c]; for(let j=i+1;j<n;j++) v-=l[j*n+i]*out[j*columns+c]; out[i*columns+c]=v/l[i*n+i]; }
    }
    return out;
  }
  function ridge(features, labels, width, classes, lambda=.01) {
    const count=labels.length, mean=new Float64Array(width), scale=new Float64Array(width), prior=new Float64Array(classes);
    for(let r=0;r<count;r++) { prior[labels[r]]+=1/count; for(let c=0;c<width;c++) mean[c]+=features[r*width+c]/count; }
    for(let r=0;r<count;r++) for(let c=0;c<width;c++) scale[c]+=(features[r*width+c]-mean[c])**2/count;
    for(let c=0;c<width;c++) scale[c]=Math.max(Math.sqrt(scale[c]),1e-6);
    const x=new Float64Array(count*width), y=new Float64Array(count*classes);
    for(let r=0;r<count;r++) {
      for(let c=0;c<width;c++) x[r*width+c]=(features[r*width+c]-mean[c])/scale[c];
      for(let k=0;k<classes;k++) y[r*classes+k]=(labels[r]===k?1:0)-prior[k];
    }
    let weights;
    // Dual solve avoids a large reservoir-sized cubic system when width > sample count.
    if(width>count) {
      const gram=new Float64Array(count*count);
      for(let i=0;i<count;i++) for(let j=0;j<=i;j++) {
        let v=0; for(let k=0;k<width;k++) v+=x[i*width+k]*x[j*width+k];
        gram[i*count+j]=gram[j*count+i]=v+(i===j?count*lambda:0);
      }
      const alpha=choleskySolve(gram,y,count,classes); weights=new Float64Array(width*classes);
      for(let c=0;c<width;c++) for(let k=0;k<classes;k++) for(let r=0;r<count;r++) weights[c*classes+k]+=x[r*width+c]*alpha[r*classes+k];
    } else {
      const gram=new Float64Array(width*width), rhs=new Float64Array(width*classes);
      for(let i=0;i<width;i++) {
        for(let j=0;j<=i;j++) { let v=0; for(let r=0;r<count;r++) v+=x[r*width+i]*x[r*width+j]; gram[i*width+j]=gram[j*width+i]=v+(i===j?count*lambda:0); }
        for(let k=0;k<classes;k++) for(let r=0;r<count;r++) rhs[i*classes+k]+=x[r*width+i]*y[r*classes+k];
      }
      weights=choleskySolve(gram,rhs,width,classes);
    }
    return {width,classes,mean,scale,prior,weights};
  }
  function predict(readout, features) {
    const {width,classes,mean,scale,prior,weights}=readout, count=features.length/width, labels=[];
    for(let r=0;r<count;r++) {
      let best=-Infinity, label=0;
      for(let k=0;k<classes;k++) {
        let score=prior[k]; for(let c=0;c<width;c++) score+=(features[r*width+c]-mean[c])/scale[c]*weights[c*classes+k];
        if(score>best) {best=score;label=k;}
      }
      labels.push(label);
    }
    return labels;
  }
  function metrics(predicted, actual, classes=4) {
    const confusion=Array.from({length:classes},()=>Array(classes).fill(0)); let correct=0;
    actual.forEach((y,i)=>{ confusion[y][predicted[i]]++; if(y===predicted[i]) correct++; });
    return {accuracy:correct/actual.length, count:actual.length, confusion};
  }
  const api={random,shuffle,Reservoir,choleskySolve,ridge,predict,metrics};
  scope.PilotCore=api;
  if(typeof module!=='undefined') module.exports=api;
})(typeof self!=='undefined'?self:globalThis);
