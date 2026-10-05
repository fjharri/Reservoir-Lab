const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES?process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES+'/playwright':'playwright');
const root=path.resolve(__dirname,'..');
const baseUrl=process.env.RESERVOIR_LAB_URL||'http://127.0.0.1:8765';
(async()=>{
 fs.mkdirSync(path.join(root,'work'),{recursive:true});
 const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.PILOT_CHROME_PATH?{executablePath:process.env.PILOT_CHROME_PATH}:{})});
 const page=await browser.newPage({viewport:{width:1440,height:1100}}), errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(baseUrl,{waitUntil:'networkidle'});
 await page.screenshot({path:path.join(root,'work/desktop.png'),fullPage:true});
 await page.locator('a[href="/examples/double-pendulum/"]').first().click();
 await page.waitForLoadState('networkidle');
 if(!(await page.locator('h1').innerText()).includes('Learn while')) throw Error('Pendulum route did not render');
 const pendulumDuration=process.env.PENDULUM_DURATION||'15';await page.locator('#reservoirSize').selectOption(process.env.PENDULUM_FULL_ONLY==='1'?'200':'100');await page.locator('#duration').selectOption(pendulumDuration);if(process.env.PENDULUM_SCENARIO)await page.locator('#physicsChange').selectOption(process.env.PENDULUM_SCENARIO);if(process.env.PENDULUM_SENSOR_NOISE==='1')await page.locator('#sensorNoise').check();if(process.env.PENDULUM_AUDIT==='1')await page.locator('#seedAuditToggle').check();await page.locator('#startPendulum').click();
 await page.waitForFunction(()=>document.getElementById('updateTime').textContent!=='—',{timeout:10000});
 await page.waitForFunction(changeAt=>parseFloat(document.getElementById('pendulumClock').textContent)>=changeAt,+pendulumDuration*.4+.2,{timeout:(+pendulumDuration*.4+6)*1000});
 if((await page.locator('#pendulumStatus').innerText()).startsWith('Experiment failed')) throw Error(await page.locator('#pendulumStatus').innerText());
 await page.screenshot({path:path.join(root,'work/pendulum.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});const pendulumOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(pendulumOverflow)throw Error('Pendulum mobile horizontal overflow');await page.screenshot({path:path.join(root,'work/pendulum-mobile.png'),fullPage:true});await page.setViewportSize({width:1440,height:1100});
 if(process.env.PENDULUM_FULL_ONLY==='1'){
  const completedStatus=process.env.PENDULUM_AUDIT==='1'?'Experiment and model-seed check complete':'Experiment complete';
  await page.waitForFunction(status=>document.getElementById('pendulumStatus').textContent===status,completedStatus,{timeout:process.env.PENDULUM_AUDIT==='1'?120000:(+pendulumDuration*.7+6)*1000});
  await page.screenshot({path:path.join(root,'work/pendulum-result.png'),fullPage:true});
  const metrics=await page.evaluate(()=>({adaptive:document.getElementById('forecastError').textContent,frozen:document.getElementById('frozenError').textContent,update:document.getElementById('updateTime').textContent,missed:document.getElementById('deadlinesMissed').textContent,recovery:document.getElementById('recoveryTime').textContent,achieved:document.getElementById('achievedRate').textContent,lateTicks:document.getElementById('lateTicks').textContent,dropped:document.getElementById('droppedSamples').textContent,drift:document.getElementById('clockDrift').textContent,finding:document.getElementById('runFinding').textContent,audit:document.getElementById('auditSummary').textContent,auditRows:document.querySelectorAll('#auditRows tr').length}));
  if(!/Hz$/.test(metrics.achieved)||!/ms$/.test(metrics.drift)||!metrics.finding.includes('Engineered'))throw Error('Timing audit or stress-test disclosure missing from final result');
  if(process.env.PENDULUM_SENSOR_NOISE==='1'&&!metrics.finding.includes('noisy'))throw Error('Sensor challenge disclosure missing from final result');
  if(process.env.PENDULUM_AUDIT==='1'&&(metrics.auditRows!==5||!metrics.audit.includes('reservoir initializations')))throw Error('Model-seed repeatability results missing');
  await page.setViewportSize({width:390,height:844});if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Completed pendulum result has mobile horizontal overflow');await page.screenshot({path:path.join(root,'work/pendulum-result-mobile.png'),fullPage:true});
  console.log(JSON.stringify({errors,overflow:false,pendulum:metrics}));await browser.close();if(errors.length)process.exitCode=1;return;
 }
 await page.locator('#stopPendulum').click();
 await page.goto(baseUrl,{waitUntil:'networkidle'});
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:path.join(root,'work/mobile.png'),fullPage:true});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 if(overflow) throw Error('Mobile horizontal overflow');
 // Check cancellation resets the form even during synchronous reservoir computation.
 await page.locator('#reservoirSize').selectOption('100');
 await page.locator('#run').click();
 await page.waitForFunction(()=>{const text=document.getElementById('status').textContent;return text.startsWith('ESN:')||text.startsWith('GRU:')||text.startsWith('Experiment failed');},{timeout:20000});
 if((await page.locator('#status').innerText()).startsWith('Experiment failed')) throw Error(await page.locator('#status').innerText());
 await page.locator('#stop').click();
 if(await page.locator('#run').isDisabled()) throw Error('Stop did not restore form');
 if(process.env.BROWSER_SMOKE_ONLY==='1') {
  console.log(JSON.stringify({errors,overflow,routes:['/','/examples/double-pendulum/'],worker:'started',cancellation:'passed'}));
  await browser.close();if(errors.length)process.exitCode=1;return;
 }
 await page.locator('#run').click();
 await page.waitForFunction(()=>document.getElementById('status').textContent==='Experiment complete'||document.getElementById('status').textContent.startsWith('Experiment failed'),{timeout:90000});
 const text=await page.locator('#status').innerText();if(text!=='Experiment complete') throw Error(text);
 const result=await page.evaluate(()=>JSON.parse(JSON.stringify(globalThis.eval('result'))));
 fs.writeFileSync(path.join(root,'work/browser-match.json'),JSON.stringify(result,null,2));
 if(result.gru.equalBudget&&result.gru.equalBudget.trainingMs>result.esn.trainingMs) throw Error('Selected GRU exceeded budget');
 if(result.esn.test.count!==120) throw Error('Test count incorrect');
 await page.locator('#matrixModel').selectOption('equal');
 await page.screenshot({path:path.join(root,'work/mobile-result.png'),fullPage:true});
 await page.setViewportSize({width:1440,height:1100});
 await page.screenshot({path:path.join(root,'work/desktop-result.png'),fullPage:true});
 const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#downloadJson').click()]);
 await download.saveAs(path.join(root,'work/download-check.json'));
 console.log(JSON.stringify({errors,overflow,esnMs:result.esn.trainingMs,esnTest:result.esn.test.accuracy,gruEqual:result.gru.equalBudget?.test?.accuracy,updates:result.gru.equalBudget?.updates,download:download.suggestedFilename()}));
 await browser.close();if(errors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
