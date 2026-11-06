import assert from 'node:assert/strict'
import {test} from 'node:test'
import {advancePlayback,PLAYBACK_SECONDS,nearestFrame,decodeBasis,component,extent,projection,bestPlane} from '../web/run_monitor/plot.js'

test('one eight-second playback convention, independent of simulation time',()=>{
  assert.equal(PLAYBACK_SECONDS,8)
  assert.deepEqual(advancePlayback(0,4000,false),{fraction:.5,playing:true})
  assert.deepEqual(advancePlayback(0,8000,false),{fraction:1,playing:false})
  assert.deepEqual(advancePlayback(.75,4000,true),{fraction:.25,playing:true})
  assert.equal(nearestFrame([0,1,9],.5),1)
  assert.equal(nearestFrame([0,100,900],.5),1)
  assert.equal(nearestFrame([0,1,9],1),2)
})
test('missing data, complex phase and 64-bit basis identity stay honest',()=>{
  assert.equal(component(null,1),null)
  assert.equal(component(0,0,'phase'),null)
  assert.equal(component(0,-2,'phase'),-Math.PI/2)
  assert.deepEqual(extent([null,NaN,-3,-1]),[-3,-1])
  const bits=decodeBasis(String(2n**60n+1n),61)
  assert.equal(bits[0],1);assert.equal(bits.at(-1),1)
  assert.equal(bits.reduce((a,b)=>a+b),2)
})
test('array geometry keeps equal aspect and chooses a noncollapsed plane',()=>{
  const sites=[{pos:[0,0,0]},{pos:[6,0,0]},{pos:[0,6,0]}]
  const p=projection(sites,'xy',600,300).positions
  assert.equal(Math.abs(p[1][0]-p[0][0]),Math.abs(p[2][1]-p[0][1]))
  assert.equal(bestPlane([{pos:[0,0,0]},{pos:[0,3,4]}]),'yz')
  const line=projection(sites.slice(0,2),'xy',600,300).positions
  assert.equal(line[0][1],150);assert.equal(line[1][1],150)
})
