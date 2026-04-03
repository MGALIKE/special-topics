import { Worker, isMainThread, parentPort } from 'worker_threads';
import { fileURLToPath } from 'url';

if(isMainThread){
  const w = new Worker(fileURLToPath(import.meta.url));
  const m = new WebAssembly.Memory({initial:1, maximum:1, shared:true});
  w.postMessage(m);
} else {
  parentPort.on('message', m => console.log('Instance WebAssembly.Memory?', m instanceof WebAssembly.Memory));
}
