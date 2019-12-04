import * as tf from '@tensorflow/tfjs';
import * as BackendWasm from 'tfjs-backend-wasm';
import {BackendWasmModule} from 'tfjs-backend-wasm';


window.onload = async (e) => {
  const element = document.getElementById('backend_name');
  tf.registerBackend('wasm', async () => {
    return new BackendWasm(new BackendWasmModule());
  }, 3 /*priority*/);
  await tf.setBackend("wasm");
  const backendName = tf.getBackend();
  console.log(backendName);
  element.innerText = backendName;

  const t1 = tf.tensor([1,2,3]);
  t1.print();
}
