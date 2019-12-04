import * as tf from '@tensorflow/tfjs';

window.onload = async (e) => {
  const element = document.getElementById('backend_name');
  await tf.ready();
  const backendName = tf.getBackend();
  console.log(backendName);
  element.innerText = backendName;
}