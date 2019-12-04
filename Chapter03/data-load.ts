import * as tf from '@tensorflow/tfjs';

async function webcamLaunch() {
  const display = document.getElementById('display');
  const videoElement = document.createElement('video');
  display.appendChild(videoElement);
  videoElement.width = 500;
  videoElement.height = 500;
  const webcamIterator = await tf.data.webcam(videoElement);

  // img is a tensor showing the input webcam image.
  const img = await webcamIterator.capture();
  img.print();
}

webcamLaunch();