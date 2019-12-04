import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

const model = tf.sequential({
  layers: [
    tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
    tf.layers.dense({units: 10, activation: 'softmax'}),
  ]
 });

const t1 = tf.randomNormal([100, 3]);

tfvis.show.layer({ name: 'Layer Inspection', tab: 'Layer' }, model.getLayer('first layer', 1));
tfvis.show.valuesDistribution({name: 'Values Distribution', tab: 'Model Inspection'}, t1);

model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

const data = tf.randomNormal([100, 784]);
const labels = tf.randomUniform([100, 10]);

async function showTrainingProgress() {
  const history = await model.fit(data, labels, {
     epochs: 5,
     batchSize: 32,
     callbacks: tfvis.show.fitCallbacks({name: 'Training Inspection', tab: 'Training'}, ['loss', 'acc']),
  });
  tfvis.show.history({name: 'Training History', tab: 'History'}, history, ['loss', 'acc']);
}

showTrainingProgress();