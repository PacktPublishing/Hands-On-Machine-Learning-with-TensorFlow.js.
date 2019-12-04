import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';
import {loadData, loadMetadataTemplate} from './imdb';
import {writeEmbeddingMatrixAndLabels} from './embedding_util';

function buildModel(numWords: number, maxLen: number, embeddingSize: number) {
  const model = tf.sequential();
  model.add(tf.layers.embedding({
    inputDim: numWords,
    outputDim: embeddingSize,
    inputLength: maxLen
  }));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  return model;
}

async function main() {
  const numWords = 10000;
  const maxLen = 100;
  const embeddingSize = 8;
  console.log('Loading data...');
  const {xTrain, xTest, yTrain, yTest} = await loadData(numWords, maxLen);
  console.log(xTrain);

  const model = buildModel(numWords, maxLen, embeddingSize);

  model.compile({
    loss: 'binaryCrossentropy',
    optimizer: 'adam',
    metrics: ['acc']
  });
  model.summary();

  console.log('Training model...');
  await model.fit(xTrain, yTrain, {
    epochs: 5,
    batchSize: 64,
    validationSplit: 0.2
  });

  console.log('Evaluating model...');
  const [testLoss, testAcc] = model.evaluate(xTest, yTest, {batchSize: 64});
  console.log(`Evaluation loss: ${(await testLoss.data())[0].toFixed(4)}`);
  console.log(`Evaluation accuracy: ${(await testAcc.data())[0].toFixed(4)}`);

  const metadata = await loadMetadataTemplate();

  await writeEmbeddingMatrixAndLabels(
    model, "path", metadata.word_index, metadata.index_from);
}

main();