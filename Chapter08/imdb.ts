import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import {OOV_INDEX, padSequences} from './sequence_utils';

import * as request from 'request-promise-native';

/**
 * Load IMDB data features from a local file.
 *
 * @param {string} filePath Data file on local filesystem.
 * @param {string} numWords Number of words in the vocabulary. Word indices
 *   that exceed this limit will be marked as `OOV_INDEX`.
 * @param {string} maxLen Length of each sequence. Longer sequences will be
 *   pre-truncated; shorter ones will be pre-padded.
 * @param {string} multihot Whether to use multi-hot encoding of the words.
 *   Default: `false`.
 * @return {tf.Tensor} If `multihot` is `false` (default), the dataset
 *   represented as a 2D `tf.Tensor` of shape `[numExamples, maxLen]` and
 *   dtype `int32`. Else, the dataset represented as a 2D `tf.Tensor` of
 *   shape `[numExamples, numWords]` and dtype `float32`.
 */
async function loadFeatures(filePath, numWords, maxLen, multihot = false) {
  const result = await request.get(filePath);
  const sequences = JSON.parse(result);

  // Get some sequence length stats.
  let minLength = Infinity;
  let maxLength = -Infinity;
  sequences.forEach(seq => {
    const length = seq.length;
    if (length < minLength) {
      minLength = length;
    }
    if (length > maxLength) {
      maxLength = length;
    }
  });
  console.log(`Sequence length: min = ${minLength}; max = ${maxLength}, samples = ${sequences.length}`);

  if (multihot) {
    // If requested by the arg, encode the sequences as multi-hot
    // vectors.
    const buffer = tf.buffer([sequences.length, numWords]);
    sequences.forEach((seq, i) => {
      seq.forEach(wordIndex => {
        if (wordIndex !== OOV_INDEX) {
          buffer.set(1, i, wordIndex);
        }
      });
    });
    return buffer.toTensor();
  } else {
    const paddedSequences =
        padSequences(sequences, maxLen, 'pre', 'pre');
    return tf.tensor2d(
        paddedSequences, [paddedSequences.length, maxLen], 'int32');
  }
}

/**
 * Load IMDB targets from a file.
 *
 * @param {string} filePath Path to the binary targets file.
 * @return {tf.Tensor} The targets as `tf.Tensor` of shape `[numExamples, 1]`
 *   and dtype `float32`. It has 0 or 1 values.
 */
async function loadTargets(filePath) {
  const result = await request.get(filePath);
  const ys = JSON.parse(result);

  let numPositive = 0;
  let numNegative = 0;
  for (let i = 0; i < ys.length; ++i) {
    const y = ys[i]
    if (y === 1) {
      numPositive++;
    } else {
      numNegative++;
    }
  }

  console.log(
      `Loaded ${numPositive} positive examples and ` +
      `${numNegative} negative examples.`);
  return tf.tensor2d(ys, [ys.length, 1], 'float32');
}

/**
 * Load data by downloading and extracting files if necessary.
 *
 * @param {number} numWords Number of words to in the vocabulary.
 * @param {number} len Length of each sequence. Longer sequences will
 *   be pre-truncated and shorter ones will be pre-padded.
 * @return
 *   xTrain: Training data as a `tf.Tensor` of shape
 *     `[numExamples, len]` and `int32` dtype.
 *   yTrain: Targets for the training data, as a `tf.Tensor` of
 *     `[numExamples, 1]` and `float32` dtype. The values are 0 or 1.
 *   xTest: The same as `xTrain`, but for the test dataset.
 *   yTest: The same as `yTrain`, but for the test dataset.
 */
export async function loadData(numWords, len) {
  const dataDir = 'http://localhost:1234/ch7';
  const trainFeaturePath = `${dataDir}/imdb_train_data.bin.json`;
  const xTrain = await loadFeatures(trainFeaturePath, numWords, len);
  const testFeaturePath = `${dataDir}/imdb_test_data.bin.json`;
  const xTest = await loadFeatures(testFeaturePath, numWords, len);
  const trainTargetsPath = `${dataDir}/imdb_train_targets.bin.json`;
  const yTrain = await loadTargets(trainTargetsPath);
  const testTargetsPath = `${dataDir}/imdb_test_targets.bin.json`;
  const yTest = await loadTargets(testTargetsPath);

  tf.util.assert(
      xTrain.shape[0] === yTrain.shape[0],
      () => `Mismatch in number of examples between xTrain and yTrain`);
  tf.util.assert(
      xTest.shape[0] === yTest.shape[0],
      () => `Mismatch in number of examples between xTest and yTest`);
  return {xTrain, xTest, yTrain, yTest};
}

/**
 * Load a metadata template by downloading and extracting files if necessary.
 *
 * @return A JSON object that is the metadata template.
 */
export async function loadMetadataTemplate() {
  const filePath = 'http://localhost:1234/ch7/metadata.json';
  const result = await request.get(filePath);
  return JSON.parse(result);
}