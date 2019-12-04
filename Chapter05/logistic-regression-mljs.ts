import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';
import {LogisticRegression} from 'machinelearn/linear_model';

function renderOriginal(ctx, c1, c2) {
  const c1Original = c1.dataSync();
  const c1Data = [];
  for (let i = 0; i < c1Original.length; i += 2) {
    c1Data.push({x: c1Original[i], y: c1Original[i+1]});
  }
  const c2Original = c2.dataSync();
  const c2Data = [];
  for (let i = 0; i < c2Original.length; i += 2) {
    c2Data.push({x: c2Original[i], y: c2Original[i+1]});
  }
  const myChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Class 1',
          data: c1Data,
          // lineTension: 0.0,
          // pointBorderColor: 'rgba(0, 0, 0, 1.0)',
          // borderColor: 'rgba(0, 128, 256, 0.3)',
          backgroundColor: 'rgba(0, 128, 256, 0.3)',
          pointStyle: 'rect'
          // fill: false
        },
        {
          label: 'Class 2',
          data: c2Data,
          backgroundColor: 'rgba(128, 0, 256, 0.3)',
          pointStyle: 'triangle'
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Original Clusters'
      },
      scales: {
        xAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: 'X'
            }
          },
        ],
        yAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: 'Y'
            }
          }
        ]
      }
    }
  });
}

function renderPrediction(ctx, xs, preds) {
  const xsOriginal = xs.dataSync();
  const predsOriginal = preds;
  console.log(predsOriginal);
  const c1Data = [];
  const c2Data = [];
  for (let i = 0; i < xsOriginal.length; i += 2) {
    if (predsOriginal[i/2] > 0.5) {
      c1Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
    } else {
      c2Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
    }
  }
  const myChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Class 1',
          data: c1Data,
          // lineTension: 0.0,
          // pointBorderColor: 'rgba(0, 0, 0, 1.0)',
          // borderColor: 'rgba(0, 128, 256, 0.3)',
          backgroundColor: 'rgba(0, 128, 256, 0.3)',
          pointStyle: 'rect'
          // fill: false
        },
        {
          label: 'Class 2',
          data: c2Data,
          backgroundColor: 'rgba(128, 0, 256, 0.3)',
          pointStyle: 'triangle'
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Prediction Result'
      },
      scales: {
        xAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: 'X'
            }
          },
        ],
        yAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: 'Y'
            }
          }
        ]
      }
    }
  });
}

// Fit a quadratic function by learning the coefficients a, b, c.
const N = 100;

const c1 = tf.randomNormal([N, 2]).add([1.0, 1]);
const c2 = tf.randomNormal([N, 2]).add([-1.0, -1.0]);
const l1 = tf.ones([N, 1]);
const l2 = tf.zeros([N, 1]);
// Make predictions.

const xs = c1.concat(c2);
const ys = l1.concat(l2);

const model = new LogisticRegression();

async function training() {
  const xsData = tf.util.toNestedArray([N * 2, 2], xs.dataSync()) as number[][];
  const ysData = tf.util.toNestedArray([N * 2], ys.dataSync()) as number[];
  console.log("start training...")
  await model.fit(xsData, ysData);

  console.log(model.toJSON());

  const ctx1 = document.getElementById('original');
  renderOriginal(ctx1, c1, c2);
  const ctx2 = document.getElementById('prediction');
  renderPrediction(ctx2, xs, model.predict(xsData));
}

training();


