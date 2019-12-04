import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';
import * as numeric from 'numeric';

const N = 30;
const D = 3;

function renderPrediction(ctx, xs) {
  const xsOriginal = xs.dataSync();
  const c1Data = [];
  const c2Data = [];
  const c3Data = [];
  for (let i = 0; i < xsOriginal.length; i += 2) {
    const n = i/2;
    if (n < N) {
      c1Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
    } else if (N <= n && n < 2*N) {
      c2Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
    } else {
      c3Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
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
        },
        {
          label: 'Class 3',
          data: c3Data,
          backgroundColor: 'rgba(128, 256, 0, 0.3)',
          pointStyle: 'rectRot'
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Data Points'
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

async function pca(xs: tf.Tensor, nComponents: number) {
  const batch = xs.shape[0];
  const meanValues = xs.mean(0);
  const sub = tf.sub(xs, meanValues);
  const covariance = tf.matMul(sub.transpose(), sub);
  const covarianceData = tf.util.toNestedArray([D, D], covariance.dataSync()) as number[][];
  const eig = numeric.eig(covarianceData);
  const eigenvectors = tf.tensor(eig.E.x).slice([0, 0], [-1, nComponents]);
  return tf.matMul(sub, eigenvectors);
}

function variance(xs: tf.Tensor) {
  const v = xs.sub(xs.mean(0)).pow(2).mean();
  console.log(v.dataSync());
}

async function main() {
  const c1 = tf.randomNormal([N, D]).add([1.0, 0.0, 0.0]);
  const c2 = tf.randomNormal([N, D]).add([-1.0, 0.0, 0.0]);
  const c3 = tf.randomNormal([N, D]).add([0.0, 1.0, 1.0]);

  const xs = c1.concat(c2).concat(c3);

  const xs1 = xs.gather([0, 1], 1);
  console.log("Variance of xs1");
  variance(xs1);
  const xs2 = xs.gather([0, 2], 1);
  console.log("Variance of xs2");
  variance(xs2);
  const xs3 = xs.gather([0, 2], 1);
  console.log("Variance of xs3");
  variance(xs3);
  const pcaXs = await pca(xs, 2);
  console.log("Variance of pca");
  variance(pcaXs);

  const ctx1 = document.getElementById('0-1-dim');
  renderPrediction(ctx1, xs1);
  const ctx2 = document.getElementById('0-2-dim');
  renderPrediction(ctx2, xs2);
  const ctx3 = document.getElementById('1-2-dim');
  renderPrediction(ctx3, xs3);
  const ctx4 = document.getElementById('pca');
  renderPrediction(ctx4, pcaXs);
}

main();