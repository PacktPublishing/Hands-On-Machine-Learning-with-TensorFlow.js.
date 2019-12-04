import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';
import * as numeric from 'numeric';
import {MnistData} from './mnist';

const N = 30;
const D = 784;

function renderOriginal(ctx, c1, c2, c3) {
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
  const c3Original = c3.dataSync();
  const c3Data = [];
  for (let i = 0; i < c3Original.length; i += 2) {
    c3Data.push({x: c3Original[i], y: c3Original[i+1]});
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

function renderPrediction(ctx, xs) {
  const xsOriginal = xs.dataSync();
  const c1Data = [];
  const c2Data = [];
  const c3Data = [];
  for (let i = 0; i < xsOriginal.length; i += 2) {
    if (i < N) {
      c1Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
    } else if (N <= i && i < 2*N) {
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

async function pca(xs: tf.Tensor, nComponents: number) {

  const batch = xs.shape[0];
  const tiled = tf.matMul(tf.ones([batch, 1]), xs.mean(0).reshape([1, -1]));
  console.log(tiled);
  const sub = tf.sub(xs, tiled);
  console.log(sub);
  const xsData = tf.util.toNestedArray([D, D], tf.matMul(sub.transpose(), sub).dataSync()) as number[][];
  console.log(xsData[0]);
  const eig = numeric.eig(xsData);
  // const eigenvectors = tf.tensor(eig.E.x).slice([0, 0], [-1, nComponents]);
  // return tf.matMul(sub, eigenvectors);
}

async function main() {
  const data = new MnistData();
  console.log('Loading data...');
  await data.load();

  const {xs, labels} = data.getTrainData();
  console.log(xs);
  const batch = xs.shape[0];

  const newXs = await pca(xs.reshape([batch, 784]).slice([0, 0], [100, D]), 2);
  // console.log(newXs);
}

main();