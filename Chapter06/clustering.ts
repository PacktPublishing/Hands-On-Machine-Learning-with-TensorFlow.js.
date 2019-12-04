import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';
import {KMeans} from './kmeans';

function renderElbow(ctx, xs: tf.Tensor) {
  const iterations = [];
  const losses = [];
  for (let k = 1; k < 8; k++) {
    const model = new KMeans(xs, k);
    let d: tf.Tensor;
    for (let i = 0; i < 5; i++) {
      d = model.update();
    }

    iterations.push(k);
    losses.push(d.dataSync());
  }

  console.log(losses);

  const myChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: iterations,
      datasets: [
        {
          label: 'Elbow Method',
          data: losses,
          lineTension: 0.0
        }
      ]
    },
    options: {
      scales: {
        xAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: 'K'
            }
          },
        ],
        yAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: 'SSE Loss'
            }
          }
        ]
    }
  });
}

function renderLoss(ctx, iterations, losses) {
  const myChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: iterations,
      datasets: [
        {
          label: 'SSE Loss',
          data: losses,
          lineTension: 0.0
        }
      ]
    },
    options: {
    }
  });
}

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

function renderPrediction(ctx, xs, centroids, clusterIndex) {
  const xsOriginal = xs.dataSync();
  const centroidsOriginal = centroids.dataSync();
  const clusterIndexOriginal = clusterIndex.dataSync();
  const c1Data = [];
  const c2Data = [];
  const c3Data = [];
  for (let i = 0; i < xsOriginal.length; i += 2) {
    if (clusterIndexOriginal[i/2] === 1) {
      c1Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
    } else if (clusterIndexOriginal[i/2] === 0) {
      c2Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
    } else {
      c3Data.push({x: xsOriginal[i], y: xsOriginal[i+1]});
    }
  }
  const centroidsData = [];
  for (let i = 0; i < centroidsOriginal.length; i+= 2) {
    centroidsData.push({x: centroidsOriginal[i], y: centroidsOriginal[i+1]});
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
        },
        {
          label: 'Centroid',
          data: centroidsData,
          backgroundColor: 'rgba(0, 0, 256, 0.5)',
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
const N = 30;

const c1 = tf.randomNormal([N, 2]).add([2.0, 1.0]);
const c2 = tf.randomNormal([N, 2]).add([-2.0, -1.0]);
const c3 = tf.randomNormal([N, 2]).add([-2.0, 2.0]);

const xs = c1.concat(c2).concat(c3);

const model = new KMeans(xs, 3);
const iterations = [];
const losses = [];
for (let i = 0; i < 10; i++) {
  const d = model.update();
  iterations.push(i);
  losses.push(d.dataSync());
}

const ctx1 = document.getElementById('original');
renderOriginal(ctx1, c1, c2, c3);
const ctx2 = document.getElementById('prediction');
renderPrediction(ctx2, xs, model.centroids, model.clusterAssignment);
const ctx3 = document.getElementById('loss');
renderLoss(ctx3, iterations, losses);
const ctx4 = document.getElementById('elbow');
renderElbow(ctx4, xs);
