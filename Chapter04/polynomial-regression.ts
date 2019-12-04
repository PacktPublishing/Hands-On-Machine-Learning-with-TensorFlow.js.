import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';

function renderLoss(ctx, iterations, losses) {
  const myChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: iterations,
      datasets: [
        {
          label: 'Loss',
          data: losses,
          lineTension: 0.0
        }
      ]
    },
    options: {
    }
  });
}

function renderPrediction(ctx, xs, ys, preds) {
  const fixedX = [];
  const ret = xs.dataSync().forEach(x => {
    fixedX.push(x.toFixed(3));
  });
  const myChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: fixedX,
      datasets: [
        {
          label: 'Sin',
          data: ys.dataSync(),
          lineTension: 0.0,
          pointBorderColor: 'rgba(0, 0, 0, 1.0)',
          borderColor: 'rgba(0, 128, 256, 0.3)',
          backgroundColor: 'rgba(0, 128, 256, 0.3)',
          fill: false
        },
        {
          label: 'Prediction',
          data: preds,
          lineTension: 0.0,
          pointBorderColor: 'rgba(0, 0, 0, 1.0)',
          borderColor: 'rgba(0, 200, 0, 0.3)',
          backgroundColor: 'rgba(0, 200, 0, 0.3)',
          fill: false
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Sin Curve with Noise'
      },
      scales: {
        xAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: 'X'
            },
            ticks: {
              // Include a dollar sign in the ticks
              // callback: function(value, index, values) {
              //   console.log(value);
              //   return parseFloat(value).toFixed(2);
              // },
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
const pi = tf.scalar(2.0 * Math.PI);
const xs = tf.mul(pi, tf.range(-0.5, 0.5, 0.01));
const noise = tf.randomNormal([xs.size]).mul(0.05);
const ys = tf.sin(xs).add(noise);

const w0 = tf.scalar(Math.random() - 0.5).variable();
const w1 = tf.scalar(Math.random() - 0.5).variable();
const w2 = tf.scalar(Math.random() - 0.5).variable();
const w3 = tf.scalar(Math.random() - 0.5).variable();
const w4 = tf.scalar(Math.random() - 0.5).variable();
const w5 = tf.scalar(Math.random() - 0.5).variable();
const w6 = tf.scalar(Math.random() - 0.5).variable();

// f(x) = g*x^6 + f*x^5 + e*x^4 + d*x^3 + c*x^2 + b*x + a
const f_x = x => {
  // return w6.mul(x).mul(x).mul(x).mul(x).mul(x).mul(x)
  // .add(w5.mul(x).mul(x).mul(x).mul(x).mul(x))
  // .add(w4.mul(x).mul(x).mul(x).mul(x))
  // return (w3.mul(x).mul(x).mul(x))
  return (w2.mul(x).mul(x))
  .add(w1.mul(x))
  .add(w0);
}
const loss = (pred, label) => pred.sub(label).square().mean();

const learningRate = 0.03;
const optimizer = tf.train.adam(learningRate);

const iterations = [];
const losses = [];
// Train the model.
for (let i = 0; i < 100; i++) {
  const l = optimizer.minimize(() => loss(f_x(xs), ys), true);
  iterations.push(i);
  losses.push(l.dataSync());
}

// Make predictions.

const preds = f_x(xs).dataSync();

const ctx1 = document.getElementById('prediction');
renderPrediction(ctx1, xs, ys, preds);
const ctx2 = document.getElementById('loss');
renderLoss(ctx2, iterations, losses);
