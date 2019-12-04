import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';

function renderGraph(ctx, xs, samples, target) {
  const myChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: xs,
      datasets: [
        {
          label: 'Sample with Noise',
          data: samples,
          borderColor: 'rgba(0, 0, 0, 1.0)',
          lineTension: 0.0,
          fill: false,
          showLine: false,
          pointRadius: 5
        },
        // {
        //   label: 'Target',
        //   data: target,
        //   borderColor: 'rgba(0, 100, 256, 0.7)',
        //   lineTension: 0.0,
        //   fill: false
        // },
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Simple Sin Curve'
      },
      scales: {
        xAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: 'X'
            }
          }
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

const xs = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const noise = tf.randomNormal([xs.size]).mul(0.2);
const ys = tf.sin(xs.mul(Math.PI / 5.0)).add(noise);

const ctx = document.getElementById('line');
renderGraph(ctx, xs.dataSync(), ys.dataSync(), null);