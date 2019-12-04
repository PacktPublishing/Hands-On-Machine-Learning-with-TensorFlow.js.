import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';

function renderCurve(ctx, xs, ys) {
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
          label: 'Cos',
          data: ys.dataSync(),
          lineTension: 0.0,
          pointBorderColor: 'rgba(0, 0, 0, 1.0)',
          borderColor: 'rgba(0, 128, 256, 0.3)',
          backgroundColor: 'rgba(0, 128, 256, 0.3)',
          fill: false
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Cosine Curve'
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

function renderFrequency(ctx, xs, ys) {
  const fixedX = [];
  const ret = xs.dataSync().forEach(x => {
    fixedX.push(x.toFixed(3));
  });
  const myChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: fixedX,
      datasets: [
        {
          label: 'frequency',
          data: ys.dataSync(),
          lineTension: 0.0,
          pointBorderColor: 'rgba(0, 0, 0, 1.0)',
          borderColor: 'rgba(256, 128, 256, 0.3)',
          backgroundColor: 'rgba(256, 128, 256, 0.3)',
          fill: false
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Frequencies'
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

const pi = tf.scalar(2.0 * Math.PI);
const xs = tf.mul(pi, tf.range(-1.5, 1.5, 0.05));

const xs2 = tf.mul(pi, tf.range(-1.5, 1.5, 0.05)).mul(2.0);
const ys = tf.cos(xs).add(tf.cos(xs2).mul(0.2));

const transformed = tf.fft(tf.complex(ys, ys.zerosLike()));
const real = tf.real(transformed);
const imag = tf.imag(transformed);

const norm = tf.sqrt(real.square().add(imag.square()))
const frequency = tf.range(0, xs.shape[0]).mul(pi).div(xs.shape[0]);

const inversed = tf.ifft(transformed);
const ctx1 = document.getElementById('curve1');
renderCurve(ctx1, xs, ys);
const ctx2 = document.getElementById('frequency');
renderFrequency(ctx2, frequency, norm);
const ctx3 = document.getElementById('curve2');
renderCurve(ctx3, xs, tf.real(inversed));
