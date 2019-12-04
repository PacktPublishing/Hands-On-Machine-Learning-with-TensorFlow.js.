import * as tf from '@tensorflow/tfjs';
import {Chart} from 'chart.js';

function renderChart(ctx, xs, values) {
  const a1Values = values.map(v => v[0]);
  const a2Values = values.map(v => v[1]);
  const myChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: xs,
      datasets: [
        {
          label: 'Action 1 Value',
          data: a1Values,
          lineTension: 0.0,
          pointBorderColor: 'rgba(0, 0, 0, 1.0)',
          borderColor: 'rgba(0, 128, 256, 0.3)',
          backgroundColor: 'rgba(0, 128, 256, 0.3)',
          fill: false
        },
        {
          label: 'Action 2 Value',
          data: a2Values,
          lineTension: 0.0,
          pointBorderColor: 'rgba(0, 0, 0, 1.0)',
          borderColor: 'rgba(256, 128, 0, 0.3)',
          backgroundColor: 'rgba(256, 128, 0, 0.3)',
          fill: false
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Action Value'
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

class Environment {
  private states = [0, 1, 2, 3];
  private actions = [
    [2, 1],
    [0, 3],
    [3, 0],
    [3, 3], // End state
  ];
  // Reward is decided based on the current state and action an agent takes.
  private rewards = [
    [0, 1],
    [-1, 1],
    [50, -100],
    [0, 0],
  ];

  private currentState: number;
  constructor() {
    this.currentState = 0;
  }

  getCurrentState(): number {
    return this.currentState;
  }

  getStates(): number[] {
    return this.states;
  }

  getNumStates(): number {
    return this.states.length;
  }

  getNumActions(): number {
    return this.actions[0].length;
  }

  isEnd(): boolean {
    if (this.currentState === 3) {
      return true;
    } else {
      return false;
    }
  }

  update(action: number): number {
    const reward = this.rewards[this.currentState][action];
    this.currentState = this.actions[this.currentState][action];
    return reward;
  }

  reset() {
    this.currentState = 0;
  }
}

function policy() {
  if (Math.random() < 0.1) {
    return 0;
  } else {
    return 1;
  }
}

async function qlearning() {
  const episodes = [];
  for (let i = 0; i < 1000; i++) {
    episodes.push(i);
  }
  const env = new Environment();
  let actionValue = tf.fill([env.getNumStates(), env.getNumActions()], 10);
  const alpha = 0.01;
  const discount = 0.8;

  const s0Values = [];
  const s2Values = [];
  const xs = [];
  actionValue.print();
  for (let i of episodes) {
    let isEnd = false;
    if (i % 10 === 0) {
      xs.push(i);
      const rawActionValue = tf.util.toNestedArray([env.getNumStates(), env.getNumActions()], actionValue.dataSync());
      s0Values.push(rawActionValue[0]);
      s2Values.push(rawActionValue[2]);
    }
    while (!isEnd) {
      const action = policy();
      const prevState = env.getCurrentState();
      const reward = env.update(action);
      const currentState = env.getCurrentState();
      const array = new Float32Array(env.getNumStates() * env.getNumActions());
      array.fill(1.0);
      const buffer = tf.buffer([env.getNumStates(), env.getNumActions()], 'float32', array);
      const maxValue = tf.util.toNestedArray([env.getNumStates(), 1], actionValue.max(1).dataSync())[currentState];
      buffer.set(reward + discount * maxValue, prevState, action);

      actionValue = tf.mul((1 - alpha), actionValue).add(tf.mul(alpha, buffer.toTensor()));
      isEnd = env.isEnd();
    }
    env.reset();
  }

  actionValue.print();
  const ctx = document.getElementById('chart');
  renderChart(ctx, xs, s2Values);
}

qlearning();
