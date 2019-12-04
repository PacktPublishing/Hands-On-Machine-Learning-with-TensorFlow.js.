import * as tf from '@tensorflow/tfjs';

export class KMeans {
  k: number;
  dim: number;
  centroids: tf.Tensor;
  xs: tf.Tensor;
  clusterAssignment: tf.Tensor;

  constructor(xs: tf.Tensor, k: number) {
    this.dim = xs.shape[1];
    this.k = k;
    this.centroids = tf.randomNormal([this.k, this.dim]);
    this.xs = xs;
    this.closestCentroids();
  }

  closestCentroids() {
    const expandedXs = tf.expandDims(this.xs, 0);
    const expandedCentroids = tf.expandDims(this.centroids, 1);

    const d = expandedXs.sub(expandedCentroids).square().sum(2);

    this.clusterAssignment = d.argMin(0);
    return d.min(0).mean();
  }

  updateCentroids() {
    const centers = [];
    for (let i = 0; i < this.k; i++) {
      const cond = this.clusterAssignment.equal(i).dataSync();
      let index = [];
      for (let j = 0; j < cond.length; j++) {
        if (cond[j] == 1) {
          index.push(j);
        }
      }
      const cluster = tf.gather(this.xs, index);
      const center = cluster.mean(0);
      centers.push(center);
    }

    this.centroids = tf.concat(centers).reshape([this.k, this.dim]);
  }

  update() {
    this.updateCentroids();
    return this.closestCentroids();
  }
}