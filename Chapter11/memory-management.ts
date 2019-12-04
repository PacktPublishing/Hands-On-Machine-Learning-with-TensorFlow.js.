import * as tf from '@tensorflow/tfjs';

const t1 = tf.tensor1d([1, 2, 3, 4]);
const t2 = tf.tensor1d([1, 2, 3, 4]);

console.log("Before Dispose");
console.log(tf.memory());

const t3 = tf.tidy(() => {
    const result = t1.add(t2).square().log().neg();
    return result;
});

// const t3 = t1.add(t2).square().log().neg();
t3.dataSync();
t3.dispose();

console.log("After Dispose");
console.log(tf.memory());
