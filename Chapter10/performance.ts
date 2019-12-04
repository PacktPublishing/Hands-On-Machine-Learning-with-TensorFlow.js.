import * as tf from '@tensorflow/tfjs';

function benchmark(size: number) {
    for (let i = 0; i < 100; i++) {
        const t1 = tf.randomNormal([size, size]);
        const t2 = tf.randomNormal([size, size]);
        const result = t1.matMul(t2);
    }
}

async function runBenchmark() {
    console.log('size,kernelMs,wallMs');
    for (let s = 1; s < 100; s++) {
        const time = await tf.time(() => benchmark(s));
        console.log(`${s},${time.kernelMs},${time.wallMs}`);
    }
}

runBenchmark();
