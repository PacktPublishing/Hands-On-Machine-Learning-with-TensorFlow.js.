import * as tf from '@tensorflow/tfjs';

function benchmark(size: number) {
    for (let i = 0; i < 10; i++) {
        const t1 = tf.randomNormal([size, size]);
        const t2 = tf.randomNormal([size, size]);
        t1.matMul(t2).dataSync();
    }
}

async function runBenchmark() {
    const backend = 'cpu';
    tf.setBackend(backend);
    let resultStr = "";
    resultStr += `size,kernelMs(${backend}),wallMs(${backend})\n`;
    for (let s = 10; s < 600; s += 10) {
        const time = await tf.time(() => benchmark(s));
        resultStr += `${s},${time.kernelMs},${time.wallMs}\n`;
    }
    console.log(resultStr);
}

runBenchmark();
