import express, { Request, Response } from 'express';
import * as tf from '@tensorflow/tfjs';

const app = express();

app.get("/", (req: Request, res: Response) => {
    console.log(`Get request ${req}`);
    const t1 = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const t2 = tf.tensor2d([5, 6, 7, 8], [2, 2]);
    const result = t1.add(t2).toString();
    res.send(`Hello World! ${result}`);
});

app.listen(3000, () => {
    console.log("Example server listening on port 3000!");
});