import * as tf from '@tensorflow/tfjs-node';

window.addEventListener("DOMContentLoaded", () => {
  const replaceText = (selector: string, text: string) => {
    const element = document.getElementById(selector);
    if (element) {
      element.innerText = text;
    }
  };

  for (const type of ["chrome", "node", "electron"]) {
    replaceText(`${type}-version`, (process.versions as any)[type]);
  }

  const a = tf.tensor([[1, 2], [3, 4]]);
  const b = tf.tensor([[1, 2], [3, 4]]);
  const c = a.add(b);
  replaceText('tensor-a', a.toString());
  replaceText('tensor-b', b.toString());
  replaceText('tensor-c', c.toString());
});
