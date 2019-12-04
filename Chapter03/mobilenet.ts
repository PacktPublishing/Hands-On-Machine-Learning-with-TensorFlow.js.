import * as mobilenet from '@tensorflow-models/mobilenet';

async function loadAndPredict() {
  const img = document.getElementById('cat');  
  const model = await mobilenet.load();

  // Classify the image.
  const predictions = await model.classify(img);

  console.log('Predictions: ');
  console.log(predictions);

  // Display the prediction result.
  const preds = document.getElementById('predictions');
  preds.innerHTML = predictions.map((p) => {
    return p['className'];
  }).join('<br>');
}

loadAndPredict();