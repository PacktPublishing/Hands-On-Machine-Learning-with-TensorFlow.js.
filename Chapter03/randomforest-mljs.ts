import { RandomForestClassifier } from 'machinelearn/ensemble';
import { PCA } from 'machinelearn/decomposition';

async function trainAndPredict() {
  const pca = new PCA();
  const X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];

  const y = [
    0,
    1,
    1,
    0
  ];

  const model = new RandomForestClassifier();
  model.fit(X, y);

  const result = model.predict(X);
  console.log(result);

  const modelStr = JSON.stringify(model.toJSON());
  console.log(modelStr);

  const loadedModel = JSON.parse(modelStr);

  model.fromJSON(loadedModel);
}

trainAndPredict();