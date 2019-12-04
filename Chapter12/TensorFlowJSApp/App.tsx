import React from 'react';
import {Component} from 'react';
import { StyleSheet, Text, View } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import * as tfjsNative from '@tensorflow/tfjs-react-native';

export default class TensorFlowJSApp extends Component {
  private aStr: string;
  private bStr: string;
  private cStr: string;
  constructor(props) {
    super(props);
    const a = tf.tensor([[1, 2], [3, 4]]);
    const b = tf.tensor([[1, 2], [3, 4]]);

    const c = a.add(b);

    this.aStr = a.toString();
    this.bStr = b.toString();
    this.cStr = c.toString();
  }

  render() {
    return (
      <View style={styles.container}>
        <Text>Hello, TensorFlow.js! {'\n'}{'\n'}{this.aStr} + {'\n'}{this.bStr} = {'\n'}{this.cStr}</Text>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
