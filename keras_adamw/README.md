### Which optimizers to use?

```python
TensorFlow 1.14.0 + Keras 2.2.5 + 'keras'    >> optimizers_225.py   + utils_225.py
TensorFlow 1.14.0 + Keras 2.2.5 + 'tf.keras' >> optimizers_225tf.py + utils_225.py
TensorFlow 2.0.0  + Keras 2.3.0 + 'keras'    >> optimizers.py       + utils.py
TensorFlow 2.0.0  + Keras 2.3.0 + 'tf.keras' >> optimizers_v2.py    + utils.py
```

- `'keras'` --> using `keras` imports. _Ex_: `from keras.layers import Dense`
- `'tf.keras'` --> using `tensorflow.keras` imports. _Ex_: `from tensorflow.keras.layers import Dense`
- `TensorFlow 1.14.0` optimizers should also run with `TensorFlow 1.15.0` (and 1.13.0)
- `Keras 2.2.5` optimizers should also run with `Keras 2.2.4` (and 2.2.3)
