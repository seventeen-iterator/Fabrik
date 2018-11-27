# [Fabrik] Research implementation method for RNN package support in tensorflow import
This research was made for Google Code-in task from [Cloud-CV](https://cloudcv.org).

Before running code install common requirements.
## Problem
As for now, Fabrik cannot export models for Tensorflow containing any recurrence. And it's pretty hard to implement, because there's no operations inside `.pbtxt` like "SimpleRNN" or "LSTM", so we have to find another way to add support for recurrence.
## Observations
If we look at several `.pbtxt` files, we may notice some facts:
- The nodes which belong to specific layer have prefix, which has the following pattern: `<type_of_layer>_<number>`. We can indicate type of layer by trying to fetch this pattern. Also different layers have different numbers.
- Some nodes have an initialized values, unlike Keras models. It's painful when it's orthogonal and we have to determine it.
- If we try to search dropout rates, we'll end up with nothing. But we'll find `1 - dropout` values. Nodes containing them named `keep_prob`.
- Regularizations `l1` includes abs operation and `l2` includes square operation. And we can notice respective nodes which can be used as regularization indicator for specific values.
There's also `observe_consts.py` to make observations easier.
## Suggestions
If we wish to keep the same functionality, there's `detect_rnn.py` that can extract some values from `.pbtxt` models. It just fetches needed nodes and read values from them. Initializer extraction logic is defined in `init_detector.py`.
You can play around with code and attached models typing:
```
python detect_rnn.py models/<desired_model_file>
```
You can compare result with [original settings](models/models_original_settings.md).

This seems to be pretty naive implementation of tensorflow import. But as long as there's no specified recurrent operations, it's the simplest of the only ways to enhance tensorflow import with recurrence support.

But if we don't care about saving already implemented functionality and are ready to rewrite the whole code of some Fabrik modules, we can:
- Generate a Python code for init model and export through syntax analysis of user code.
- Consider implementing `.onnx` file format, since they can be exported and imported by a wide range of other frameworks including Tensorflow. Following [this tutorial](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowExport.ipynb) we can come up with that we don't have to use `.pbtxt`.
## Notes
### `init_detector.py`
- Detecting truncating initializers is not working at the moment. Also detection of random normal is naive. There's `RandomNormal` and `TruncatedNormal` nodes inside model. Fetching them will make import more robust.
### `detect_rnn.py`
- There are some values which script cannot detect. I was unable to fetch any indicative nodes of constraints. For the rest of regularizers and "checkbox" values I didn't do anything.
- LSTM also have additional activation function. It's Sigmoid by default. But it isn't changing inside file if I set another function.
- Method still have no support for bidirectional wrapper
### Useful links
- [Netron](https://lutzroeder.github.io/netron/) - viewer of neural networks, alternative to observing exported models via editor. If you export any `.pbtxt` file here, you'll end up with a web of different operations. Might be useful if we decide to observe more patterns for import implementation.
- @Ram81 in his [PR](https://github.com/Cloud-CV/Fabrik/pull/314) did the similar work earlier on observing patterns, but the models he was working with had some another pattern.
## References
- [Tensorflow](https://www.tensorflow.org/) - deep learning framework, its graph export features and `.pbtxt` file format are the main reasons of doing such research
- [NumPy and SciPy](https://www.scipy.org/) - libraries for N-dimensional arrays and fundamental science calculations respectively. Mostly used in `init_detector.py` for using defining properties of distributions.
- [Keras](https://keras.io) - another deep learning library, used for generating some model files (among with exporting through local Fabrik setup).
- [ONNX](http://onnx.ai/) - open format to represent deep learning models. Good support for some frameworks that are still not supported in Fabrik, see [supported tools section](http://onnx.ai/supported-tools).
