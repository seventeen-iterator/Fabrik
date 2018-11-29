# ONNX Research
Attached Python code here has [separate requirements](requirements.txt). Install them via typing:
```
pip install -r requirements.txt
```

## Observations
ONNX-based method of implementation TF support for RNN deserves separate research. Unlike [mmdnn tools](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/tensorflow/README.md), ONNX file format [somewhat supports RNN](https://github.com/onnx/onnx-tensorflow/blob/master/doc/support_status.md). However, I'm unable to convert `.pbtxt` to `.json` because of either KeyError with "_output_shapes" or unsupported Acos operation, depending on value of `OPSET` variable (see [this file](pbtxt_to_json.py)).

## References
- [ONNX](https://github.com/onnx/onnx) - Open Neural Network EXchange.
- [ONNX-Tensorflow](https://github.com/onnx/onnx-tensorflow) a model converter between Tensorflow and ONNX.
- [ONNXMLTools](https://github.com/onnx/onnxmltools) for JSON support.
- [MMdnn](https://github.com/Microsoft/MMdnn) - another toolset for dealing with different model formats.
