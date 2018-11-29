from google.protobuf import text_format
from onnx_tf.frontend import tensorflow_graph_to_onnx_model
from onnx_tf.common import get_output_node_names
import tensorflow as tf
import onnxmltools
import sys

OPSET = 7

try:
    model_file = sys.argv[1]
except KeyError:
    raise ValueError("Usage: python tf_to_json.py <file.pbtxt>")

with tf.gfile.GFile(model_file, "r") as f:
    graph_def = text_format.Parse(f.read(),
                                  tf.GraphDef())

output = get_output_node_names(graph_def)
onnx_model = tensorflow_graph_to_onnx_model(graph_def,
                                            output,
                                            ignore_unimplemented=True,
                                            opset=OPSET)
onnxmltools.utils.save_text(onnx_model, model_file.replace('.pbtxt', '.json'))
