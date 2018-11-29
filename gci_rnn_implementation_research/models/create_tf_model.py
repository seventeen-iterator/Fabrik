import tensorflow as tf


# This code snippet was honestly taken from Ahmet Taspinar's blog
# http://ataspinar.com/2018/07/05/building-recurrent-neural-networks-in-tensorflow/
# You can replace this code with another one from blog to generate other model files
def rnn_lstm_model(data, num_hidden, num_labels, activation):
    splitted_data = tf.unstack(data, axis=1)

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)

    outputs, current_state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)
    output = outputs[-1]

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))

    logit = tf.matmul(output, w_softmax) + b_softmax

    return logit


signal_length = 16
num_components = 9
num_hidden = 32
num_labels = 6

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=(None, signal_length, num_components))
    logit = rnn_lstm_model(x, num_hidden, num_labels, tf.nn.tanh)
    output = tf.nn.tanh(logit)

with tf.Session(graph=graph) as sess:
    tf.train.write_graph(graph.as_graph_def(add_shapes=True), '.',
                         'LSTM2.pbtxt', as_text=True)
