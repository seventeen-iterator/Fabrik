from keras.models import Sequential
from keras.layers import LSTM
import keras.backend as K
import tensorflow as tf

model = Sequential()
model.add(LSTM(512,
               return_sequences=True,
               input_shape=(10,13),
               activation='tanh',
               kernel_initializer='he_uniform',
               recurrent_initializer='RandomNormal',
               bias_initializer='lecun_uniform',
               kernel_regularizer='l1',
               recurrent_regularizer='l2',
               dropout=.17,
               recurrent_dropout=.15))

sess = K.get_session()
tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), '.',
                     'LSTM.pbtxt', as_text=True)
