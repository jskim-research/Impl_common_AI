"""기본 RNN 구조 구현

Many-to-one, many-to-many, stacked many-to-one, stacked many-to-many
reference: https://yjjo.tistory.com/32
must-read NLP paper: https://github.com/mhagiwara/100-nlp-papers
"""
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import layers


# input shape = (batch #, seq #, feature #)
def many_to_one():
    x = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]], dtype=np.float32)  # (3, 3, 1)
    y = np.array([[4], [5], [6]], dtype=np.float32)

    # make model
    layer_input = keras.Input(shape=(3, 1), name='input')
    layer_rnn = layers.SimpleRNN(100, name='RNN')(layer_input)
    layer_output = layers.Dense(1, name='output')(layer_rnn)

    model = keras.Model(layer_input, layer_output)
    print(model.summary())

    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=100, batch_size=1, verbose=0)
    print("predict:", model.predict(x))
    print("after 4, 5, 6: ", model.predict([[[4], [5], [6]]]))


def many_to_many():
    x = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]], dtype=np.float32)
    y = np.array([[[2], [3], [4]], [[3], [4], [5]], [[4], [5], [6]]], dtype=np.float32)

    # make model
    layer_input = keras.Input(shape=(3, 1), name='input')
    layer_rnn = layers.SimpleRNN(100, return_sequences=True, name='RNN')(layer_input)
    layer_output = layers.TimeDistributed(layers.Dense(1, name='output'))(layer_rnn)

    model = keras.Model(layer_input, layer_output)
    print(model.summary())

    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=100, batch_size=1, verbose=0)
    print("predict:", model.predict(x))
    print("after 4, 5, 6: ", model.predict([[[4], [5], [6]]]))


def stacked_many_to_one():
    x = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]], dtype=np.float32)  # (3, 3, 1)
    y = np.array([[4], [5], [6]], dtype=np.float32)

    # make model
    layer_input = keras.Input(shape=(3, 1), name='input')
    layer_rnn0 = layers.SimpleRNN(100, return_sequences=True, name='RNN_cell_0')(layer_input)
    layer_rnn1 = layers.SimpleRNN(100, name="RNN_cell_1")(layer_rnn0)
    layer_output = layers.Dense(1, name='output')(layer_rnn1)

    model = keras.Model(layer_input, layer_output)
    print(model.summary())
    #
    # model.compile(loss='mse', optimizer='adam')
    # model.fit(x, y, epochs=100, batch_size=1, verbose=0)
    # print("predict:", model.predict(x))
    # print("after 4, 5, 6: ", model.predict([[[4], [5], [6]]]))


def stacked_many_to_many():
    x = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]], dtype=np.float32)
    y = np.array([[[2], [3], [4]], [[3], [4], [5]], [[4], [5], [6]]], dtype=np.float32)

    # make model
    layer_input = keras.Input(shape=(3, 1), name='input')
    layer_rnn0 = layers.SimpleRNN(100, return_sequences=True, name='RNN_cell_0')(layer_input)
    layer_rnn1 = layers.SimpleRNN(100, return_sequences=True, name='RNN_cell_1')(layer_rnn0)
    layer_output = layers.TimeDistributed(layers.Dense(1, name='output'))(layer_rnn1)

    model = keras.Model(layer_input, layer_output)
    print(model.summary())

    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=100, batch_size=1, verbose=0)
    print("predict:", model.predict(x))
    print("after 4, 5, 6: ", model.predict([[[4], [5], [6]]]))


if __name__ == "__main__":
    # many_to_one()
    many_to_many()
    # stacked_many_to_one()
    # stacked_many_to_many()
