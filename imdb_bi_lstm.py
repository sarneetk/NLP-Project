import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from util import f1_score, recall, precision, plot_graph
import matplotlib.pyplot as plt
plt.style.use('ggplot')

PLOT_GRAPH = False
PLOT_MODEL = False

max_features = 5000  # Only consider the top 5k words
maxlen = 500  # Only consider the first 500 words of each movie review

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer in a 64-dimensional vector
x = layers.Embedding(max_features, 64)(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

# load imdb data
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

# train and evaluate
model.compile("adam", "binary_crossentropy", metrics=["accuracy", f1_score, precision, recall])
history = model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=1, validation_data=(x_val, y_val))  # starts training

# Final evaluation of the model
scores = model.evaluate(x_val, y_val, verbose=0)
print(scores)
print("Accuracy:%.2f%%" % (scores[1] * 100))

if PLOT_GRAPH:
    plot_graph(history)

if PLOT_MODEL:
    img_file = 'tmp/lstm-bi.png'
    keras.utils.plot_model(model, to_file=img_file, show_shapes=True,  show_layer_names=True)