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
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_test), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# train and evaluate
model.compile("adam", "binary_crossentropy", metrics=["accuracy", f1_score, precision, recall])
history = model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=1, validation_data=(x_test, y_test))  # starts training


# Evaluation of the model with training data
scores_train = model.evaluate(x_train, y_train, verbose=0)
print("Training Data: ")
print("Accuracy: %.2f%%, F_1Score: %.2f%% , Precision: %.2f%%, Recall: %.2f%% " % (scores_train[1]*100, scores_train[2]*100,
                                                                                   scores_train[3]*100, scores_train[4]*100))

# Evaluation of the model with test data
scores = model.evaluate(x_test, y_test, verbose=0)
print("Test Data:")
print("Accuracy: %.2f%%, F_1Score: %.2f%% , Precision: %.2f%%, Recall: %.2f%%" % (scores[1] * 100, scores[2] * 100,
                                                                                 scores[3] * 100, scores[4] * 100))

if PLOT_GRAPH:
    plot_graph(history)

if PLOT_MODEL:
    img_file = 'tmp/lstm-bi.png'
    keras.utils.plot_model(model, to_file=img_file, show_shapes=True,  show_layer_names=True)