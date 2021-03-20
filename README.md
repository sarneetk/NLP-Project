# NLP Project - Sentiment Analysis

Project to do sentiment analysis on customer reviews (IMDB) and compare performance of various models.

### Contributors
1. Archana Balachandran
2. Sarneet Kaur

### Models
We implemented sentiment analysis on keras IMDB reviews dataset using following AL algorithms:
1. CNN
2. LSTM
3. LSTM bi-directional
4. LSTM with CNN
5. LSTM with dropout
6. Transformer 

We also experimented with attention in LSTM-bidirectional model and multiple hidden-layers in CNN/LSTM.


### How to run
Ensure you have installed all the dependencies:

```
pip install tensorflow
pip install keras
```

if you want to create graphs:

```
pip install matplotlib
pip install pydot
pip install pydotplus
pip install graphviz
```

To run a model, simply run:

```
python LSTM.py
python LSTM_bidirectional.py
python transformer.py
```