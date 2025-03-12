# chapter 4 -> getting started: with neural networks classfication and regression

# listing 4.1 loading the imdb dataset
from tensorflow.keras.datasets import imdb
(train_data, test_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print