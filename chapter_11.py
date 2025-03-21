# chapter 11 two approaches for representing groups of words: sets and sequences

# one time execution to setup the files for train, val , test

# import os, pathlib, shutil, random

# base_dir = pathlib.Path("aclImdb")
# val_dir = base_dir / "val"
# train_dir = base_dir / "train"

# for category in ("neg", "pos"):
#     os.makedirs(val_dir / category)
#     files = os.listdir(train_dir / category)
#     random.Random(1337).shuffle(files)
#     num_val_samples = int(0.2 * len(files))
#     val_files = files[-num_val_samples:]
#     for fname in val_files:
#         shutil.move(train_dir / category / fname, val_dir / category / fname)


from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)

val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", 
)

# listing 11.2 displaying the shapes and dtypes of the first batch

for inputs, targets in train_ds:
    print("inputs shape: ", inputs.shape)
    print("inputs dtype: ", inputs.dtype)
    print("targets shape: ", targets.shape)
    print("targets dtype: ", targets.dtype)
    print("inputs[0]: ", inputs[0])
    print("targets[0]: ", targets[0])
    break




# listing 11.3 preprocessing our datasets with a textvectorization layer

text_vectorization = TextVectorization(max_tokens=20000, output_mode = "multi_hot")
text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

binary_lgram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

binary_lgram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

binay_lgram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4 
)

# listing 11.4 inspecting the output of our binary unigram dataset

for inputs, targets in binary_lgram_train_ds:
    print("inputs shape: ", inputs.shape)
    print("inputs dtype: ", inputs.dtype)
    print("targets shape: ", targets.shape)
    print("targets dtype: ", targets.dtype)
    print("inputs[0]: ", inputs[0])
    print("targets[0]: ", targets[0])
    break


# listing 11.5 our model building utility

from tensorflow import keras
from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="rmsprop", 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    return model

# listing 11.6 training and testing the binary unigram model

model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_lgram.keras", save_best_only=True)
]
model.fit(
    binary_lgram_train_ds.cache(), 
    validation_data=binary_lgram_val_ds.cache(), 
    epochs=10, 
    callbacks=callbacks
)
model = keras.models.load_model("binary_lgram.keras")
print(f"Test acc: {model.evaluate(binay_lgram_test_ds)[1]:.3f}")

# listing 11.7 configuring the text vectorization layer to return bigrams

text_vectorization = TextVectorization(
    ngrams= 2,
    max_tokens=20000,
    output_mode="multi_hot"
)

# listing 11.8 training and testing the binary bigram model

text_vectorization.adapt(text_only_train_ds)
binary_2gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)

binary_2gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)

binary_2gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)


model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_2gram.keras", save_best_only=True)
]

model.fit(
    binary_2gram_train_ds.cache(), 
    validation_data=binary_2gram_val_ds.cache(),
    epochs=10,
    callbacks=callbacks
)

model = keras.models.load_model("binary_2gram.keras")
print(f"Test accuracy: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")

