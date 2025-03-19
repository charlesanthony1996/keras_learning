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


# from tensorflow import keras
# batch_size = 32

# train_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/train", batch_size=batch_size
# )

# val_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/val", batch_size=batch_size
# )

# test_ds = keras.utils.text_dataset_from_directory(
#     "aclImdb/test", 
# )

# listing 11.2 displaying the shapes and dtypes of the first batch

