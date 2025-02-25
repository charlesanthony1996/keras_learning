import numpy as np
import tensorflow as tf
random_numbers = np.random.normal(size=(1000, 16))
# print(random_numbers)
dataset = tf.data.Dataset.from_tensor_slices(random_numbers)


# for i, element in enumerate(random_numbers):
#     print(element.shape)
#     if i >= 2:
#         break

# batched_dataset = dataset.batch(32)
# for i, element in enumerate(batched_dataset):
#     print(element.shape)
#     if i >= 2:
#         break

# reshaped_dataset = dataset.map(lambda x: tf.reshape(x, (4, 4)))
# for i, element in enumerate(reshaped_dataset):
#     print(element.shape)
#     if i >= 2:
#         break

# displaying the shapes of the data and labels yielded by the dataset
for data_batch, labels_batch in train_dataset:
    print("data batch shape: ", data_batch.shape)
    print("labels batch shape: ", labels_batch.shape)
    break

