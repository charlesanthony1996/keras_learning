from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# print(model.weights)
model.build(input_shape=(None, 3))
print(model.weights)

model.summary()

# naming models and layers with the name argument
model = keras.Sequential(name="my_example_model")
model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
model.build((None, 3))
model.summary()


# specifying the input shape of your model in advance
model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(64, activation="relu"))
model.summary()

inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs = outputs)

inputs = keras.Input(shape=(3,), name="my_input")

print(inputs.shape)
print(inputs.dtype)


features = layers.Dense(64, activation="relu")(inputs)
# print(features)
print(features.shape)

outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs = outputs)

model.summary()

# a multi input, multi output functional model
# page 178
vocabulary_size = 10000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name='title')
text_body = keras.Input(shape=(vocabulary_size,), name = 'text_body')
tags = keras.Input(shape=(num_tags,), name='tags')

features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)

priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(num_departments , activation="softmax", name="department")(features)

model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])


import numpy as np

num_samples = 1280

# dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# dummy target data
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(
    optimizer="rmsprop", 
    loss=["mean_squared_error", "categorical_crossentropy"],
    metrics=[["mean_squared_error"], ["accuracy"]])

model.fit([title_data, text_body_data, tags_data], [priority_data, department_data], epochs= 1)

model.evaluate([title_data, text_body_data, tags_data], [priority_data, department_data])

priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])


# listing 7.11 training a model by providing dicts of input and target arrays
model.compile(
    optimizer="rmsprop",
    loss={"priority": "mean_squared_error", "department": "categorical_crossentropy"},
    metrics={"priority": ["mean_absolute_error"], "department": ["accuracy"]})


model.fit(
    {"title":title_data, "text_body": text_body_data, "tags": tags_data}, 
    {"priority": priority_data, "department": department_data}, epochs= 1)

model.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data})

priority_preds, deparment_preds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data}
)

# keras.utils.plot_model(model, "ticket_classifier.png")
keras.utils.plot_model(model, "ticket_classifier.png", show_shapes=True)


# listing 7.12 retrieving the inputs or outputs of a layer in a functional model
# print(model.layers)
# print(model.layers[3].input)


# listing 7.13 creating a new model by reusing intermediate layer ouputs
features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)

new_model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department, difficulty]
)

keras.utils.plot_model(new_model, "updated_ticket_classifier.png", show_shapes=True)


# listing 7.14 a simple subclassed model
class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(num_departments, activation="softmax")

    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]


        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)

        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department


model = CustomerTicketModel(num_departments = 4)

priority, department = model({"title": title_data, "text_body": text_body_data, "tags": tags_data})

model.compile(
    optimizer="rmsprop",
    loss=["mean_squared_error", "categorical_crossentropy"],
    metrics=[["mean_absolute_error"], ["accuracy"]])

model.fit({
    "title": title_data,
    "text_body": text_body_data,
    "tags": tags_data
    }, [priority_data, department_data])

priority_preds, department_preds = model.predict({"title": title_data, "text_body": text_body_data, "tags": tags_data})


# listing 7.15 creating a functional model that includes a subclassed model

class Classifier(keras.Model):
    def __init__(self, num_classes = 2):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        self.dense = layers.Dense(num_units, activation=activation)

    def call(self, inputs):
        return self.dense(inputs)

inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = Classifier(num_classes = 10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

# listing 7.16 creating a subclassed model that includes a functional model
inputs = keras.Input(shape=(64,))
outputs = layers.Dense(1, activation="sigmoid")(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)

class MyModel(keras.Model):
    def __init__(self, num_classes= 2):
        super().__init__()
        self.dense = layers.Dense(64, activation="relu")
        self.classifier = binary_classifier

    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)


model = MyModel()

# listing 7.17 the standard workflow: compile, fit, evaluate, predict
from tensorflow.keras.datasets import mnist

def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)

    model = keras.Model(inputs, outputs)
    return model

print(model.summary())

(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels))

test_metrics = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)


# listing 7.18 implementing a custom metric by subclassing the metric class

import tensorflow as tf

class RootMeanSquaredError(keras.metrics.Metric):
    def __init__(self, name='rmse', **kwargs):
        super().__init__(name = name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        return tf.sqrt(self.mse_sum/ tf.cast(self.total_samples, tf.float32))

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)

model = get_mnist_model()
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", RootMeanSquaredError()]
)

model.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)

# listing 7.19 using the callbacks argument in the fit() method

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
    )
]

model = get_mnist_model()
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=1, callbacks=callbacks_list, validation_data=(val_images, val_labels))

model = keras.models.load_model("checkpoint_path.keras")


# writing your own callbacks

# listing 7.20 creating a custom callback by subclassing the classfication
from matplotlib import pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses),
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"Plot_at_epoch_{epoch}")
        self.per_batch_losses = []


model = get_mnist_model()
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=1, callbacks=[LossHistory()], validation_data=(val_images, val_labels))

# using the tensorboard

model = get_mnist_model()
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

tensorboard = keras.callbacks.TensorBoard(log_dir="./logs_for_chapter_7")

model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels), callbacks=[tensorboard])

# low level usage of metrics

metric = keras.metrics.SparseCategoricalAccuracy()
targets = [0, 1, 2]
predictions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
metric.update_state(targets, predictions)
current_result = metric.result()
print(f"result: {current_result}")

values = [0, 1, 2, 3, 4]
mean_tracker = keras.metrics.Mean()
for value in values:
    print(f"Mean of values: {mean_tracker.result():.2f}")

# 7.21 writing a step by step training loop: the training step function

model = get_mnist_model()

loss_fn = keras.losses.SparseCategoricalAccuracy()
optimizer = keras.optimizer.RMSProp()
metrics = [keras.metrics.SparseCategoricalAccuracy()]
loss_tracking_metric = keras.metrics.Mean()

def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, predictions)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.training_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    logs = {}
    for metric in metrics:
        metric.update_space(targets, predictions)
        logs[metric.name] = metric.result()

    logs_tracking_metric.update_state(loss)
    logs["logs"] = loss_tracking_metric.result()
    return logs


# listing 7.22 writing a step by step training loop: resetting the metrics

def reset_metrics:
    pass