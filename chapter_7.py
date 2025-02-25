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

