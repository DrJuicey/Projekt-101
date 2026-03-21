import tensorflow as tf
from tensorflow import keras

# MNIST laden
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Werte von 0–255 auf 0–1 skalieren
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNN braucht Kanal-Dimension, Graustufen = 1 Kanal
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Einfaches Baseline CNN
model = keras.Sequential()

# Input Shape der Bilder
model.add(keras.layers.InputLayer(input_shape=(28,28,1)))

# Erkennt einfache Features wie Kanten etc.
model.add(keras.layers.Conv2D(32,(3,3),activation="relu"))

# Verkleinert Feature Maps
model.add(keras.layers.MaxPooling2D((2,2)))

# Matrix, Vektor für Dense Layer
model.add(keras.layers.Flatten())

# Kombiniert Features
model.add(keras.layers.Dense(64,activation="relu"))

# 10 Outputs wegen Ziffern 0–9
model.add(keras.layers.Dense(10,activation="softmax"))

# Modell vorbereiten
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Training
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# Testen
loss, accuracy = model.evaluate(x_test, y_test)

print("Test accuracy:", accuracy)
