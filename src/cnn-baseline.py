import tensorflow as tf
from tensorflow import keras

# 1 Datensatz laden
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Bilder normalisieren
x_train = x_train / 255
x_test = x_test / 255

# Kanal hinzufügen
x_train = x_train[..., None]
x_test = x_test[..., None]

# 2 Modell definieren
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32,(3,3),activation="relu"),                        #Convolution layer, 32 Filter, vertikale, horizontale Linien, Kurve
    keras.layers.MaxPooling2D((2,2)),                                       #Bild verkleinern
    keras.layers.Flatten(),                                                 #Polling layer aus Matrix wird Vektor
    keras.layers.Dense(64,activation="relu"),                               #Dense layer kombiniert Features
    keras.layers.Dense(10,activation="softmax")                             #Output layer 10 da 10 Ziffern
])

# 3 Modell kompilieren
model.compile(
    optimizer="adam",                                                       #Adam optimiert
    loss="sparse_categorical_crossentropy",                                 #loss misst Fehler
    metrics=["accuracy"]                                                    #wie viele Bilder wurden richtig erkannt
)

# 4 Modell trainieren
model.fit(
    x_train,
    y_train,
    epochs=10,                                                              #Datensatz wird 10 mal durchlaufen
    batch_size=64                                                           #Wie viele Bilder parallel trainiert werden
)

# 5 Modell testen
model.evaluate(x_test, y_test)