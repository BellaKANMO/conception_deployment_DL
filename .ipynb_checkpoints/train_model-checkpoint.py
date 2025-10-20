import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# chargement des  donnees mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalisation des donnees
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#redimmensionnnement des images pour le reseaux fully connected
x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))

# construction du modele
model = keras.Sequential(
    [
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ]
)

#Compilation du modele
modele.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

#Entrainement du modele
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size = 128,
    validation_split = 0.1
)

#Evaluation du modele
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Pr cision sur les donn es de test: {test_acc:.4f}")

#Sauvegarde du modele
model.save("mnist_model.h5")
print(" Mod le sauvegard sous mnist_model.h5")