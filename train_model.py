import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow

#Variables pour les parametres
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2
L2_FACTOR = 0.001

#Lancement de la session de suivi MLflow
with mlflow.start_run():
     # Enregistrement des param tres
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)
    mlflow.log_param("l2_factor", L2_FACTOR)

# Construction et entra nement du modele (utiliser les variables definies)

# chargement des  donnees mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Use 90% for training and 10% for validation (dev)
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]
# normalisation des donnees
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#redimmensionnnement des images pour le reseaux fully connected
x_train = x_train.reshape((54000, 784))
x_val = x_val.reshape((6000, 784))
x_test = x_test.reshape((10000, 784))

optimizers = {
    "SGD_with_momentum": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": "rmsprop",
    "Adam": "adam"
}
for opt_name, optimizer in optimizers.items():
    with mlflow.start_run(run_name=f"Optimizer_Comparison_{opt_name}"):
        # construction du modele
        model = keras.Sequential(
            [
                keras.layers.Dense(512, activation='relu', input_shape=(784,),  kernel_regularizer=keras.regularizers.l2(L2_FACTOR)),
                keras.layers.Dropout(DROPOUT_RATE),
                keras.layers.Dense(10, activation='softmax')
            ]
        )


        #Compilation du modele
        model.compile(
            optimizer=optimizer,
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"]
        )

        history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_data=(x_val, y_val) # Use the validation set
        )




        #Entrainement du modele
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split = 0.1
        )

        #Evaluation du modele
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"Précision sur les donn es de test: {test_acc:.4f}")

        # Log metrics and parameters with MLflow
        mlflow.log_param("optimizer", opt_name)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)

# Enregistrement du mod le complet
mlflow.keras.log_model(model, name = "mnist-model-regularized")
print(" Modele sauvegardé sous mnist_model_regularized dans MLflow")

#Sauvegarde du modele
#model.save("mnist_model.h5")
#print(" Mod le sauvegard sous mnist_model.h5")


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss - Regularized Model')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy - Regularized Model')
plt.legend()
plt.show()
