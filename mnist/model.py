import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import numpy as np


np.random.seed(2019)

PATH_OUT = os.path.join(os.getcwd(), "mnist_model_py.tf")

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
LIMIT_EPOCHS = 10000


def prepare_data() -> tuple:
    """ Function to read and process data """

    (train_img, train_label), (test_img, test_label) = mnist.load_data()
    train_img = train_img / 255.
    test_img = test_img / 255.

    return train_img, train_label, test_img, test_label


def model_definition() -> keras.Sequential:
    """ Function to define the model """

    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


def callbacks_definition(learning_rate: float = .001) -> list:
    """ Kears callbacks preparation """

    def _scheduler(epoch: int) -> float:
        epoch_thresh = 5
        alpha = .1
        
        if epoch < eposh_thresh:
            return learning_rate
        return learning_rate * np.exp(alpha * (eposh_thresh - epoch))

    return [keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=1e-2,
                                          patience=2,
                                          verbose=0,
                                          mode='min',
                                          baseline=None,
                                          restore_best_weights=True),
            keras.callbacks.LearningRateScheduler(_scheduler),
            ]


if __name__ == "__main__":
    try:
        model = model_definition()

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
    except Exception as ex:
        print("Model definition and compilation problem. Error:\n%s", ex)
        sys.exit(1)

    try:
        x_train, y_train, x_test, y_test = prepare_data()

    except Exception as ex:
        print("Data not available. Error:\n%s", ex)
        sys.exit(1)
        
    callbacks = callbacks_definition(learning_rate=LEARNING_RATE)

    model.fit(x=x_train, y=y_train,
              validation_data=(x_test, y_test),
              batch_size=BATCH_SIZE, epochs=LIMIT_EPOCHS,   
              use_multiprocessing=True, workers=4,
              shuffle=True,
              callbacks=callbacks)

    evaluation = model.evaluate(x_test, y_test, verbose=0)

    print("Model test evaluation: {}% accuracy.".format(round(evaluation[1] * 100, 2)))

    print(f"Saving model to {PATH_OUT}")
    model.save(PATH_OUT)
