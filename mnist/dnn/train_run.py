import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import numpy as np
import json
import time
import warnings
warnings.filterwarnings("ignore")


np.random.seed(2019)

PATH_OUT = os.path.join(os.getcwd(), "model/mnist_model_py")

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
LIMIT_EPOCHS = 1000


def prepare_data() -> tuple:
    """ Function to read and process data """

    (train_img, train_label), (test_img, test_label) = mnist.load_data()
    train_img = train_img / 255.
    test_img = test_img / 255.

    return train_img, train_label, test_img, test_label


def model_definition() -> keras.Sequential:
    """ Function to define the model """

    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28), name="l_0"),
        keras.layers.Dense(200, activation=keras.activations.relu, 
                           kernel_regularizer=keras.regularizers.l2(.001), 
                           name="l_1"),
        keras.layers.Dropout(0.2, name='dropout_0'),
        keras.layers.Dense(10, activation='softmax', name="l_out")
    ])
    

def model_remove_dropout(model_train: keras.Sequential) -> keras.Sequential:
        """ Function to remove dropout layer prior to saving Sequential model """
        
        # get layers names and classes
        model_layers = model_train.get_config()['layers']
        layers_info = {layer['config']['name']: layer['class_name'] for layer in model_layers}
        if "Dropout" not in layers_info.values():
            return model_train
        
        layers_selected = [layer_name for layer_name, layer_class in layers_info.items() if layer_class != "Dropout"]
        model = keras.Sequential()
        
        for layer_name in layers_selected:
            model.add(model_train.get_layer(name=layer_name))

        return model
                        

def model_train_history_as_json(train_history: keras.callbacks.History) -> dict:
    """ Function to convert History object into a json """
    
    def _type_cast(lst):
        """ Cast numpy to in-built py types"""
        
        if 'numpy.float' in str(type(lst[0])):
            return [float(i) for i in lst]
        if 'numpy.int' in str(type(lst[0])):
            return [int(i) for i in lst]
        return lst
        
    history_json = {k: _type_cast(v) for k, v in train_history.history.items()}
    history_json['epoch'] = train_history.epoch
    
    return history_json


def callbacks_definition(learning_rate: float = .001) -> list:
    """ Kears callbacks preparation """

    def _scheduler(epoch: int) -> float:
        epoch_thresh = 5
        alpha = .1
        
        if epoch < epoch_thresh:
            return learning_rate
        return learning_rate * np.exp(alpha * (epoch_thresh - epoch))

    return [keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=1e-4,
                                          patience=2,
                                          verbose=0,
                                          mode='min',
                                          restore_best_weights=True),
            keras.callbacks.LearningRateScheduler(_scheduler),
            ]


if __name__ == "__main__":
    try:
        # init session
        # tf.keras.backend.clear_session()
        # session = tf.compat.v1.Session()
        # tf.compat.v1.keras.backend.set_session(session)
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

    history = model.fit(x=x_train, y=y_train,
                        validation_data=(x_test, y_test),
                        batch_size=BATCH_SIZE, epochs=LIMIT_EPOCHS,   
                        use_multiprocessing=True, workers=4,
                        shuffle=True,
                        callbacks=callbacks)

    evaluation = model.evaluate(x_test, y_test, verbose=0)
    print("Model test evaluation: {}% accuracy.".format(round(evaluation[1] * 100, 2)))
    
    # remove dropout layers
    model_out = model_remove_dropout(model)
    
    print(f"Saving model to {PATH_OUT}")
    if os.path.isdir(PATH_OUT):
        os.system(f"mv -f {PATH_OUT} {PATH_OUT}_dump_{time.strftime('%Y%m%d%H%M%S')}")

    tf.keras.experimental.export_saved_model(model_out, PATH_OUT)
    model.save(f"{PATH_OUT}_keras.h5")
    
    with open(f"{PATH_OUT}_train_history.json", 'w') as f:
        json.dump(model_train_history_as_json(history), f)
    
    # builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(PATH_OUT)
    # builder.add_meta_graph_and_variables(session, ["model_py"])
    # builder.save()    
    
    # session.close()
