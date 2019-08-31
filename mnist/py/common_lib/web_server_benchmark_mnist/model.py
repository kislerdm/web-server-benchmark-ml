from typing import Tuple
import numpy as np
import cv2
from tensorflow import keras
from abc import ABC, abstractmethod


class Predictor(ABC):

    """ Mode class tp predict a digit """

    def __init__(self, path: str):
        try:
            self.model = keras.models.load_model(path)
        except Exception as ex:
            raise ex

    def get_prediction(self, input: np.array) -> Tuple[Tuple[float, int], str]:
        """ 
            Function to run a prediction of a digit:

                Args:
                    input: np.array of the "image" with the shape (1,28,28)

                Returns:
                    tuple of tuple with probabily of prediction and the predicted digit value
                        and the err str in case of exception
        """
        try:
            prediction = self.model.predict(input)
            return float(prediction.max()), int(prediction.argmax()), None

        except Exception as ex:
            return None, None, ex

    def image_adjust(self, img_binary: bytes, image_size=(28, 28)) -> np.array:
        """ 
            Function to read and prepare the image to input the model input layer 

                Args:
                    img_binary: image binary in-memory object
                    image_size: tuple with the desired image size

                Returns:
                    numpy array for prediction with the model
        """

        img_array = np.frombuffer(img_binary, dtype=np.uint8)
        # convert image to grayscale
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        # resize the image
        img = cv2.resize(img, image_size)
        # invert the image scale
        img = 255. - img
        # norm to 1
        img = img / 255.
        # reshape to align with the model's input layer requirement
        img = img.reshape(1, image_size[0], image_size[1])

        return img
    
    @abstractmethod
    def handler(self, request):
        """ Web server handler function """
        pass
