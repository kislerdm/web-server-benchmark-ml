import os
import sys
import json
import logging
from typing import Tuple
import numpy as np
import cv2
from tensorflow import keras
from sanic import Sanic, response

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logs = logging.getLogger('logs')

DIR = os.getcwd()
PATH_MODEL = os.path.join(DIR, "../../mnist/mnist_model_py_keras.h5")

if not os.path.isfile(PATH_MODEL):
    logs.error(f"File {PATH_MODEL} doesn't exist, cannot load the model")
    sys.exit(1)

PORT = 4500


class Predictor:

    """ Mode class tp predict a digit """

    def __init__(self, path: str):
        try:
            self.model = keras.models.load_model(path)
        except Exception as ex:
            raise ex

    async def get_prediction(self, input: np.array) -> Tuple[Tuple[float, int], str]:
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


    async def image_adjust(self, img_binary: bytes, image_size: tuple = (28, 28)) -> np.array:
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


    async def handler(self, request) -> response.json:
        """ Web server handler function """
        
        file = request.files.get("image")
        if "image" not in file.type and\
                "octet-stream" not in file.type:
            return response.json(body={"data": None}, status=406)

        try:
            image = await self.image_adjust(file.body)
        
        except Exception as ex:
            logs.error("Image transformation error.\n%s", ex)
            return response.json(body={"data": None}, status=500)

        prob, digit, err = await self.get_prediction(image)
        if err:
            logs.error("Prediciton error.\n%s", err)
            return response.json(body={"data": None}, status=500)

        return response.json(body={"data":
                                    {"digit": digit,
                                    "probability": prob
                                    }
                                }
                            )
        

if __name__ == "__main__":
    endpoint = Predictor(PATH_MODEL)
    
    app = Sanic()
    app.add_route(endpoint.handler, '/', methods=['POST', 'GET'])
    app.run(host="0.0.0.0", port=PORT, access_log=False)
