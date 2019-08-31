import os
import sys
from web_server_benchmark_mnist.utils import get_logger
from web_server_benchmark_mnist.model import Predictor
from sanic import Sanic, response


PORT = 4500

logs = get_logger()

DIR = os.getcwd()
PATH_MODEL = os.path.join(
    DIR, "../../model_train/model/mnist_model_py_keras.h5")

if not os.path.isfile(PATH_MODEL):
    logs.error(f"File {PATH_MODEL} doesn't exist, cannot load the model")
    sys.exit(1)

# test image with the digit "2" on it
PATH_IMAGE_TEST = "../../test_2.jpeg"
if not os.path.isfile(PATH_IMAGE_TEST):
    logs.error(f"File {PATH_IMAGE_TEST} doesn't exist, cannot load the test image")
    sys.exit(1)
    

# wrk workaroung to use GET request instead of POST
class io_file:
    body = open(PATH_IMAGE_TEST, 'rb').read()
    type = "image/jpeg"

FILE = io_file        


class Endpoint(Predictor):

    async def handler(self, request) -> response.json:
        """ Web server handler function """

        # file = request.files.get("image")
        # workaround for wrk to use GET request instead of POST
        file = FILE
        if file is None:
            return response.json(body={"data": None}, status=406)

        if "image" not in file.type and\
                "octet-stream" not in file.type:
            return response.json(body={"data": None}, status=406)

        try:
            image = self.image_adjust(file.body)

        except Exception as ex:
            logs.error("Image transformation error.\n%s", ex)
            return response.json(body={"data": None}, status=500)

        prob, digit, err = self.get_prediction(image)
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
    endpoint = Endpoint(PATH_MODEL)

    app = Sanic()
    app.add_route(endpoint.handler, '/', methods=['POST', 'GET'])
    app.run(host="0.0.0.0", port=PORT, access_log=False)
