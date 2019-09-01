import os
import sys
from web_server_benchmark_mnist.utils import get_logger
from web_server_benchmark_mnist.model import Predictor
from aiohttp import web
from multidict import MultiDict
from collections import namedtuple


PORT = 4500
POST_OBJ_KEY = "image"

logs = get_logger()

DIR = os.getcwd()
PATH_MODEL = os.path.join(DIR, "../../model_train/model/mnist_model_py_keras.h5")

if not os.path.isfile(PATH_MODEL):
    logs.error(f"File {PATH_MODEL} doesn't exist, cannot load the model")
    sys.exit(1)

# test image with the digit "2" on it
PATH_IMAGE_TEST = os.path.join(DIR, "../../test_2.jpeg")
if not os.path.isfile(PATH_IMAGE_TEST):
    logs.error(
        f"File {PATH_IMAGE_TEST} doesn't exist, cannot load the test image")
    sys.exit(1)

# wrk workaroung to use GET request instead of POST
FileField = namedtuple("FileField", ['name', 'file_name', 'content_type', 'file'])
DATA = MultiDict({"image": FileField(
    name=POST_OBJ_KEY,
    content_type="image/jpeg",
    file_name="test_2.jpeg",
    file=open(PATH_IMAGE_TEST, 'rb').read()
)})


class Endpoint(Predictor):

    async def handler(self, request) -> web.json_response:
        """ Web server handler function """

        try:
            # data = await request.post()
            # workaround for wrk to use GET request instead of POST
            data = DATA.copy()
            if POST_OBJ_KEY not in data:
                return web.json_response({"data": None}, status=web.HTTPNotAcceptable)

            if "image" not in data[POST_OBJ_KEY].content_type and\
                    "octet-stream" not in data[POST_OBJ_KEY].content_type:
                return web.json_response({"data": None}, status=web.HTTPNotAcceptable)
            
            # workaround for wrk to use GET request instead of POST
            file = data[POST_OBJ_KEY].file
            # image = file.read()
            image = file

        except Exception as ex:
            print(ex)
            return web.json_response({"data": None}, status=web.HTTPNotAcceptable)

        try:
            image = self.image_adjust(image)

        except Exception as ex:
            logs.error("Image transformation error.\n%s", ex)
            return web.json_response({"data": None}, status=web.HTTPInternalServerError)

        prob, digit, err = self.get_prediction(image)
        if err:
            logs.error("Prediciton error.\n%s", err)
            return web.json_response(body={"data": None}, status=web.HTTPInternalServerError)

        return web.json_response({"data":
                                  {"digit": digit,
                                   "probability": prob
                                   }
                                  }
                                 )


if __name__ == "__main__":
    endpoint = Endpoint(PATH_MODEL)

    app = web.Application()
    app.router.add_route(method="GET", path="/", handler=endpoint.handler)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)
