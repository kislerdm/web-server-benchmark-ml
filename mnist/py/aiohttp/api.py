import os
import sys
from web_server_benchmark_mnist.utils import get_logger
from web_server_benchmark_mnist.model import Predictor
from aiohttp import web


PORT = 4500
POST_OBJ_KEY = "image"

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
    logs.error(
        f"File {PATH_IMAGE_TEST} doesn't exist, cannot load the test image")
    sys.exit(1)


# wrk workaroung to use GET request instead of POST
class io_file:
    body = open(PATH_IMAGE_TEST, 'rb').read()
    type = "image/jpeg"


FILE = io_file


class Endpoint(Predictor):

    async def handler(self, request) -> web.json_response:
        """ Web server handler function """
        
        try:
            reader = await request.multipart()
            data = await reader.next()
            image = await data.read()
        except Exception as ex:
            logs.error(ex)
            return web.json_response(body={"data": None}, status=web.HTTPNotAcceptable)

        try:
            image = self.image_adjust(images)

        except Exception as ex:
            logs.error("Image transformation error.\n%s", ex)
            return web.json_response(body={"data": None}, status=web.HTTPInternalServerError)
        return web.json_response(body={"data": 1})
        # prob, digit, err = self.get_prediction(image)
        # if err:
        #     logs.error("Prediciton error.\n%s", err)
        #     return web.json_response(body={"data": None}, status=web.HTTPInternalServerError)

        # return web.json_response(body={"data":
        #                                {"digit": digit,
        #                                 "probability": prob
        #                                 }
        #                                }
        #                          )


# async def handler(request) -> web.json_response:
#         """ Web server handler function """

#         reader = await request.post()
#         image = await reader.next()
#         print(await image.read(decode=True))
        
#         return web.json_response({"a": 1})

if __name__ == "__main__":
    endpoint = Endpoint(PATH_MODEL)

    app = web.Application()
    app.router.add_route(method="POST", path="/", handler=endpoint.handler)
    web.run_app(app, host="0.0.0.0", port=PORT)
