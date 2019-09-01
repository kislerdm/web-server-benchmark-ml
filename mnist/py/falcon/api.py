import os
import sys
import json
import falcon
from falcon_multipart.middleware import MultipartMiddleware
import mimetypes
from web_server_benchmark_mnist.utils import get_logger
from web_server_benchmark_mnist.model import Predictor


POST_OBJ_KEY = "image"

logs = get_logger()

DIR = os.getcwd()
PATH_MODEL = os.path.join(
    DIR, "../../model_train/model/mnist_model_py_keras.h5")

if not os.path.isfile(PATH_MODEL):
    logs.error(f"File {PATH_MODEL} doesn't exist, cannot load the model")
    sys.exit(1)

# test image with the digit "2" on it
PATH_IMAGE_TEST = os.path.join(DIR, "../../test_2.jpeg")
if not os.path.isfile(PATH_IMAGE_TEST):
    logs.error(
        f"File {PATH_IMAGE_TEST} doesn't exist, cannot load the test image")
    sys.exit(1)


class Model(Predictor):
    def handler(self, request):
        pass


model = Model(PATH_MODEL)


# wrk workaroung to use GET request instead of POST
FILE = open(PATH_IMAGE_TEST, 'rb').read()

app = falcon.API(middleware=[MultipartMiddleware()])


class Handler(object):
    # def on_post(self, req, resp):
    #     data = req.get_param(POST_OBJ_KEY)
    #     # # Read image as binary
    #     image = data.file.read()
    def on_get(self, req, resp):
        # wrk workaroung to use GET request instead of POST
        image = FILE
        try:
            image = model.image_adjust(image)

        except Exception as ex:
            logs.error("Image transformation error.\n%s", ex)
            resp.status = falcon.HTTP_500
            resp.body(json.dumps({"data": None}))

        prob, digit, err = model.get_prediction(image)
        if err:
            logs.error("Prediciton error.\n%s", err)
            resp.status = falcon.HTTP_500
            resp.body(json.dumps({"data": None}))

        resp.status = falcon.HTTP_200
        resp.body = json.dumps({"data":
                                {"digit": digit,
                                 "probability": prob
                                 }
                                })


handler = Handler()
app.add_route('/', handler)
