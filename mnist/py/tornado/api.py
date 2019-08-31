import os
import sys
from web_server_benchmark_mnist.utils import get_logger
from web_server_benchmark_mnist.model import Predictor
import tornado.ioloop
import tornado.web
import logging

PORT = 4500
POST_OBJ_KEY = "image"

logs = get_logger()

DIR = os.getcwd()
PATH_MODEL = os.path.join(DIR, "../../model_train/model/mnist_model_py_keras.h5")

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
FILE = open(PATH_IMAGE_TEST, 'rb').read()


class Model(Predictor):
    def handler(self, request):
        pass


model = Model(PATH_MODEL)


# turn off tornado logger
logs_tornado = logging.NullHandler()
logs_tornado.setLevel(logging.DEBUG)
logging.getLogger("tornado.access").addHandler(logs_tornado)
logging.getLogger("tornado.access").propagate = False


class Handler(tornado.web.RequestHandler):
    def get(self):
        """ Function to handle image POST request """
        
        # wrk workaroung to use GET request instead of POST
        file = FILE
        # if POST_OBJ_KEY not in self.request.files.keys():
        #     self.set_status(406)
        #     self.write({"data": None})

        # file = self.request.files[POST_OBJ_KEY][0]['body']

        # type = self.request.files[POST_OBJ_KEY][0]['content_type']
        # if "image" not in type and\
        #         "octet-stream" not in type:
        #     self.set_status(406)
        #     self.write({"data": None})
        
        try:
            image = model.image_adjust(file)

        except Exception as ex:
            logs.error("Image transformation error.\n%s", ex)
            self.set_status(406)
            self.write({"data": None})

        prob, digit, err = model.get_prediction(image)
        if err:
            logs.error("Prediciton error.\n%s", err)
            self.set_status(500)
            self.write({"data": None})

        self.write({"data":
                    {"digit": digit,
                     "probability": prob
                     }
                    }
                   )


def make_app():
    return tornado.web.Application([
        (r"/", Handler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(4500, address='0.0.0.0')
    tornado.ioloop.IOLoop.current().start()
