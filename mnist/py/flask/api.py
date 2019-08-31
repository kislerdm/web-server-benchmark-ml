import os
import sys
from web_server_benchmark_mnist.utils import get_logger
from web_server_benchmark_mnist.model import Predictor
from flask import Flask, Response, flash, request
from werkzeug.utils import secure_filename
import json
from multidict import MultiDict
from collections import namedtuple


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
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


class Model(Predictor):
    def handler(self, request):
        pass


model = Model(PATH_MODEL)


# wrk workaroung to use GET request instead of POST
class File:
    filename = "image_2.jpeg"

    def read(self):
        with open(PATH_IMAGE_TEST, 'rb') as f:
            return f.read()


requestObj = namedtuple("request", ["files"])
REQUEST = requestObj(files=MultiDict({"image": File()}))

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def handler():

    # wrk workaroung to use GET request instead of POST
    request = REQUEST

    if POST_OBJ_KEY not in request.files:
        return Response(response=json.dumps({"data": None}),
                        content_type='application/json',
                        status=406)
    
    file = request.files[POST_OBJ_KEY]
    if not file or not allowed_file(file.filename):
        return Response(response=json.dumps({"data": None}),
                        content_type='application/json',
                        status=406)
    
    image = file.read()
    try:
        image = model.image_adjust(image)

    except Exception as ex:
        logs.error("Image transformation error.\n%s", ex)
        return Response(response=json.dumps({"data": None}),
                        content_type='application/json',
                        status=500)

    prob, digit, err = model.get_prediction(image)
    if err:
        logs.error("Prediciton error.\n%s", err)
        return Response(response=json.dumps({"data": None}),
                        content_type='application/json',
                        status=500)

    return Response(response=json.dumps({"data":
                                         {"digit": digit,
                                          "probability": prob
                                          }
                                         }
                                        ),
                    content_type='application/json',
                    status=200)


if __name__ == "__main__":
    app.run()
