const PORT = process.env.PORT,
  PATH_MODEL = process.env.PATH_MODEL,
  PATH_IMAGE_TEST = process.env.PATH_IMAGE_TEST,
  POST_OBJ_KEY = process.env.POST_OBJ_KEY;

var fs = require('fs'),
  express = require('express'),
  tf = require('@tensorflow/tfjs-node'),
  sharp = require('sharp'),
  multer = require('multer');

var app = express(),
  upload = multer();

// read pre-trained model
const model = tf.loadLayersModel("file://" + PATH_MODEL);

// workaround for wrk to use GET request instead of POST
app.get('/', (req, res) => {
  req = {
    file: {
      buffer: fs.readFileSync(PATH_IMAGE_TEST),
      mimetype: 'image/jpeg',
    }
  };
  // app.post('/', upload.single(POST_OBJ_KEY), (req, res) => {
  if (!req.file) {
    res.status(401).json({ data: null });
  }
  if (req.file.mimetype.split('/')[0] != 'image') {
    res.status(406).json({ data: null });
  }

  sharp(req.file.buffer)
    .grayscale() // set input image to grayscale
    .resize(28, 28) // resize input image
    .raw()
    .toBuffer((err, data) => {
      res.setHeader('Content-Type', 'application/json');
      if (err) {
        res.status(500).json({ data: null });
        console.log(err);
      } else {
        var image_raw = [];
        data.toJSON().data.forEach((pixel) => {
          const pix_invert = 255. - pixel; // invert the image
          image_raw.push(pix_invert / 255.); // normalize pixels values
        });

        model.then((model) => {
          const prediction = model.predict(tf.tensor3d(image_raw, [1, 28, 28]))
                                  .arraySync()[0];
          const prob = Math.max(...prediction);
          res.status(200).json({
            data: {
              digit: prediction.indexOf(prob),
              probability: prob,
            }
          });
        });
      };
    });
});

app.listen(PORT);
