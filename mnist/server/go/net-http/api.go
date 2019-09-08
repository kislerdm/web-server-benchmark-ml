package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
)

// Response type
type Response struct {
	Data ResponseObj `json:"data"`
}

// ResponseObj type, payload of Response object
type ResponseObj struct {
	Digit       uint8   `json:"digit"`
	Probability float32 `json:"probability"`
}

var (
	graphModel   *tf.Graph
	sessionModel *tf.Session
	reqType      string
)

var fileImage = "/Users/dkisler/projects/web-server-benchmark-ml/mnist/test_2.jpeg"
var modelPath = "/Users/dkisler/projects/web-server-benchmark-ml/mnist/dnn/model/mnist_model_py/saved_model.pb"

func loadModel() error {
	// Load inception model
	model, err := ioutil.ReadFile(modelPath)
	if err != nil {
		return err
	}
	graphModel = tf.NewGraph()
	if err := graphModel.Import(model, ""); err != nil {
		return err
	}

	sessionModel, err = tf.NewSession(graphModel, nil)
	if err != nil {
		return err
	}

	return nil
}

func handler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// workareoung to use wrk benchmark
	imageFile, err := os.Open(fileImage)
	reqType := "image/jpeg"
	if err != nil {
		log.Print(fmt.Sprintf("Cannot read image. Error:\n%s", err))
		os.Exit(1)
	}

	// imageFile, imageHeader, err := r.FormFile("image")
	// if err != nil {
	// 	w.WriteHeader(http.StatusBadRequest)
	// 	w.Write([]byte(`{"data": null}`))
	// }
	// reqType := imageHeader.Header.Get("Content-Type")

	mimeType := strings.Split(reqType, "/")[0]
	if mimeType != "image" && reqType != "application/octet-stream" {
		w.WriteHeader(http.StatusBadRequest)
		log.Print(fmt.Sprintf("Wrong MIME type. Error:\n%s", err))
		w.Write([]byte(`{"data": null}`))
	}

	// defer imageFile.Close()
	var imageBuffer bytes.Buffer
	// Copy image data to a buffer
	io.Copy(&imageBuffer, imageFile)

	tensor, err := makeTensorFromImage(&imageBuffer, "jpeg")
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"data": null}`))
	}

	_, err = sessionModel.Run(
		map[tf.Output]*tf.Tensor{
			graphModel.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			graphModel.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"data": null}`))
		log.Print(fmt.Sprintf("Could not run tf session prediction. Error:\n, %s", err))
	}

	reposnseObj := Response{
		Data: ResponseObj{
			Digit:       uint8(2),
			Probability: 0.6915019751,
		},
	}

	reposnse, err := json.Marshal(reposnseObj)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"data": null}`))
		log.Print(fmt.Sprintf("Cannot parse %b as JSON", reposnseObj))
	} else {
		w.WriteHeader(http.StatusOK)
		w.Write(reposnse)
	}
}

func main() {

	if err := loadModel(); err != nil {
		log.Fatal(err)
		os.Exit(1)
	}

	PORT := 4500
	http.HandleFunc("/", handler)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", PORT), nil))
}
