package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
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
	predictor   *tf.SavedModel
	imageBuffer bytes.Buffer
)

// workaround for wrk to use GET request instead of POST
var (
	reqType    = "image/jpeg"
	uploadName = os.Getenv("POST_OBJ_KEY")
)

func main() {

	model, err := tf.LoadSavedModel(os.Getenv("PATH_MODEL"), []string{"serve"}, nil)
	if err != nil {
		log.Print(fmt.Sprintf("Error loading model from %s", os.Getenv("PATH_MODEL")))
		os.Exit(1)
	}
	predictor = model

	// workaround for wrk to use GET request instead of POST
	image, err := os.Open(os.Getenv("PATH_IMAGE_TEST"))
	if err != nil {
		log.Print(fmt.Sprintf("Cannot read image. Error:\n%s", err))
		os.Exit(1)
	}
	io.Copy(&imageBuffer, image)

	PORT, err := strconv.Atoi(os.Getenv("PORT"))
	if err != nil {
		PORT = 4500
	}
	http.HandleFunc("/", handler)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", PORT), nil))
}

func findMaxArgMax(arr []float32) (max float32, argmax uint8) {
	argmax = 0
	max = arr[argmax]

	for i, value := range arr {
		if value > max {
			max = value
			argmax = uint8(i)
		}
	}
	return max, argmax
}

func handler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	// workaround for wrk to use GET request instead of POST
	// imageFile, imageHeader, err := r.FormFile(uploadName)
	// if err != nil {
	// 	w.WriteHeader(http.StatusBadRequest)
	// 	w.Write([]byte(`{"data": null}`))
	// }
	// reqType := imageHeader.Header.Get("Content-Type")

	mimeType := strings.Split(reqType, "/")[0]
	if mimeType != "image" && reqType != "application/octet-stream" {
		w.WriteHeader(http.StatusBadRequest)
		log.Print(fmt.Sprintf("Wrong MIME type"))
		w.Write([]byte(`{"data": null}`))
	}
	imgType := strings.Split(reqType, "/")[1]

	if reqType == "application/octet-stream" {
		imgType = "png"
	}
	// workaround for wrk to use GET request instead of POST
	// defer imageFile.Close()
	// var imageBuffer bytes.Buffer
	// // Copy image data to a buffer
	// io.Copy(&imageBuffer, imageFile)

	tensor, err := makeTensorFromImage(&imageBuffer, imgType)
	if err != nil {
		log.Print(fmt.Sprintf("Error on making tensor out of the image. Error:\n%s", err))
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"data": null}`))
	}

	result, err := predictor.Session.Run(
		map[tf.Output]*tf.Tensor{
			predictor.Graph.Operation("l_0_input").Output(0): tensor,
		},
		[]tf.Output{
			predictor.Graph.Operation("l_out/Softmax").Output(0),
		},
		nil,
	)
	if err != nil {
		log.Print(fmt.Sprintf("Error on prediction. Error:\n%s", err))
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"data": null}`))
	}

	probability, digit := findMaxArgMax(result[0].Value().([][]float32)[0])

	reposnseObj := Response{
		Data: ResponseObj{
			Digit:       digit,
			Probability: probability,
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
