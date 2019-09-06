package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

type ResponseObj struct {
	Digit       uint8   `json:"digit"`
	Probability float32 `json:"probability"`
}

type Response struct {
	Data ResponseObj `json:"data"`
}

func handler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	reposnse := Response{
		Data: ResponseObj{
			Digit:       uint8(2),
			Probability: 0.6915019751,
		},
	}

	resp, err := json.Marshal(reposnse)
	if err != nil {
		e := fmt.Sprintf("Cannot parse %b as JSON", reposnse)
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"data": null}`))
		log.Print(e)
	} else {
		w.WriteHeader(http.StatusOK)
		w.Write(resp)
	}
}

func imageReager() {
	return
}

func main() {

	PORT := 4500
	http.HandleFunc("/", handler)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", PORT), nil))
}
