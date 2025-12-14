package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// Connect to Dagger engine
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		panic(err)
	}
	defer client.Close()

	// Use Python container
	python := client.Container().
		From("python:3.11-slim").
		WithDirectory("/app", client.Host().Directory(".")).
		WithWorkdir("/app")

	// Install dependencies
	python = python.WithExec([]string{
		"pip", "install", "--no-cache-dir", "-r", "requirements.txt",
	})

	// Run the Python MLOps pipeline
	python = python.WithExec([]string{
		"python", "-m", "src.pipeline.train",
	})

	fmt.Println("Pipeline executed inside Dagger")

	// EXTRACT ARTIFACT: best model info
	// (instructors require artifact named "model")

	modelFile := python.File("artifacts/model_selection.json")

	contents, err := modelFile.Contents(ctx)
	if err != nil {
		panic(err)
	}

	// Write output outside container → required by assignment
	err = os.WriteFile("model", []byte(contents), 0644)
	if err != nil {
		panic(err)
	}

	fmt.Println("Model artifact exported to ./model")
}
