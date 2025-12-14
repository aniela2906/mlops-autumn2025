package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func main() {

	ctx := context.Background()

	// Start Dagger client
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(log.Writer()))
	if err != nil {
		log.Fatalf("Failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	// Choose python image
	container := client.Container().From("python:3.11-slim")

	// Mount ROOT of project (not dagger folder!)
	source := client.Host().Directory("..")

	container = container.
		WithMountedDirectory("/app", source).
		WithWorkdir("/app").
		WithExec([]string{"pip", "install", "--no-cache-dir", "-r", "requirements.txt"}).
		WithExec([]string{"python", "-m", "src.pipeline.train"})

	// ---- EXPORT artifacts/ FROM CONTAINER ----
	artifacts := container.Directory("artifacts")
	_, err = artifacts.Export(ctx, "./artifacts")
	if err != nil {
		log.Fatalf("Failed to export artifacts directory: %v", err)
	}
	log.Println("Exported artifacts/ folder")

	// ---- EXPORT mlruns/ FROM CONTAINER ----
	mlruns := container.Directory("mlruns")
	_, err = mlruns.Export(ctx, "./mlruns")
	if err != nil {
		log.Printf("Warning: could not export mlruns/: %v", err)
	} else {
		log.Println("Exported mlruns/ folder")
	}

	log.Println("Pipeline executed successfully inside Dagger")
}
