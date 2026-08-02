#!/bin/bash
# Login to AWS ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 905418184838.dkr.ecr.ap-south-1.amazonaws.com

# Pull the latest image
docker pull 905418184838.dkr.ecr.ap-south-1.amazonaws.com/supi_ecr:latest

# Check if the container 'sentiment-app' is running
if [ "$(docker ps -q -f name=sentiment-app)" ]; then
    # Stop the running container
    docker stop sentiment-app
fi

# Check if the container 'sentiment-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=sentiment-app)" ]; then
    # Remove the container if it exists
    docker rm sentiment-app
fi

# Run a new container
docker run -d -p 80:8000 -e DAGSHUB_PAT=fdb68a077f605501b6104869e8094e514c8265e0 --name sentiment-app 905418184838.dkr.ecr.ap-south-1.amazonaws.com/supi_ecr:latest