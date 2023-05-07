ECR_REPO=$(shell git config --file config.txt --get aws.ecr-repo)
IMAGE_URI=$(shell git config --file config.txt --get aws.image-uri)
IMAGE=$(ECR_REPO):latest
IMAGE_ID=$(shell docker images -q $(IMAGE))

help:
	@echo " - create-ecr-repo	: Creates repo in ECR with name specified in Makefile"
	@echo " - build-container	: Build container using Dockerfile for Sagemaker processing"
	@echo " - tag-image     	: Tag the last built image to latest version"
	@echo " - push-image     	: Push latest tagged image to ECR"
	@echo " - all			    : build-tag-push"
	@echo " - run-container 	: Run the last built image interactively"


create-ecr-repo:
	aws ecr create-repository --repository-name $(ECR_REPO) --region $(AWS_DEFAULT_REGION)


build-container:
	docker build -t $(IMAGE) -f Dockerfile .


run-container:
	docker run -it $(IMAGE_ID)


tag-image:
	docker tag $(IMAGE_ID) $(IMAGE_URI)


push-image:
	aws ecr get-login-password --region $(AWS_DEFAULT_REGION) | docker login --username AWS --password-stdin $(IMAGE_URI);docker push $(IMAGE_URI)


all: build-container tag-image push-image

