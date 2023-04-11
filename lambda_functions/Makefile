REPO_LAMBDA=lambda-cdk
IMAGE_LAMBDA=$(REPO_LAMBDA):latest
IMAGE_ID_LAMBDA=$(shell docker images -q $(IMAGE_LAMBDA))

build-container-lambda:
	docker build -t $(IMAGE_LAMBDA) -f Dockerfile .

tag-image:
	docker tag $(IMAGE_ID) $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_DEFAULT_REGION).amazonaws.com/$(IMAGE)


tag-image-lambda:
	docker tag $(IMAGE_ID_LAMBDA) $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_DEFAULT_REGION).amazonaws.com/$(IMAGE_LAMBDA)

push-image-lambda:
	aws ecr get-login-password --region $(AWS_DEFAULT_REGION) | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_DEFAULT_REGION).amazonaws.com/$(IMAGE_LAMBDA);docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_DEFAULT_REGION).amazonaws.com/$(IMAGE_LAMBDA)

all-lambda: build-container-lambda tag-image-lambda push-image-lambda