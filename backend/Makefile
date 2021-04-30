
# ==========
# backend tasks
webapi:
	uvicorn \
        --host=0.0.0.0 \
        --log-config=conf/logging.ini \
        --app-dir=. \
        webapi:app

test-unit: 
	python -m pytest test

log-access:
	tail -0f log/access.log

hello:
	@sh bin/request_hello.sh

post:
	@sh bin/request_post.sh

test-request:
	@sh bin/test_request.sh

# ==========
# mlflow tasks
mlflow-run:
	mlflow run --no-conda .

mlflow-ui:
	mlflow ui --host=0.0.0.0

mlflow-server:
    mlflow server --host=0.0.0.0 \
		--backend-store-uri sqlite:///result/mlflow.db \
		--default-artifact-root=mlruns

# ==========
# tensorboard tasks
tensorboard:
	$(eval logdir:=$(shell ls -trd result/* | tail -n 1))
	echo $(logdir)
	tensorboard --host=0.0.0.0 --logdir=$(logdir)
