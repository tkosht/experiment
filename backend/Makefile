
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
mlflow-server:
	mlflow server --host=0.0.0.0 \
		--backend-store-uri sqlite:///result/mlflow.db \
		--default-artifact-root=mlruns

mlflow-ui:
	mlflow ui --host=0.0.0.0

# ==========
# tensorboard tasks
tensorboard:
	$(eval logdir:=$(shell ls -trd result/* | egrep -v db | tail -n 1))
	echo $(logdir)
	tensorboard --host=0.0.0.0 --logdir=$(logdir)


nvidia nvidia-smi:
	nvidia-smi \
	  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used \
	  --format=csv \
	  -l 1


clean-experiment:
	rm -rf mlruns result/*