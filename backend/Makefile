# backend tasks

# ==========
# environments
poetry-install:
	@sh bin/poetry_install.sh

poetry:
	@bash -i -c 'SHELL=/usr/bin/bash poetry shell'

# ==========
# # for app
demo:
	@sh bin/run_demo.sh

tool-maker:
	python -m app.llm_toolmaker.tool_maker --task-name=word_sorting

eval:
	python -m app.llm_toolmaker.eval --task-name=word_sorting

summary-semantic-memory:
	python -m app.semantic_kernel.component.semantic_memory

demo-semantic-memory:
	python -m app.semantic_kernel.executable.demo


# ==========
# # webapi tasks
webapi:
	uvicorn \
        --host=0.0.0.0 \
        --log-config=conf/logging.ini \
        --app-dir=. \
        webapi:app

test-unit ut:
	python -m pytest test

tail-log:
	@tail -0f log/app.log

hello-request:
	@sh bin/request_hello.sh

post-request:
	@sh bin/request_post.sh

test-request:
	@sh bin/test_request.sh
# ==========
# # docker task
bash:
	docker exec -it experiment.app bash

# ==========
# # mlflow tasks
mlflow mlflow-server:
	mlflow server --host=0.0.0.0 \
		--backend-store-uri sqlite:///data/mlflow.db \
		--default-artifact-root=mlruns

mlflow-ui:
	mlflow ui --host=0.0.0.0

# ==========
# # tensorboard tasks
tensorboard:
	$(eval logdir:=$(shell ls -trd result/* | egrep -v db | tail -n 1))
	echo $(logdir)
	tensorboard --host=0.0.0.0 --logdir=$(logdir)

# # GPU tasks
nvidia nvidia-smi:
	nvidia-smi \
	  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used \
	  --format=csv \
	  -l 1

# ==========
# clean tasks
clean-pycache:
	find . -name '__pycache__' | xargs rm -rf

clean-result:
	rm -rf result/*

clean-mlruns:
	sh bin/clean_mlruns.sh

clean-venv:
	@rm -rf .venv/
	@rm -f poetry.lock

_clean-experiment:	# if use, be carefull !!
	rm -rf mlruns data/mlflow.db

# clean: clean-result clean-mlruns clean-venv
clean: clean-result clean-venv
