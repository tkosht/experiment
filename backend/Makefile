default: all

all: build up

prepare: install download

install:
	sh bin/morph/install_mecabdic.sh

download:
	sh bin/morph/dl.sh

run: run-aozora

run-aozora:
	sh bin/morph/run.sh --dataset=aozora
	mv log/run.log log/run.log.aozora

run-ldcc:
	sh bin/morph/run.sh --dataset=ldcc
	mv log/run.log log/run.log.ldcc

# ==========
# backend tasks
test-unit: 
	python -m pytest test

log-access:
	tail -0f log/access.log

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
