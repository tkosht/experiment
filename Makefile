default: all

all: up

# ==========
# interaction tasks
bash:
	docker compose exec app bash

python: up
	docker compose exec app python

# switch mode
cpu gpu:
	@rm -f compose.yml
	@ln -s docker/compose.$@.yml compose.yml

mode:
	@echo $$(ls -l compose.yml | awk -F. '{print $$(NF-1)}')


# ==========
# general tasks
pip: _pip commit

_pip:
	docker compose exec app python -m pip install -r requirements.txt

commit:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	docker commit experiment.app experiment.app:latest
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

save: commit
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	docker save experiment.app:latest | gzip > data/experiment.app.tar.gz
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

load:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	docker load < data/experiment.app.tar.gz
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

# ==========
# docker compose aliases
up: _up ssh

_up:
	docker compose up -d app

ssh:
	docker compose exec app sudo service ssh start

active:
	docker compose up

ps images down:
	docker compose $@

im:images

build:
	docker compose build

build-no-cache:
	docker compose build --no-cache

reup: down up

clean: clean-logs clean-container

clean-logs:
	rm -rf log/*.log

clean-container:
	docker compose down --rmi all
	sudo rm -rf app/__pycache__

# ==========
# frontend tasks
frontend-install frontend-init frontend-ci frontend-prod frontend-dev frontend-unit frontend-e2e : up
	$(eval task_name=$(shell echo "$@" | perl -pe 's/frontend-//'))
	@echo "runnning task @ frontend: $(task_name)"
	docker compose exec app sudo service dbus start
	docker compose exec app bash -c "cd frontend && make $(task_name)"

frontend-restore: frontend-ci

# ==========
# backend tasks
backend-webapi backend-test-unit backend-log-access backend-hello backend-post backend-test-request backend-mlflow-server backend-tensorboard: up
	$(eval task_name=$(shell echo "$@" | perl -pe 's/backend-//'))
	@echo "runnning task @ backend: $(task_name)"
	docker compose exec app bash -c "cd backend && make $(task_name)"

