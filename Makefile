default: all

all: up up-neo4j

# ==========
# interaction tasks
bash: up
	docker-compose exec app bash

python: up
	docker-compose exec app python

# ==========
# experiment tasks
up-neo4j:
	docker-compose up -d neo4j

down-neo4j:
	docker-compose stop neo4j
	docker-compose rm -f neo4j

bash-neo4j:
	docker-compose exec neo4j bash

# ==========
# frontend tasks
frontend-install frontend-init frontend-ci frontend-prod frontend-dev frontend-unit frontend-e2e : up
	$(eval task_name=$(shell echo "$@" | perl -pe 's/frontend-//'))
	@echo "runnning task @ frontend: $(task_name)"
	docker-compose exec app sudo service dbus start
	docker-compose exec app bash -c "cd frontend && make $(task_name)"

frontend-restore: frontend-ci

# ==========
# backend tasks
backend-webapi backend-test-unit backend-log-access backend-hello backend-post backend-test-request backend-mlflow-server backend-tensorboard: up
	$(eval task_name=$(shell echo "$@" | perl -pe 's/backend-//'))
	@echo "runnning task @ backend: $(task_name)"
	docker-compose exec app bash -c "cd backend && make $(task_name)"

# switch mode
cpu gpu:
	@rm -f docker-compose.yml
	@ln -s docker/docker-compose.$@.yml docker-compose.yml

# ==========
# docker-compose aliases
up:
	docker-compose up -d app

active:
	docker-compose up

ps images down:
	docker-compose $@

im:images

build:
	docker-compose build --no-cache

reup: down up

clean: clean-logs clean-container

clean-logs:
	rm -rf log/*.log

clean-app:
	docker-compose down app
	docker rmi app.experiment

clean-container:
	docker-compose down --rmi all
	sudo rm -rf app/__pycache__
