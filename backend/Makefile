
# ==========
# backend tasks
# # train models

wikidata:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	python -m app.auto_topic.executable.make_wikidata
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

wordvector:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	python -m app.auto_topic.executable.train_wordvector
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

tuned-wordvector:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	python -m app.auto_topic.executable.tune_wordvector
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

topicmodel:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	python -m app.auto_topic.executable.train_topicmodel
	@echo "$$(date +'%Y/%m/%d %T') - End $@"


# -----
# # crawling tasks
news:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	sh app/scrapy/bin/crawl.sh
	@echo "$$(date +'%Y/%m/%d %T') - End $@"


# # webapi
flask:
	FLASK_APP=flask_app.py flask run --host=0.0.0.0 --port=8000

webapi:
	uwsgi conf/uwsgi.ini 

# webapi:
# 	uvicorn \
#         --host=0.0.0.0 \
#         --log-config=conf/logging.ini \
#         --app-dir=. \
#         webapi:app

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


nvidia nvidia-smi:
	nvidia-smi \
	  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used \
	  --format=csv \
	  -l 1


