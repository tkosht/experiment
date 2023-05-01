FROM python:3.9
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	unzip

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

COPY ./requirements.txt ./
COPY ./bat01.py ./

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "bat01.py"]
