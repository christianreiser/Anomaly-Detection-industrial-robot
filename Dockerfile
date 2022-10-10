# https://medium.com/fintechexplained/running-python-in-docker-container-58cda726d574
# sudo docker build -t anomaly_detector-docker .
FROM python:3.9-slim-buster

WORKDIR /src

RUN python -m venv venv

COPY src/requirements.txt requirements.txt
RUN /src/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . /src

CMD ["/src/venv/bin/activate && python", "src/main.py" ]