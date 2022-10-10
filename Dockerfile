# https://medium.com/fintechexplained/running-python-in-docker-container-58cda726d574
# sudo docker build --network=host -t anomaly_detector-docker .
FROM python:3.9-slim-buster

WORKDIR /src

COPY src/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /src

CMD ["python", "src/main.py" ]