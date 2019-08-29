FROM python:3.7-slim-stretch AS build

WORKDIR /app

COPY . .

RUN python -m pip install --no-cache-dir -r requirements.txt

RUN python model.py

CMD [ "/bin/sh" ]