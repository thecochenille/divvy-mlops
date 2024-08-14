FROM python:3.11-slim-buster AS builder

EXPOSE 8080

WORKDIR /app

RUN pip install -U pip
RUN pip install pipenv 

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY main.py ./
COPY data/* ./data/

COPY models ./models

ENTRYPOINT ["pipenv", "run", "streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]