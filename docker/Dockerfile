# FROM python:3.12.2
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as build

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp

# Install python
RUN apt update && apt install -y \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev wget

# Download different versions here https://www.python.org/downloads/source/
RUN wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz

RUN tar -xf Python-3.12.2.tgz

WORKDIR /tmp/Python-3.12.2

RUN ./configure --enable-optimizations

RUN make install

RUN python3 --version


FROM build as app

WORKDIR /app

COPY pyproject.toml poetry.lock /app/
RUN pip3 install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY ./ /app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --workers 1 --host 0.0.0.0 --port 8080"]
