FROM ubuntu:noble

RUN apt-get update && apt-get install -y python3 python3.12-venv curl git plantuml && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install -r /tmp/requirements.txt