FROM docker.m.daocloud.io/library/python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/webapp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r /tmp/requirements.txt

COPY src/ /app/src/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"
EXPOSE 5000

CMD ["python", "src/webapp/server.py"]
