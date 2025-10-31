FROM nvcr.io/nvidia/rapidsai/notebooks:25.10-cuda13-py3.13

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir uv && \
    uv pip install --system -r /tmp/requirements.txt

CMD ["tail", "-f", "/dev/null"]

