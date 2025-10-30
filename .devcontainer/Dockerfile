FROM nvcr.io/nvidia/rapidsai/notebooks:25.10-cuda13-py3.13

WORKDIR /workspace

COPY .devcontainer/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir uv && \
    uv pip install --no-cache-dir --system -r /tmp/requirements.txt

CMD ["tail", "-f", "/dev/null"]

