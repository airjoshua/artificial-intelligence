ARG CUDA_VER=12.9.1
ARG PYTHON_VER=3.13
ARG LINUX_DISTRO=ubuntu
ARG LINUX_DISTRO_VER=24.04

FROM nvidia/cuda:${CUDA_VER}-runtime-${LINUX_DISTRO}${LINUX_DISTRO_VER}

ARG PYTHON_VER

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VER} \
    python${PYTHON_VER}-venv \
    wget \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python${PYTHON_VER} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VER} /usr/bin/python3

# Create rapids user
RUN groupadd -g 1001 rapids && \
    useradd -rm -d /home/rapids -s /bin/bash -g rapids -u 1001 rapids

USER rapids
WORKDIR /home/rapids

# Create and activate virtual environment
RUN python -m venv /home/rapids/venv
ENV PATH="/home/rapids/venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/rapids/venv"

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/home/rapids/.cargo/bin:$PATH"

# Copy requirements file
COPY --chmod=644 requirements.txt /home/rapids/requirements.txt

# Install dependencies using uv
RUN uv pip install -r requirements.txt

# Default shell
CMD ["bash"]
