# 12.8 is recommended by evo-model
# if this needs to change, make two containers
# instead of just one.
FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    python-is-python3 \
    tmux \
    git \
    curl \
    zstd \
    && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://claude.ai/install.sh | bash

WORKDIR /workspace

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN python -m venv /opt/boltz && \
    /opt/boltz/bin/pip install -U pip && \
    /opt/boltz/bin/pip install -U torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128 && \
    /opt/boltz/bin/pip install --no-cache-dir -U "boltz[cuda]" && \
    ln -sf /opt/boltz/bin/boltz /usr/local/bin/boltz

CMD ["/bin/bash"]
