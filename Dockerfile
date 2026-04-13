# 12.8 is recommended by evo-model
# if this needs to change, make two containers 
# instead of just one.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    tmux \
    git \
    curl \
    zstd \
    && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3
WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["/bin/bash"]
