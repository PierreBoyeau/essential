# 12.8 is recommended by evo-model
# if this needs to change, make two containers 
# instead of just one.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    tmux \
    git \
    && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["/bin/bash"]
