FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-11.8 \
    DEBCONF_NOWARNINGS=yes
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y \
        poppler-utils \
        libpoppler-dev \
        wget \
        curl \
        git \
        python3-dev \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

ADD . /app/

RUN pip install -r requirements.txt
RUN pip install \
        "git+https://github.com/facebookresearch/detectron2.git"

RUN git clone https://github.com/microsoft/unilm.git /unilm \
        && sed -i 's/from collections import Iterable/from collections.abc import Iterable/' \
        /unilm/dit/object_detection/ditod/table_evaluation/data_structure.py



ENTRYPOINT [ "cd /app && python3 server.py" ]