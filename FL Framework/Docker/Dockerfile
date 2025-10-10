FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    tcpdump \
    xterm \
    python3 \
    python3-pip \
    net-tools \
    iputils-ping \
    iproute2 \
    traceroute \
    nano \
    openssh-client \
    software-properties-common \
    curl \
    ethtool \
    build-essential

WORKDIR /tmp
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


WORKDIR /opt

RUN apt-get install -y \
    wget \
    gnupg2 \
    lsb-release

RUN wget https://www.emqx.com/en/downloads/enterprise/5.10.0/emqx-enterprise-5.10.0-ubuntu22.04-amd64.deb \
    && apt install -y ./emqx-enterprise-5.10.0-ubuntu22.04-amd64.deb \
    && rm emqx-enterprise-5.10.0-ubuntu22.04-amd64.deb

COPY ../emqx-base.hocon /etc/emqx/base.hocon

WORKDIR /usr/src/app

RUN add-apt-repository ppa:mosquitto-dev/mosquitto-ppa && \
    apt-get -y install mosquitto mosquitto-clients

COPY ../mosquitto.conf /etc/mosquitto/mosquitto.conf

COPY ../FedOpt ./FedOpt

WORKDIR /usr/src/app/FedOpt/src
