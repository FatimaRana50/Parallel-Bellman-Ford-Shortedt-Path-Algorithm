FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    gcc \
    make \
    openssh-server \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && \
    ssh-keygen -t rsa -b 2048 -f /root/.ssh/id_rsa -N "" && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/authorized_keys

RUN echo "Host *" >> /root/.ssh/config && \
    echo "    StrictHostKeyChecking no" >> /root/.ssh/config && \
    echo "    UserKnownHostsFile /dev/null" >> /root/.ssh/config && \
    chmod 600 /root/.ssh/config

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' \
    /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' \
    /etc/ssh/sshd_config

WORKDIR /project

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
