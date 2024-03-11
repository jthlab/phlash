FROM nvidia/cuda:12.3.2-base-ubuntu22.04

RUN apt-get update
RUN apt-get install -y python3.10 python3.10-venv python3-pip git
RUN pip3 install -U pip setuptools wheel pdm


COPY . /usr/src/phlash
WORKDIR /usr/src/phlash
RUN pdm install -d -G webui

EXPOSE 8888

CMD [ "pdm", "run", "jupyter", "notebook" ]
