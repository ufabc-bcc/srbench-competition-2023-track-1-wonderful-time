FROM python:3.8.16-slim

COPY symmfea.tar.gz /tmp
RUN mkdir /tmp/symmfea
RUN tar -xzf /tmp/symmfea.tar.gz -C /tmp/symmfea
# RUN . /tmp/symmfea/bin/activate
SHELL ["/bin/bash", "-c", "source /tmp/symmfea/bin/activate"]
RUN conda-unpack

