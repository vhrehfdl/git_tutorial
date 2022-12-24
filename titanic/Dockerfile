FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install -y git python3.7 python3.7-dev python3-pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2
RUN python -m pip install pip --upgrade
RUN pip install --upgrade setuptools
RUN pip install --upgrade wheel

RUN pip install pandas==1.3.5
RUN pip install scikit-learn==1.0.2

COPY . .

CMD ["python", "train.py"]
# CMD tail -f /dev/null