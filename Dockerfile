FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .