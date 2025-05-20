FROM python:3.7-slim

RUN apt-get update &&\
    apt-get install -y --no-install-recommends build-essential && \ 
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install  \
    numpy==1.19.5 \
    pandas==1.1.5 \
    scikit-learn==0.23.2

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
