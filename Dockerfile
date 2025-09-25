FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PORT=10000
EXPOSE 10000
CMD ["streamlit","run","Home.py","--server.port=${PORT}","--server.address=0.0.0.0"]


