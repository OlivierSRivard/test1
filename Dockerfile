FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl 
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip 
    && pip install --no-cache-dir -r requirements.txt
COPY . /app
# Let Render inject ; bind Streamlit to it and 0.0.0.0
CMD ["bash","-lc","streamlit run Home.py --server.port  --server.address 0.0.0.0"]
