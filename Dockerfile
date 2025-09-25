FROM python:3.11-slim

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY . /app

# Run Streamlit; PORT is provided by hosting (defaults to 8501 locally)
CMD ["sh","-lc","streamlit run Home.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
