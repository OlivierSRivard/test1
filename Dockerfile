FROM python:3.11-slim

# Safer defaults for Python in containers
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# System deps (build tools only if needed by wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" PIP_DISABLE_PIP_VERSION_CHECK=1

# App workdir
WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Drop root for runtime
RUN useradd -m appuser && chown -R appuser:appuser /app /opt/venv
USER appuser

# Optional (local dev convenience)
EXPOSE 8501

# Start Streamlit
CMD ["sh","-lc","streamlit run Home.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
