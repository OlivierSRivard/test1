FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && python -c "import streamlit, sys; print('streamlit', streamlit.__version__)"

# app code + entrypoint
COPY . /app
RUN chmod +x /app/entrypoint.sh

# non-root
RUN useradd -m appuser && chown -R appuser:appuser /app /opt/venv
ENV HOME=/home/appuser
RUN mkdir -p "$HOME/.streamlit" && chown -R appuser:appuser "$HOME"
USER appuser

EXPOSE 8501

# force start through venv python regardless of Render's defaults
ENTRYPOINT ["/app/entrypoint.sh"]
