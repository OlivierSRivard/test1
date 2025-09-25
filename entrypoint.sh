#!/usr/bin/env sh
set -e

# optional: materialize secrets.toml from env vars if you want st.secrets to work
if [ -n "${BASIC_USER:-}" ] || [ -n "${BASIC_PASSWORD:-}" ]; then
  mkdir -p "$HOME/.streamlit"
  cat > "$HOME/.streamlit/secrets.toml" <<EOF
BASIC_USER = "${BASIC_USER:-}"
BASIC_PASSWORD = "${BASIC_PASSWORD:-}"
EOF
fi

exec /opt/venv/bin/python -m streamlit run Home.py --server.port "${PORT:-8501}" --server.address "0.0.0.0"
