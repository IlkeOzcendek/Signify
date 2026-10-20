#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -x .venv/bin/python ]]; then
  signify_python=.venv/bin/python
else
  signify_python=python3
fi

exec "$signify_python" -m streamlit run app.py --server.address 127.0.0.1 "$@"
