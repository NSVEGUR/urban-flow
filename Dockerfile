# Reproduces the UrbanFlow pipelines in a clean environment.
#
# Usage:
#   docker build -t urban-flow .
#   docker run --rm -v "$(pwd)/app/results:/app/app/results" -v "$(pwd)/app/models:/app/app/models" urban-flow
#   docker run --rm urban-flow uv run app/eda/main.py
#   docker run --rm urban-flow uv run app/uncertainty/main.py
#   docker run --rm -p 8501:8501 urban-flow uv run streamlit run app/demo/streamlit_app.py --server.address 0.0.0.0

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "app/classic/main.py"]
