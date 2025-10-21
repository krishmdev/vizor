FROM ghcr.io/astral-sh/uv:0.9.28 AS uv

FROM python:3.11-slim
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH \
    VIZOR_ROOT=/app HF_HOME=/app/.models PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ml --extra ui --no-install-project
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY experiments ./experiments
COPY models.lock ./
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev --extra ml --extra ui
# Model weights are not baked in: mount the verified .models/ from `make models` read-only.
EXPOSE 8000 8501
ENTRYPOINT ["vizor"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
