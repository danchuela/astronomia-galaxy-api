FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock /app/
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY . /app

RUN if [ -f /app/docker/zscaler-root-ca.crt ] && grep -q "BEGIN CERTIFICATE" /app/docker/zscaler-root-ca.crt 2>/dev/null; then \
      cp /app/docker/zscaler-root-ca.crt /usr/local/share/ca-certificates/zscaler-root-ca.crt && \
      update-ca-certificates; \
    fi

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
