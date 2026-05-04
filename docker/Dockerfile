FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY pyproject.toml /app/
RUN pip install --no-cache-dir $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")

COPY . /app

RUN if [ -f /app/docker/zscaler-root-ca.crt ] && grep -q "BEGIN CERTIFICATE" /app/docker/zscaler-root-ca.crt 2>/dev/null; then \
      cp /app/docker/zscaler-root-ca.crt /usr/local/share/ca-certificates/zscaler-root-ca.crt && \
      update-ca-certificates; \
    fi

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
