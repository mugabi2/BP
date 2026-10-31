FROM python:3.12-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

COPY requirements-collector.txt .
RUN pip install --no-cache-dir -r requirements-collector.txt \
    && python -m playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright

COPY collect_virtuals.py run_collector.py ./
COPY src ./src

RUN useradd --create-home --uid 10001 collector \
    && mkdir -p /app/auth/browser-profile /app/logs /app/data \
    && chown -R collector:collector /app

USER collector

CMD ["python", "run_collector.py"]
