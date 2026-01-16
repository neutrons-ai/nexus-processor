FROM python:3.12

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md .

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["nexus-processor"]


