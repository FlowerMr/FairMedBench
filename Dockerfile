FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "fairmedbench.cli", "demo", "--config", "configs/demo.yaml"]
