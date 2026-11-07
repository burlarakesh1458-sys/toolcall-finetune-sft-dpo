# CPU image that reproduces the small-scale pipeline (baseline -> LoRA SFT -> DPO -> report).
#   docker build -t toolcall-ft .
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/outputs:/app/outputs" toolcall-ft
# Base weights download from the Hugging Face Hub on first run (mount a cache at /app/.cache/hf to keep them).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/hf PYTHONPATH=/app/src TOKENIZERS_PARALLELISM=false
WORKDIR /app

RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data
RUN pip install --no-deps . \
 && useradd -m trainer && mkdir -p outputs results .cache/hf && chown -R trainer /app
USER trainer

CMD ["bash", "scripts/run_pipeline.sh", "configs/cpu_small.yaml"]
