# The search API: python -m vector.service, CPU only.
FROM python:3.13-slim

WORKDIR /app
# CPU-only PyTorch: the default wheel bundles GPU libraries this image would never use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY vector vector
RUN pip install --no-cache-dir '.[embed,service]'

# Bake the model and the SciFact corpus into the image, so the container starts
# without network access and the first start only has to build the index.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
 && python -c "from vector.data.beir import load_scifact; load_scifact()"

ENV HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1
# The saved index lives here; mount a volume so it survives container restarts.
VOLUME /app/datasets/service
EXPOSE 8000
# /health is readiness: 503 while the first-start index build runs (about 9 min on 4 CPU
# cores), 200 after. /livez is liveness: 200 whenever the process answers.
HEALTHCHECK --interval=10s --timeout=3s --start-period=900s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
# Set VECTOR_API_KEYS (comma-separated) to require API keys; add --rate/--burst for rate limiting.
ENTRYPOINT ["python", "-m", "vector.service", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--rate", "20", "--burst", "40"]
