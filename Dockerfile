# Industrial Knowledge Intelligence - Hybrid GraphRAG
# docker build -t ikig .
# docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... ikig
FROM python:3.11-slim

# tesseract powers the OCR fallback for the scanned-PDF corpus
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so first start doesn't download 130MB
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

# Build the vector store + graph at image build time (no API key needed),
# so the container answers its first query immediately.
RUN python -c "from ingest.vector_builder import load_or_build_vector_store; from ingest.graph_builder import load_or_build_graph; load_or_build_vector_store(); load_or_build_graph()"

ENV PYTHONUNBUFFERED=1
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
