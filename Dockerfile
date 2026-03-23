FROM python:3.11-slim

# Metadados
LABEL description="API de Predição de Salários — UC Aprendizado de Máquina 2026/1"

# Evitar prompts interativos durante build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Copiar requirements primeiro (aproveita cache do Docker)
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Expor porta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/modelo')" || exit 1

# Comando de inicialização
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
