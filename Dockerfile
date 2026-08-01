# Entorno reproducible para la evaluación de SLMs en interpretación de
# comandos de domótica en español rioplatense (CACIC 2026).
#
# Pensado para emular las condiciones de hardware del paper: CPU only,
# sin GPU. Correlo con límites explícitos de recursos para reproducir
# fielmente el entorno acotado del estudio (2 núcleos, ~8GB RAM):
#
#   docker build -t slm-domotica .
#   docker run --rm --memory=8g --cpus=2 \
#       -v $(pwd)/outputs:/app/outputs \
#       -v $(pwd)/.hf_cache:/app/.hf_cache \
#       slm-domotica
#
# Nota: la primera corrida necesita red para descargar los pesos de los
# cuatro modelos desde HuggingFace (varios GB en total). Las corridas
# siguientes pueden aislarse completamente de la red agregando
# --network=none, ya que los pesos quedan cacheados en .hf_cache/.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_HOME=/app/.hf_cache
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/run_evaluation.py"]
