FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Agregar el entorno virtual de uv al PATH global del contenedor
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Instalar uv
RUN pip install --no-cache-dir uv

# Copiar archivos de dependencias
COPY pyproject.toml uv.lock ./

# Copiar el paquete de Python y los modelos
COPY credit_scoring ./credit_scoring
COPY models ./models

# Instalar solo las dependencias en .venv
RUN uv sync --frozen --no-install-project

# Puerto de la API
EXPOSE 9696

# Arrancar uvicorn DIRECTAMENTE sin pasar por "uv run"
CMD ["uvicorn", "credit_scoring.api.main:app", "--host", "0.0.0.0", "--port", "9696"]