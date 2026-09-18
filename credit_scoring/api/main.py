import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

# 1. Rutas
# Asumiendo que este script está en credit_scoring/credit_scoring/api/main.py
from credit_scoring.config import MODELS_DIR

# Diccionario global para almacenar los modelos en memoria
ml_models = {}

def cargar_modelo(nombre_modelo: str):
    ruta = MODELS_DIR / f"pipe_ejecucion_{nombre_modelo}.pickle"
    if not ruta.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
    with open(ruta, mode="rb") as file:
        return pickle.load(file)

# 2. Manejo de ciclo de vida moderno (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Cargar modelos al iniciar el servidor
    try:
        ml_models["pd"] = cargar_modelo("pd")
        ml_models["ead"] = cargar_modelo("ead")
        ml_models["lgd"] = cargar_modelo("lgd")
        print("Modelos cargados exitosamente en memoria.")
    except Exception as e:
        print(f"Error crítico cargando modelos: {e}")
    
    yield  # Aquí es donde la API está "viva" y recibiendo peticiones
    
    # Teardown: Limpiar memoria al apagar el servidor
    ml_models.clear()

# 3. Esquemas
class ClienteData(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    principal: float = Field(..., gt=0)
    dti: float = Field(..., ge=0)
    rating: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G']
    vivienda: Literal['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    ingresos_verificados: Literal['Verified', 'Source Verified', 'Not Verified']
    num_lineas_credito: int = Field(..., ge=0)
    porc_uso_revolving: float = Field(..., ge=0)
    num_derogatorios: int = Field(..., ge=0)
    finalidad: str
    tipo_interes: float = Field(..., ge=0)
    imp_cuota: float = Field(..., gt=0)
    
    # --- Agregados faltantes ---
    ingresos: float = Field(..., ge=0)
    num_cuotas: Literal[' 36 months', ' 60 months'] # Revisa si tu dataset original traía el espacio inicial (' 36 months')
    
    # ALIAS aplicado para coincidir con la pipeline de entrenamiento
    antiguedad_empleo: Optional[str] = Field(default="desconocido", alias="antigüedad_empleo")
    
class PredictResponse(BaseModel):
    pd: float
    ead: float
    lgd: float
    perdida_esperada: float

# 4. App FastAPI
app = FastAPI(
    title="Credit Scoring API",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
def root():
    """Redirige automáticamente la raíz a la documentación de Swagger."""
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silencia la petición del icono del navegador sin arrojar 404."""
    return Response(status_code=204)

@app.get("/ping")
def ping():
    return {"status": "ok", "models_loaded": len(ml_models) == 3}


@app.post("/predict", response_model=PredictResponse)
def predict(cliente: ClienteData):
    if len(ml_models) < 3:
        raise HTTPException(status_code=500, detail="Modelos no disponibles.")

    # by_alias=True asegura que el DataFrame se cree con "antigüedad_empleo"
    df_cliente = pd.DataFrame([cliente.model_dump(by_alias=True)])

    try:
        scoring_pd = float(ml_models["pd"].predict_proba(df_cliente)[:, 1])
        ead_raw = float(ml_models["ead"].predict(df_cliente))
        lgd_raw = float(ml_models["lgd"].predict(df_cliente))

        ead = float(np.clip(ead_raw, 0.0, 1.0))
        lgd = float(np.clip(lgd_raw, 0.0, 1.0))
        perdida_esperada = round(scoring_pd * cliente.principal * ead * lgd, 2)

        return PredictResponse(
            pd=round(scoring_pd, 4),
            ead=round(ead, 4),
            lgd=round(lgd, 4),
            perdida_esperada=perdida_esperada
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en inferencia: {str(e)}")