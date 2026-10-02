import json
import numpy as np
import pandas as pd
from credit_scoring.config import MODELS_DIR
import pickle

# Carga de modelos fuera del handler para reutilizar en ejecuciones cálidas (warm starts)
def cargar_modelo(nombre):
    ruta = MODELS_DIR / f"pipe_ejecucion_{nombre}.pickle"
    with open(ruta, "rb") as f:
        return pickle.load(f)

models = {
    "pd": cargar_modelo("pd"),
    "ead": cargar_modelo("ead"),
    "lgd": cargar_modelo("lgd")
}

def lambda_handler(event, context):
    try:
        # 1. Parsear el payload (por si viene de API Gateway o invocación directa)
        if "body" in event and isinstance(event["body"], str):
            payload = json.loads(event["body"])
        else:
            payload = event

        # 2. Mapear alias/tildes que espera el pipeline
        if "antiguedad_empleo" in payload:
            payload["antigüedad_empleo"] = payload.pop("antiguedad_empleo")

        # 3. Crear DataFrame
        df_cliente = pd.DataFrame([payload])

        # 4. Inferencia con los pipelines
        principal = float(payload.get("principal", 0.0))
        
        scoring_pd = float(models["pd"].predict_proba(df_cliente)[:, 1])
        ead_raw = float(models["ead"].predict(df_cliente))
        lgd_raw = float(models["lgd"].predict(df_cliente))

        ead = float(np.clip(ead_raw, 0.0, 1.0))
        lgd = float(np.clip(lgd_raw, 0.0, 1.0))
        perdida_esperada = round(scoring_pd * principal * ead * lgd, 2)

        # 5. Respuesta en formato API Gateway Proxy
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "pd": round(scoring_pd, 4),
                "ead": round(ead, 4),
                "lgd": round(lgd, 4),
                "perdida_esperada": perdida_esperada
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }