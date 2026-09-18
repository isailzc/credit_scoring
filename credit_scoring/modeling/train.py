import pickle
from loguru import logger
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from credit_scoring.config import MODELS_DIR, PROCESSED_DATA_DIR
from credit_scoring.features import crear_pipe_prepo

app = typer.Typer()

def cargar_datos_procesados(prefijo: str):
    """Carga X y Y desde data/processed/"""
    x = pd.read_pickle(PROCESSED_DATA_DIR / f"x_{prefijo}.pkl")
    y = pd.read_pickle(PROCESSED_DATA_DIR / f"y_{prefijo}.pkl")
    return x, y

def obtener_modelo(nombre_modelo: str):
    """Retorna el estimador correspondiente."""
    if nombre_modelo == "pd":
        return LogisticRegression(solver='saga', n_jobs=-1, C=1, penalty='l1', random_state=42)
    elif nombre_modelo == "ead":
        return HistGradientBoostingRegressor(
            learning_rate=0.1, max_iter=200, max_depth=10, min_samples_leaf=100,
            scoring='neg_mean_absolute_percentage_error', l2_regularization=0.25, random_state=42
        )
    elif nombre_modelo == "lgd":
        return HistGradientBoostingRegressor(
            learning_rate=0.1, max_iter=200, max_depth=20, min_samples_leaf=100,
            scoring='neg_mean_absolute_percentage_error', l2_regularization=1, random_state=42
        )
    else:
        raise ValueError(f"Modelo {nombre_modelo} no soportado.")

@app.command()
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for modelo_tag in ["pd", "ead", "lgd"]:
        logger.info(f"--- Iniciando entrenamiento para {modelo_tag.upper()} ---")

        # 1. Cargar datos
        x, y = cargar_datos_procesados(modelo_tag)

        # 2. Construir pipeline completo
        logger.info(f"Construyendo pipeline {modelo_tag.upper()}...")
        estimador = obtener_modelo(modelo_tag)
        pipe = make_pipeline(crear_pipe_prepo(), estimador)

        # 3. Entrenar
        logger.info(f"Entrenando pipeline {modelo_tag.upper()}...")
        pipe_ejecucion = pipe.fit(x, y)

        # 4. Guardar pipeline entrenado
        ruta_pipe_ejecucion = MODELS_DIR / f"pipe_ejecucion_{modelo_tag}.pickle"
        with open(ruta_pipe_ejecucion, mode="wb") as file:
            pickle.dump(pipe_ejecucion, file)

        logger.success(f"Pipeline {modelo_tag.upper()} guardado en {ruta_pipe_ejecucion}")

if __name__ == "__main__":
    app()