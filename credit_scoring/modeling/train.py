from pathlib import Path
import pickle
import pandas as pd
from loguru import logger
import typer

from credit_scoring.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def cargar_datos_procesados(prefijo: str):
    """Carga X y Y desde data/processed/"""
    x = pd.read_pickle(PROCESSED_DATA_DIR / f'x_{prefijo}.pkl')
    y = pd.read_pickle(PROCESSED_DATA_DIR / f'y_{prefijo}.pkl')
    return x, y


@app.command()
def main():
    for modelo in ['pd', 'ead', 'lgd']:
        logger.info(f"--- Iniciando entrenamiento para {modelo.upper()} ---")
        
        # 1. Cargar datos
        x, y = cargar_datos_procesados(modelo)
        
        # 2. Cargar pipeline base de entrenamiento
        ruta_pipe_base = MODELS_DIR / f'pipe_entrenamiento_{modelo}.pickle'
        if not ruta_pipe_base.exists():
            logger.error(f"No se encontró el pipeline base en: {ruta_pipe_base}")
            continue

        with open(ruta_pipe_base, mode='rb') as file:
            pipe = pickle.load(file)
            
        # 3. Entrenar
        logger.info(f"Entrenando pipeline {modelo.upper()}...")
        pipe_ejecucion = pipe.fit(x, y)
        
        # 4. Guardar pipeline entrenado
        ruta_pipe_ejecucion = MODELS_DIR / f'pipe_ejecucion_{modelo}.pickle'
        with open(ruta_pipe_ejecucion, mode='wb') as file:
            pickle.dump(pipe_ejecucion, file)
            
        logger.success(f"Pipeline {modelo.upper()} guardado en {ruta_pipe_ejecucion}")


if __name__ == "__main__":
    app()