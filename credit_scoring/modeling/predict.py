from pathlib import Path
import pickle
import numpy as np

from loguru import logger
import pandas as pd
import typer

from credit_scoring.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from credit_scoring.features import calidad_datos, filtrar_datos, preparar_datos_inferencia

app = typer.Typer()


def cargar_modelo(modelo: str):
    """Carga un pipeline entrenado de la carpeta models/"""
    ruta = MODELS_DIR / f"pipe_ejecucion_{modelo}.pickle"
    if not ruta.exists():
        raise FileNotFoundError(
            f"El modelo {ruta} no existe. Ejecuta 'python credit_scoring/train.py' primero."
        )

    with open(ruta, mode="rb") as file:
        return pickle.load(file)


@app.command()
def main(
    input_file: Path = RAW_DATA_DIR / "validacion.csv",
    output_file: Path = PROCESSED_DATA_DIR / "predictions.csv",
):
    """Ejecuta inferencia en un dataset nuevo y guarda la Pérdida Esperada."""
    if not input_file.exists():
        logger.error(f"Archivo de entrada no encontrado en: {input_file}")
        return

    logger.info(f"Cargando datos para inferencia desde {input_file}...")
    df = pd.read_csv(input_file, index_col=0)

    logger.info("Aplicando transformaciones de inferencia...")
    df_filtrado = filtrar_datos(df)
    df_limpio = calidad_datos(df_filtrado)
    x_inferencia = preparar_datos_inferencia(df_limpio)

    logger.info("Cargando modelos entrenados...")
    pipe_pd = cargar_modelo("pd")
    pipe_ead = cargar_modelo("ead")
    pipe_lgd = cargar_modelo("lgd")

    logger.info("Generando predicciones de PD, EAD y LGD...")
    scoring_pd = pipe_pd.predict_proba(x_inferencia)[:, 1]
    ead_raw = pipe_ead.predict(x_inferencia)
    lgd_raw = pipe_lgd.predict(x_inferencia)

    ead = np.clip(ead_raw, 0, 1)
    lgd = np.clip(lgd_raw, 0, 1)

    logger.info("Calculando Pérdida Esperada (Expected Loss)...")
    df_resultados = pd.DataFrame(
        {"principal": x_inferencia["principal"], "pd": scoring_pd, "ead": ead, "lgd": lgd},
        index=x_inferencia.index,
    )

    df_resultados["perdida_esperada"] = round(
        df_resultados["pd"]
        * df_resultados["principal"]
        * df_resultados["ead"]
        * df_resultados["lgd"],
        2,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(output_file)
    logger.success(f"Predicciones guardadas exitosamente en {output_file}")


if __name__ == "__main__":
    app()
