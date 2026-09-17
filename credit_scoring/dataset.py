from loguru import logger
import pandas as pd
import typer

from credit_scoring.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from credit_scoring.features import (
    calidad_datos,
    creacion_variables_ead,
    creacion_variables_lgd,
    creacion_variables_pd,
    filtrar_datos,
)

app = typer.Typer()


@app.command()
def main(input_filename: str = "credit_scoring.csv"):
    logger.info("Cargando datos crudos...")
    ruta_completa = RAW_DATA_DIR / input_filename
    if not ruta_completa.exists():
        logger.error(f"El archivo {ruta_completa} no existe.")
        return

    df = pd.read_csv(ruta_completa, index_col=0)

    logger.info("Aplicando limpieza y calidad de datos...")
    df_filtrado = filtrar_datos(df)
    df_limpio = calidad_datos(df_filtrado)

    logger.info("Generando variables objetivo (PD, EAD, LGD)...")
    x_pd, y_pd = creacion_variables_pd(df_limpio)
    x_ead, y_ead = creacion_variables_ead(df_limpio)
    x_lgd, y_lgd = creacion_variables_lgd(df_limpio)

    logger.info("Guardando matrices procesadas en data/processed/...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    x_pd.to_pickle(PROCESSED_DATA_DIR / "x_pd.pkl")
    y_pd.to_pickle(PROCESSED_DATA_DIR / "y_pd.pkl")

    x_ead.to_pickle(PROCESSED_DATA_DIR / "x_ead.pkl")
    y_ead.to_pickle(PROCESSED_DATA_DIR / "y_ead.pkl")

    x_lgd.to_pickle(PROCESSED_DATA_DIR / "x_lgd.pkl")
    y_lgd.to_pickle(PROCESSED_DATA_DIR / "y_lgd.pkl")

    logger.success("Procesamiento de datos completado.")


if __name__ == "__main__":
    app()
