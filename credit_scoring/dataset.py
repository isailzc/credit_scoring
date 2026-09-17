import numpy as np
import pandas as pd
from loguru import logger
from credit_scoring.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from credit_scoring.features import filtrar_datos, calidad_datos

def main():
    logger.info("Cargando datos crudos...")
    df = pd.read_csv(RAW_DATA_DIR / 'credit_scoring.csv', index_col=0)
    
    logger.info("Aplicando filtros y calidad de datos...")
    df = filtrar_datos(df)
    df = calidad_datos(df)
    
    # --- CREACIÓN DE TARGETS ---
    logger.info("Generando datasets para PD, EAD y LGD...")
    
    # 1. Dataset PD
    df_pd = df.copy()
    df_pd['target_pd'] = np.where(df_pd.estado.isin(['Charged Off', 'Does not meet the credit policy. Status:Charged Off', 'Default']), 1, 0)
    df_pd.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado'], inplace=True)
    x_pd, y_pd = df_pd.iloc[:, :-1], df_pd.iloc[:, -1]
    
    # 2. Dataset EAD
    df_ead = df.copy()
    df_ead['pendiente'] = df_ead.principal - df_ead.imp_amortizado
    df_ead['target_ead'] = df_ead.pendiente / df_ead.principal
    df_ead.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado', 'pendiente'], inplace=True)
    x_ead, y_ead = df_ead.iloc[:, :-1], df_ead.iloc[:, -1]
    
    # 3. Dataset LGD
    df_lgd = df.copy()
    df_lgd['pendiente'] = df_lgd.principal - df_lgd.imp_amortizado
    df_lgd['target_lgd'] = 1 - (df_lgd.imp_recuperado / df_lgd.pendiente)
    df_lgd['target_lgd'] = df_lgd['target_lgd'].fillna(0)
    df_lgd.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado', 'pendiente'], inplace=True)
    x_lgd, y_lgd = df_lgd.iloc[:, :-1], df_lgd.iloc[:, -1]

    # Guardar en procesados
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    pd.to_pickle(x_pd, PROCESSED_DATA_DIR / "x_pd.pkl")
    pd.to_pickle(y_pd, PROCESSED_DATA_DIR / "y_pd.pkl")
    pd.to_pickle(x_ead, PROCESSED_DATA_DIR / "x_ead.pkl")
    pd.to_pickle(y_ead, PROCESSED_DATA_DIR / "y_ead.pkl")
    pd.to_pickle(x_lgd, PROCESSED_DATA_DIR / "x_lgd.pkl")
    pd.to_pickle(y_lgd, PROCESSED_DATA_DIR / "y_lgd.pkl")
    
    logger.success("Datasets generados correctamente en data/processed/")

if __name__ == "__main__":
    main()