from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# VARIABLES Y REGISTROS FINALES
VARIABLES_FINALES = [
    'ingresos_verificados', 'vivienda', 'finalidad', 'num_cuotas',
    'antigüedad_empleo', 'rating', 'ingresos', 'dti', 'num_lineas_credito',
    'porc_uso_revolving', 'principal', 'tipo_interes', 'imp_cuota',
    'num_derogatorios', 'estado', 'imp_amortizado', 'imp_recuperado'
]

VARIABLES_PREDICTORAS = [
    'ingresos_verificados', 'vivienda', 'finalidad', 'num_cuotas',
    'antigüedad_empleo', 'rating', 'ingresos', 'dti', 'num_lineas_credito',
    'porc_uso_revolving', 'principal', 'tipo_interes', 'imp_cuota',
    'num_derogatorios'
]


def filtrar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra outliers y selecciona variables finales."""
    a_eliminar = df.loc[df.ingresos > 300000].index.values
    df_filtrado = df[~df.index.isin(a_eliminar)].copy()
    
    # Si contiene columnas de target (entrenamiento), filtra con VARIABLES_FINALES
    if 'estado' in df_filtrado.columns:
        return df_filtrado[VARIABLES_FINALES]
    return df_filtrado[VARIABLES_PREDICTORAS]


def calidad_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica imputación básica para valores nulos."""
    temp = df.copy()
    if 'antigüedad_empleo' in temp.columns:
        temp['antigüedad_empleo'] = temp['antigüedad_empleo'].fillna('desconocido')
    
    for column in temp.select_dtypes('number').columns:
        temp[column] = temp[column].fillna(0)
    return temp


def creacion_variables_pd(df: pd.DataFrame):
    """Genera X y Y para el modelo de Probabilidad de Default (PD)."""
    temp = df.copy()
    temp['target_pd'] = np.where(
        temp.estado.isin(['Charged Off', 'Does not meet the credit policy. Status:Charged Off', 'Default']), 
        1, 0
    )
    temp.vivienda = temp.vivienda.replace(['ANY', 'NONE', 'OTHER'], 'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding', 'educational', 'renewable_energy'], 'otros')
    temp.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado'], inplace=True)
    
    temp_x = temp.iloc[:, :-1]
    temp_y = temp.iloc[:, -1]
    return temp_x, temp_y


def creacion_variables_ead(df: pd.DataFrame):
    """Genera X y Y para el modelo de Exposure at Default (EAD)."""
    temp = df.copy()
    temp['pendiente'] = temp.principal - temp.imp_amortizado
    temp['target_ead'] = temp.pendiente / temp.principal
    temp.vivienda = temp.vivienda.replace(['ANY', 'NONE', 'OTHER'], 'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding', 'educational', 'renewable_energy'], 'otros')
    temp.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado', 'pendiente'], inplace=True)
    
    temp_x = temp.iloc[:, :-1]
    temp_y = temp.iloc[:, -1]
    return temp_x, temp_y


def creacion_variables_lgd(df: pd.DataFrame):
    """Genera X y Y para el modelo de Loss Given Default (LGD)."""
    temp = df.copy()
    temp['pendiente'] = temp.principal - temp.imp_amortizado
    temp['target_lgd'] = 1 - (temp.imp_recuperado / temp.pendiente)
    temp['target_lgd'] = temp['target_lgd'].fillna(0)
    temp.vivienda = temp.vivienda.replace(['ANY', 'NONE', 'OTHER'], 'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding', 'educational', 'renewable_energy'], 'otros')
    temp.drop(columns=['estado', 'imp_amortizado', 'imp_recuperado', 'pendiente'], inplace=True)
    
    temp_x = temp.iloc[:, :-1]
    temp_y = temp.iloc[:, -1]
    return temp_x, temp_y


def preparar_datos_inferencia(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la transformación de categorías para datos de validación/predicción."""
    temp = df.copy()
    temp.vivienda = temp.vivienda.replace(['ANY', 'NONE', 'OTHER'], 'MORTGAGE')
    temp.finalidad = temp.finalidad.replace(['wedding', 'educational', 'renewable_energy'], 'otros')
    return temp