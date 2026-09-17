import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, Binarizer, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Variables globales basadas en el notebook
VARIABLES_FINALES = [
    'ingresos_verificados', 'vivienda', 'finalidad', 'num_cuotas', 'antigüedad_empleo',
    'rating', 'ingresos', 'dti', 'num_lineas_credito', 'porc_uso_revolving',
    'principal', 'tipo_interes', 'imp_cuota', 'num_derogatorios', 'estado',
    'imp_amortizado', 'imp_recuperado'
]

ORDEN_ANTIGUEDAD = [
    'desconocido', '< 1 year', '1 year', '2 years', '3 years', '4 years',
    '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
]

ORDEN_RATING = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def filtrar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra variables y elimina valores atípicos."""
    df_filtrado = df.copy()
    
    # Eliminar atípicos
    if 'ingresos' in df_filtrado.columns:
        a_eliminar = df_filtrado.loc[df_filtrado.ingresos > 300000].index.values
        df_filtrado = df_filtrado[~df_filtrado.index.isin(a_eliminar)]
        
    # Mantener solo las columnas esperadas que existan en el df
    cols_presentes = [col for col in VARIABLES_FINALES if col in df_filtrado.columns]
    return df_filtrado[cols_presentes]

def calidad_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa nulos y homologa categorías para train y predict."""
    df_limpio = df.copy()
    
    if 'antigüedad_empleo' in df_limpio.columns:
        df_limpio['antigüedad_empleo'] = df_limpio['antigüedad_empleo'].fillna('desconocido')
    
    # Imputar nulos en numéricas
    for column in df_limpio.select_dtypes('number').columns:
        df_limpio[column] = df_limpio[column].fillna(0)
        
    # Reemplazos en categóricas 
    if 'vivienda' in df_limpio.columns:
        df_limpio['vivienda'] = df_limpio['vivienda'].replace(['ANY', 'NONE', 'OTHER'], 'MORTGAGE')
    if 'finalidad' in df_limpio.columns:
        df_limpio['finalidad'] = df_limpio['finalidad'].replace(['wedding', 'educational', 'renewable_energy'], 'otros')
        
    return df_limpio

def preparar_datos_inferencia(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el dataset para `predict.py` eliminando variables target-related."""
    cols_a_eliminar = ['estado', 'imp_amortizado', 'imp_recuperado', 'pendiente']
    df_inferencia = df.copy()
    for col in cols_a_eliminar:
        if col in df_inferencia.columns:
            df_inferencia.drop(columns=[col], inplace=True)
    return df_inferencia

def crear_pipe_prepo():
    """Construye y retorna el ColumnTransformer de preprocesamiento."""
    var_ohe = ['ingresos_verificados', 'vivienda', 'finalidad', 'num_cuotas']
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    var_oe = ['antigüedad_empleo', 'rating']
    oe = OrdinalEncoder(
        categories=[ORDEN_ANTIGUEDAD, ORDEN_RATING],
        handle_unknown='use_encoded_value',
        unknown_value=12
    )
    
    var_bin = ['num_derogatorios']
    bina = Binarizer(threshold=0)
    
    mms = MinMaxScaler()
    
    ct = make_column_transformer(
        (ohe, var_ohe),
        (oe, var_oe),
        (bina, var_bin),
        remainder='passthrough'
    )
    
    return make_pipeline(ct, mms)