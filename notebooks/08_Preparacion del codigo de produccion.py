#!/usr/bin/env python
# coding: utf-8

# ## IMPORTAR LAS LIBRERIAS

# In[1]:


import numpy as np
import pandas as pd
import pickle

#Automcompletar rápido
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# ## CARGAR LOS DATOS

# ### Ruta del proyecto

# In[2]:


from credit_scoring.config import RAW_DATA_DIR

ruta_proyecto = RAW_DATA_DIR


# ### Nombre del fichero de datos

# In[4]:


data_file = 'credit_scoring.csv'


# ### Cargar los datos

# In[5]:


ruta_completa = ruta_proyecto / data_file

df = pd.read_csv(ruta_completa,index_col=0)


# ### Seleccionar las variables finales

# In[6]:


variables_finales = ['ingresos_verificados',
                     'vivienda',
                     'finalidad',
                     'num_cuotas',
                     'antigüedad_empleo',
                     'rating',
                     'ingresos',
                     'dti',
                     'num_lineas_credito',
                     'porc_uso_revolving',
                     'principal',
                     'tipo_interes',
                     'imp_cuota',
                     'num_derogatorios',
                     'estado',
                     'imp_amortizado',
                     'imp_recuperado'
                  ]


# ## ESTRUCTURA DE LOS DATASETS

# ### Eliminar registros

# #### Por atípicos

# In[7]:


a_eliminar = df.loc[df.ingresos > 300000].index.values


# In[8]:


df = df[~df.index.isin(a_eliminar)]


# ### Seleccionar variables

# Quedarse solo con las de la lista.

# In[9]:


df = df[variables_finales]


# ## CREAR EL PIPELINE

# ### Instanciar calidad de datos

# #### Crear la función

# In[10]:


def calidad_datos(temp):
    temp['antigüedad_empleo'] = temp['antigüedad_empleo'].fillna('desconocido')
    
    for column in temp.select_dtypes('number').columns:
        temp[column] = temp[column].fillna(0)
    
    return temp


# ### Instanciar creación de variables

# Dado que la creación de variables es diferente para los 3 modelos necesitamos construir 3 funciones.

# #### Crear las funciones

# In[11]:


def creacion_variables_pd(df):
    
    temp = df.copy()
    
    temp['target_pd'] = np.where(temp.estado.isin(['Charged Off','Does not meet the credit policy. Status:Charged Off','Default']), 1, 0)
    
    temp.vivienda = temp.vivienda.replace(['ANY','NONE','OTHER'],'MORTGAGE')
    
    temp.finalidad = temp.finalidad.replace(['wedding','educational','renewable_energy'],'otros')
    
    #Eliminamos las variables que ya no usaremos
    temp.drop(columns = ['estado','imp_amortizado','imp_recuperado'],inplace = True)
    
    #Separamos entre predictoras y target
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    
    return(temp_x,temp_y)


# In[12]:


def creacion_variables_ead(df):
    
    temp = df.copy()
    
    temp['pendiente'] = temp.principal - temp.imp_amortizado
    
    temp['target_ead'] = temp.pendiente / temp.principal
    
    temp.vivienda = temp.vivienda.replace(['ANY','NONE','OTHER'],'MORTGAGE')
    
    temp.finalidad = temp.finalidad.replace(['wedding','educational','renewable_energy'],'otros')
    
    #Eliminamos las variables que ya no usaremos
    temp.drop(columns = ['estado','imp_amortizado','imp_recuperado','pendiente'],inplace = True)
    
    #Separamos entre predictoras y target
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    
    return(temp_x,temp_y)


# In[13]:


def creacion_variables_lgd(df):
    
    temp = df.copy()
    
    temp['pendiente'] = temp.principal - temp.imp_amortizado
    
    temp['target_lgd'] = 1 - (temp.imp_recuperado / temp.pendiente)
    
    temp['target_lgd'].fillna(0,inplace=True)
    
    temp.vivienda = temp.vivienda.replace(['ANY','NONE','OTHER'],'MORTGAGE')
    
    temp.finalidad = temp.finalidad.replace(['wedding','educational','renewable_energy'],'otros')
    
    #Eliminamos las variables que ya no usaremos
    temp.drop(columns = ['estado','imp_amortizado','imp_recuperado','pendiente'],inplace = True)
    
    #Separamos entre predictoras y target
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    
    return(temp_x,temp_y)


# #### Crear los dataframes de X e y

# In[14]:


x_pd, y_pd = creacion_variables_pd(calidad_datos(df))

x_ead, y_ead = creacion_variables_ead(calidad_datos(df))

x_lgd, y_lgd = creacion_variables_lgd(calidad_datos(df))


# ### Instanciar transformación de variables

# In[15]:


#ONE HOT ENCODING
var_ohe = [ 'ingresos_verificados', 'vivienda','finalidad','num_cuotas']
ohe = OneHotEncoder(sparse = False, handle_unknown='ignore')


#ORDINAL ENCODING
var_oe = ['antigüedad_empleo','rating']

orden_antigüedad_empleo = ['desconocido','< 1 year','1 year','2 years','3 years','4 years',
                           '5 years','6 years','7 years','8 years','9 years','10+ years']

orden_rating = ['A','B','C','D','E','F','G']

oe = OrdinalEncoder(categories = [orden_antigüedad_empleo,orden_rating],
                    handle_unknown = 'use_encoded_value',
                    unknown_value = 12)

#BINNING
var_bin = ['num_derogatorios']
bina = Binarizer(threshold=0)


#MIN-MAX SCALING

mms = MinMaxScaler()


# ### Crear el pipe del preprocesamiento

# #### Crear el column transformer

# In[16]:


ct = make_column_transformer(
    (ohe, var_ohe),
    (oe, var_oe),
    (bina, var_bin),
    remainder='passthrough')


# #### Crear el pipeline de preprocesamiento

# In[17]:


pipe_prepo = make_pipeline(ct, mms)


# ### Instanciar los modelos

# #### Instanciar los algoritmos

# In[18]:


modelo_pd = LogisticRegression(solver='saga', n_jobs=-1, C=1, penalty='l1', random_state=42)

modelo_ead = HistGradientBoostingRegressor(learning_rate = 0.1,
                                          max_iter = 200,
                                          max_depth = 10,
                                          min_samples_leaf = 100,
                                          scoring = 'neg_mean_absolute_percentage_error',
                                          l2_regularization = 0.25,
                                          random_state=42)

modelo_lgd = HistGradientBoostingRegressor(learning_rate = 0.1,
                                          max_iter = 200,
                                          max_depth = 20,
                                          min_samples_leaf = 100,
                                          scoring = 'neg_mean_absolute_percentage_error',
                                          l2_regularization = 1,
                                          random_state=42)


# #### Crear los pipes finales de entrenamiento

# In[19]:


pipe_entrenamiento_pd = make_pipeline(pipe_prepo,modelo_pd)

pipe_entrenamiento_ead = make_pipeline(pipe_prepo,modelo_ead)

pipe_entrenamiento_lgd = make_pipeline(pipe_prepo,modelo_lgd)


# #### Guardar el pipe final de entrenamiento

# In[20]:


from credit_scoring.config import PROJ_ROOT


# In[21]:


ruta_pipe_entrenamiento_pd = PROJ_ROOT / 'models' / 'pipe_entrenamiento_pd.pickle'

with open(ruta_pipe_entrenamiento_pd, mode='wb') as file:
   pickle.dump(pipe_entrenamiento_pd, file)


# In[22]:


ruta_pipe_entrenamiento_ead = PROJ_ROOT / 'models' / 'pipe_entrenamiento_ead.pickle'

with open(ruta_pipe_entrenamiento_ead, mode='wb') as file:
   pickle.dump(pipe_entrenamiento_ead, file)


# In[23]:


ruta_pipe_entrenamiento_lgd = PROJ_ROOT / 'models' / 'pipe_entrenamiento_lgd.pickle'

with open(ruta_pipe_entrenamiento_lgd, mode='wb') as file:
   pickle.dump(pipe_entrenamiento_lgd, file)


# #### Entrenar los pipes

# In[24]:


pipe_ejecucion_pd = pipe_entrenamiento_pd.fit(x_pd,y_pd)
pipe_ejecucion_ead = pipe_entrenamiento_ead.fit(x_ead,y_ead)
pipe_ejecucion_lgd = pipe_entrenamiento_lgd.fit(x_lgd,y_lgd)


# ## GUARDAR EL PIPE

# ### Guardar el pipe final de ejecución

# In[25]:


ruta_pipe_ejecucion_pd = PROJ_ROOT / 'models' / 'pipe_ejecucion_pd.pickle'

with open(ruta_pipe_ejecucion_pd, mode='wb') as file:
   pickle.dump(pipe_ejecucion_pd, file)


# In[26]:


ruta_pipe_ejecucion_ead = PROJ_ROOT / 'models' / 'pipe_ejecucion_ead.pickle'

with open(ruta_pipe_ejecucion_ead, mode='wb') as file:
   pickle.dump(pipe_ejecucion_ead, file)


# In[27]:


ruta_pipe_ejecucion_lgd = PROJ_ROOT / 'models' / 'pipe_ejecucion_lgd.pickle'

with open(ruta_pipe_ejecucion_lgd, mode='wb') as file:
   pickle.dump(pipe_ejecucion_lgd, file)

