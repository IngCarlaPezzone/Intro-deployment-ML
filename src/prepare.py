from dvc import api
import pandas as pd
from io import StringIO # para ller los archivo de DVC
import sys
import logging

from pandas.core.tools import numeric

# Se configura el sistema de registro (logging) en Python
# Se establece cómo se deben registrar los mensajes de registro en tu aplicación.
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

# Se instancia el logging
logger = logging.getLogger(__name__)

# Muestra un mensaje de información durante la ejecución de la aplicación
logging.info('Fetching data...')

# Se guardan en variables los path a los archivos de datos asociados a dvc
# Se guardan como strings
movie_data_path = api.read('dataset/movies.csv', remote='dataset-track', encoding='utf-8')
finantial_data_path = api.read('dataset/finantials.csv', remote='dataset-track', encoding='utf-8')
opening_data_path = api.read('dataset/opening_gross.csv', remote='dataset-track', encoding='utf-8')

# Transformamos los strings en dataframes
fin_data = pd.read_csv(StringIO(finantial_data_path))
movie_data = pd.read_csv(StringIO(movie_data_path))
opening_data = pd.read_csv(StringIO(opening_data_path))

# Se seleccionan las columnas numéricas (de tipo float o int) junto con la columna 'movie_title' 
# del DataFrame movie_data y crea un nuevo DataFrame
numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
movie_data = movie_data[numeric_columns+['movie_title']]

# Deja las columnas que le interesa de fin_data
fin_data = fin_data[['movie_title', 'production_budget', 'worldwide_gross']]

# Genera un solo dataframe
fin_movie_data = pd.merge(fin_data, movie_data, on='movie_title', how='left')
full_movie_data = pd.merge(opening_data, fin_movie_data, on='movie_title', how='left')

# Borra dos columnas
full_movie_data = full_movie_data.drop(['gross','movie_title'], axis=1)

# Guardamos en un csv
full_movie_data.to_csv('dataset/full_data.csv',index=False)

# Muestra un mensaje de datos ya preparados
logger.info('Data Fetched and prepared...')