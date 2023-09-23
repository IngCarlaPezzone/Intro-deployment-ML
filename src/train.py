from utils import update_model, save_simple_metrics_report, get_model_performance_test_set
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import logging
import sys
import numpy as np
import pandas as pd

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

# Se lee la data
logger.info('Loading Data...')
data = pd.read_csv('dataset/full_data.csv')

# Se crea el modelo
logger.info('Loading model...')
model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean',missing_values=np.nan)),
    ('core_model', GradientBoostingRegressor())
])

# Se separa la data en X e y
logger.info('Seraparating dataset into train and test')
X = data.drop(['worldwide_gross'], axis= 1)
y = data['worldwide_gross']

# Se splitea la data en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Se setean los parámetros para los hiperparámetros
logger.info('Setting Hyperparameter to tune')
param_tuning = {'core_model__n_estimators':range(20,301,20)}

# Se instancia grid search
grid_search = GridSearchCV(model, param_grid= param_tuning, scoring='r2', cv=5)

# Se hace el fiteo
logger.info('Starting grid search...')
grid_search.fit(X_train, y_train)

# Se hace la validación cruzada del mejor modelo
logger.info('Cross validating with best model...')
final_result = cross_validate(grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=5)

# Se calcula el promodio de score en train y tesr
train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])
assert train_score > 0.7
assert test_score > 0.65

logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score}')

# Recibe un modelo y lo guarda
logger.info('Updating model...')
update_model(grid_search.best_estimator_)

# Validación del modelo mediante un reporte
logger.info('Generating model report...')
validation_score = grid_search.best_estimator_.score(X_test, y_test)
save_simple_metrics_report(train_score, test_score, validation_score, grid_search.best_estimator_)

# Calcula la predicción y grafica
y_test_pred = grid_search.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, y_test_pred)

logger.info('Training Finished')

