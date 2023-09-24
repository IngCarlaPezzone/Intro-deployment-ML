# Curso de Introducción al Despliegue de Modelos de Machine Learning

Curso dictado por Gerson Perdomo en Platzi. Todos los créditos para el autor. Esto es sólo los apuntes que fui tomando para aprender sobre el tema.

# Introducción

## ****¿Qué es el despliegue de modelos y por qué es necesario?****

Definiremos el **despliegue de modelos** como la transformación de un modelo en producto que puede ser utilizado como servicio por un tercero.

Esto es importante porque podemos utilizarlo como un producto final que van a poder consumir nuestros clientes.

# Machine Learning Operations

## ****Historia del MLOps y estado del arte****

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled.png)

Hay tres grandes pilares la **DATA**, que son los datos con los que hacemos un **MODELO** y que eventualmente hacemos el **CÓDIGO**, referido esto último a un producto o servicio.

Si la data cambia, el modelo tendrá que cambiar, en este caso el modelo va a tener que generar un nuevo producto (un nuevo código). Esto pasa porque si la data cambia, nuestro modelo se va a estar degradando, por lo que necesitamos reentrenar nuestro modelo para que se redespliegue un nuevo servicio o producto.

A este ciclo se lo conoce como las ******************************************Operaciones de Machine Learning******************************************

Andriy Burkov define a la ingeniería de aprendizaje automático como el uso de principios científicos, herramientas y técnicas de aprendizaje automático e ingeniería de software para diseñar y construir sistemas complejos para que estén disponibles como productos.

La evolución del MLOps:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%201.png)

*Fuentes de consulta*: 

[ml-ops.org](https://ml-ops.org/content/motivation)

[LF AI & Data Landscape](https://landscape.lfai.foundation/)

## ****Flujo de vida de un modelo en producción****

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%202.png)

La **data** constantemente se esta extrayendo de varias fuentes (bases de datos o APIs), para luego hacerle una exploración y validación de esta data. 

Eventualmente esta data se va a limpiar para que sea utilizada en el entrenamiento. En este caso, a esta data tenemos que versionarla.

Una vez versionada procedemos a separarla en train y test. El train se usa en el Model Engineering, lo que se conoce como Ingeniería de características o Feature engineering. Eventualmente esto va a generar un **modelo** que vamos a evaluar con los datos de test en función de nuestra métrica de negocio o de coste para el performance de nuestro modelo.

Una vez que esta el modelo listo, éste se va a empaquetar, por ejemplo, en formato .pkl. Otro formato es ONNX que se usa cuando necesitamos un framework en producción pero tenemos un framework de desarrollo.

Ahora, cuando el modelo esta lista, también versionamos el modelo. Luego esto se va a una plataforma en la que se va a integrar y se va a enviar a producción, es decir, se hace un deployment. Este deployment puede estar empaquetado con Docker.

Eventualmente, recibiremos feedback de ese modelo deployado, el cual tendrá un performance en producción que vamos a monitorear. Una vez que se observe un decaimiento o degradación de este modelo, se va a activar un trigger que eventualmente va a activar el proceso de flujo de data, para traer nueva data, la cual será versionada y así continuar con el flujo para mantener vivo el modelo.

*Fuente de consulta*: 

[ml-ops.org](https://ml-ops.org/content/end-to-end-ml-workflow)

## ****Requerimientos para poder hacer MLOps****

- **Ingeniería de datos**: esto se refiere a
    - Ingesta de datos
    - Manejo de datos
- ****************Machine Learning Pipelines:**************** donde se genera el
    - Entrenamiento dle modelo
    - Evaluación del modelo
    - Empaquetado y serialización de modelo
- **Deployment Pipelines**: que sería
    - Servicio de modelo
    - Monitoreo del rendimiento del modelo

Hay varios patrones de diseño para servir el modelo:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%203.png)

- **Model-as-Service** (será el que implementaremos en el curso): El modelo es un servicio que puede ser visitado por una request (por ejemplo vía protocolo http)
- **Model-as-Depenency**: El modelo está incrustado dentro de la dependencia que se está utilizando en la aplicación. Por lo que podría ser consumido llamando al método *predict()* y tener la inferencia
- **Precompute Serving Pattern**: El modelo ya existe y hace predicciones. Esas predicciones serán guardadas en bases de datos que serán visitadas con posterioridad
- **Model-on-Demand**: Se utiliza para arquitecturas en streaming. Tenemos el *message broker* que es el encargado del flujo entre envía los inputs para las predicciones desde los datos, y envía las predicciones hechas por el modelo al servicio que lo requiera.
- **Federated Learning (o Hybrid Learning)**: Es un grupo de modelos. Se asigna un modelo por cada usuario que utilicé la aplicación, y sus datos son los que entrenarán al modelo que luego hará predicciones.

Para elegir el patrón de diseño en función de mi proyecto, debemos comparar la taxonomía entre el modelo y el servicio:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%204.png)

# Presentación del proyecto

## ****Presentación y arquitectura del proyecto MLOps****

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%205.png)

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%206.png)

Desde nuestro local, vamos a subir el proyecto a GutHub, donde usaremos GitHub Actions que van a activar tres workflows diferentes:

- **Testing**
- **CI/CD** donde vamos a tener la Integración Continua para Docker y luego el despliegue con Cloud Run utilizando una API con FastAPI
- **Entrenamiento continuo** que va a utilizar Scikitlearn, DVC para el versionado y Continue Machine Learning (CML) para publicar las métricas de performance de nuestro modelo.

Es importante la presencia de DVC entre local y Github porque va a serializar nuestros datos conectandonos con Google Claud Storage guardando allí nuestro dataset y nuestro modelo, serializandolos a partir de hashes.

## ****Revisión de notebooks y modelado****

*Archivo*: intro-deployment-ml-model-revision.zip

Creamos un repositorio en GitHub donde contrá las Actions de nuestro worklow llamado **Intro-deploymen-ML**

Clonamos el repositorio.

> NOTA: El curso propone crear .gitignore mas adelante. Yo sugiero hacerlo aca.
> 

Creamos una nueva rama: 

```jsx
git checkout -b model-revision
```

Aquí se agregan los archivos del `dataset`, `modelo` y `notebooks`.

Crear un ambiente de trabajo y lo activamos:

```jsx
python -m venv venv
venv\Scripts\activate
```

Instalamos los requerimientos que estan en la carpeta de notebooks y luego instalamos Jupyter si no tenemos la extensión de Jupyter para trabajar en Visual Studio Code:

```jsx
pip install -r notebooks/requirements.txt
pip install jupyter
```

Activamos Jupyter si queremos trabajar con el navegador. Sino, desde VSC activamos el kernel del venv creado, para lo cual será necesario instalar ipykernel y lo ofrecerá directamente VSC.

```jsx
jupyter notebook
```

Vemos la notebook modeling.ipynb donde esta todo el flujo del modelo del proyecto y ejecutamos todas las celdas.

> NOTA: en mi caso tuve errores al querer usar los requerimientos del proyecto. Tuve que instalar la última versión de pandas y scikitlearn.
> 

## ****Distribución archivos y contenido****

Esta es una distribución de archivos utilizada en este tipo de proyectos:

- **dvc/** <- guarda las configuraciones de DVC
- **.github/workflows/** <- guarda las actions que se ejecutaran, como testing, continuo integration y continue deployment, etc.
- **api/** <- contiene los archivos de la API utilizando fast API
- **datset/** <- archivos del dataset traqueados, es decir .csv.dvc
- **model/** <- archivos de modelos traqueados, es decir .pkl.dvc
- **notebooks/** <- notebooks con el modelo
- **src/** <- archivos usados para reentrenamiento
- **utilities/** <- archivos con utilidades especificas
- **…** <- archivos misceláneos del proyecto y extras que se necesitaran y se guardan en el root

# Data Version Control

## ****¿Qué es DVC y por que lo utilizaremos?****

Documentación: 

[Data Version Control · DVC](https://dvc.org/)

DVC es un sistema de control de versiones para proyectos de ML opern source. Trackea tanto datos como modelos aprovechando las facilidades que nos da Git.

Nosotros tenemos los datos y los modelos y los trackeamos con DVC y estos se envían a un Storage remoto que puede ser cualquier nube.

Los archivos son .dvc y Git puede trackearlos y GitHub puede posteriormente recibirlos.

De esta manera tendremos nuestros archivos en un repositorio remoto, actualizados y serealizados, para asegurar reproducibilidad de los modelos e incluso de los datos.

## ****Comandos básicos para implementar DVC****

Documentación: 

[Command Reference](https://dvc.org/doc/command-reference)

- **dvc init**: inicia la configuración de dvc creando la carpeta .dvc similar a cuando iniciamos git
- **dvc remote**: donde guarda los storage remotos de la nube que usamos
- **dvc add**: agrega un archivo
- **dvc pull**: enviar un archivo
- **dvc push**: empujar un archivo a la nube
- **dvc run**: crea una subtarea
- **dvc repro**: corre los pasos de la subtarea creada en run
- **dvc dag**: lo anterior genera un grafo acíclico dirigido, los cuales se generan con cada ciclo ejecutado por run

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%207.png)

## ****Implementando DVC en nuestro proyecto****

Vamos a implementar DVC combinando el Storage de Google Cloud Platform con nuestros archivos.

Actualizamos los cambios realizados al proyecto:

```jsx
git add .
git commit -m "Adding model revision"
```

Luego creamos una nueva rama:

```jsx
git checkout -d implementing_dvc
```

Ahora, vamos a ir a GCP a donde vamos a crear las credenciales del Storage Manager.

Una vez en Google Cloud crearemos un proyecto nuevo llamado MLOps-fundamentals.

Vamos a API & Services en el menú de navegación y buscamos Credentials.

Ahí creamos una nueva credencial que sea Service Account.

Le colocamos el nombre Storage-Manager. Esto es para manejar los archivos de DVC. Crear y continuar. Luego, damos los permisos pertinentes, que en este caso es Google Cloud Storage y Storage Admins. Continuar y Listo.

Ahora, vamos al link de storage-manager que aparece en Cuentas de servicio para descargar una key. Vamos a Key, creamos una nueva Key y pedimos una de formato JSON. Crear. Esta clave se descarga y la tenemos que guardar en la raíz de nuestro proyecto. 

Ahora vamos a instalar DVC:

```jsx
pip install dvc
```

Luego instalamos DVC para GCP:

```jsx
pip install dvc[gs]
```

Inicializamos con

```jsx
dvc init
```

Ahora, esto tiene que conectarse con Google Cloud Storage a través de las credenciales. Para hacer eso se tiene que setear una variable de ambiente:

```jsx
set GOOGLE_APPLICATION_CREDENTIALS=<ruta del archivo>
```

A continuación, tenemos que agregar un nuevo Storage y para ello tenemos que crear un Bucket en  Googe Cloud Storage que van a contener nuestros datos y modelos serializados. Una vez en la consola de GCP, vamos a Google Storage y Buckets.

Ponemos Crear. Ponemos un nombre: model-datasets-tracker. Continuar. Dejar la opción de Multi-region. Continuamos. Dejamos el formato Standard. Continuamos. Dejamos Uniforme. Continuamos. Dejamos Ninguno. Creamos.

Ya tenemos el bucket creado, ahora debemos crear dos carpetas `dataset` y `model`.

Ahora tenemos que conectar esas carpetas con nuestros archivos. Para ello, desde la terminal:

```jsx
dvc remote add dataset-track gs://model-datasets-tracker/dataset
dvc add dataset/<nombre del archivo> --to-remote -r dataset-track
```

Con remote estamos conectando con la carpeta remota y con add estamos subiendo el archivo a GCP. La segunda línea se repite por cada archivo.

> NOTA: si sale este error

ERROR: unexpected error - ('invalid_grant: Invalid JWT: Token must be a short-lived token (60 minutes) and in a reasonable timeframe. Check your iat and exp values in the JWT claim.', {'error': 'invalid_grant', 'error_description': 'Invalid JWT: Token must be a short-lived token (60 minutes) and in a reasonable timeframe. Check your iat and exp values in the JWT claim.'})

En Windows ir a cambiar la Fecha y Hora y en fecha y hora hacer click en Sincronizar ahora.
> 

Vemos que se van subiendo los archivos y que se generan los archivo .csv.dvc en la carpeta de dataset en local y se suben archivos en GCP.

Hacemos lo mismo para el modelo, pero cambia levemente el comando:

```jsx
dvc remote add model-track gs://model-datasets-tracker/model
dvc add model/<nombre del archivo> --to-remote -r model-track
```

También veremos que aparece un archivo .pkl.dvc en la carpeta model.

Generamos el archivo .gitignore para omitir los archivos que no queremos que se suban a Github

```jsx
venv/
.ipynb_checkpoints/
*.json
*.csv
*.pkl
```

Finalmente, gurdamos los cambios en git con:

```jsx
git add .
git commit -m "DVC has been implementing"
```

## ****Desarrollo de pipeline de reentrenamiento: preparación de la data****

Vamos a hacer el pipeline de reentrenamiento.

En primer lugar creamos una nueva branch:

```jsx
git checkout -d continuos_training_pipeline
```

En esta rama creamos el sistema de archivos que se comentó anteriormente. Primero creamos la carpeta src que contiene:

- **prepare.py** que contiene la preparación de los datos
- **train.py** que va a contener el codigo que va a reentrenar el modelo
- **utils.py** que va a contener alguna función de utilidad
- **requirements.txt** con los requerimientos que se van a necesitar para ejecutar este pipeline. En este caso se usó:

```jsx
pandas==1.3.3
scikit-learn==1.3.1
matplotlib==3.8.0
seaborn==0.12.2
dvc
dvc[gs]
```

> NOTA: en el archivo prepare.py se agregaron comentarios sobre que hace el código
> 

Para ver que todo funciona:

```jsx
python src\prepare.py
```

## ****Desarrollo de pipeline de reentrenamiento: script de entrenamiento****

El archivo train.py tiene el código para el entrenamiento.

> NOTA: en el archivo train.py se agregaron comentarios sobre que hace el código.
> 

## ****Desarrollo de pipeline de reentrenamiento: validación del modelo****

En el archivo train.py se hace la validación y se grafica.

Para probar si todo funciona, ver el reporte que genera y el gráfico:

```jsx
python src\train.py
```

Ahora aplicamos DVC para crear este flujo continuo. para ellos se usa stage add que nos permite crear un paso donde vamos a guardar un script. Le damos un nombre (prepare) con la bandera -n, luego se coloca el output con bandera -o y finalmente el script:

```jsx
dvc stage add -n prepare -o dataset/full_data.csv python src/prepare.py
```

Para el entrenamiento se hace algo similar:

```jsx
dvc stage add -n training -d dataset/full_data.csv python src/train.py
```

Ahora ya tenemos todo orquestado por DVC.

Ahora, se puede usar el siguiente comando para reproducir todo este comportamiento:

```jsx
dvc repro
```

Si se ejecuta en este momento no pasa nada, porque prepare y train no han cambiado. Pero esto será muy útil cuando nos enfrentemos a pull request o push en nuestro repositorio, porque alguien ofrece nuevas features, o nuevas formas de hacer feature engenieering.

Para forzar la reproducción de todo obligatoriamente se puede usar:

```jsx
dvc repro -f
```

Así se ve aplicado ese comando:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%208.png)

Para ver los dag que se van formando se puede usar:

```jsx
dvc dag
```

Así se ve implementado ese comando:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%209.png)

Una vez llegado hasta acá ya esta impementado nuestro pipeline de entrenamiento.

# Desarrollo de API con FastAPI

## ****Desarrollo de API con FastAPI****

Agregamos a .gitignore report.txt, *.png y __pycache__/, para ignorar esos archivos.

Luego actualizar en git commitiando como “Pipeline of training”.

Ahora creamos una nueva rama:

```jsx
git checkout -b api_creation
```

Luego, en esa rama, creamos una nueva carpeta llamada `api`.

En esta carpera se tiene main.py que contiene la aplicación en FastAPI y requirements.txt de la aplicación. Luego necesitamos una carpeta con las utilidades de la aplicación llamada app que contendrá tres archivos: models.py, views.py y utils.py.

En los requirements vamos a usar:

```jsx
fastapi==0.103.1
uvicorn==0.23.2
scikit-learn==1.3.1
pandas==2.3.3
gunicorn==21.2.0
joblib==1.3.2
```

## ****Desarrollo de API con FastAPI: creación de utilidades****

En este punto se termina de crear utils.py, donde se lee el modelo y se transforman los datos a dataframe.

Para ver que todo funciona:

```jsx
uvicorn api.main:app
```

Vemos así en el navegador:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2010.png)

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2011.png)

Cerramos la api y guardamos los cambios commitiando como “api created”.

## ****Desarrollo de testeo de API****

Creamos una nueva rama

```jsx
git checkout testing_api
```

Hacemos un requirements en la raiz para es testing llamado requirements_test.txt. En este archivo tenemos `r api/requirements.txt` que aprovecha los requerimientos que ya hicimos de la api y luego usaremos un framework para el testeo `pytest==7.4.2`

Luego se genera el archivo test.py donde se crea un cliente y dos consultas para probar la api.

Para probarlo, ejecutamos en consola:

```jsx
pytest tests.py
```

El resultado del test es:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2012.png)

## ****Empaquetando API con Docker****

Ahora vamos a empaquetar la api en una imagen de Docker.

Antes guardamos los cambios commitiando como “Adding testing”.

Hacemos una nueva rama:

```jsx
git checkout -b creating_dockerfile
```

Este archivo contiene:

```jsx
FROM python:3.8-slim-buster

WORKDIR /app

COPY api/requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY api/ ./api

COPY model/model.pkl ./model/model.pkl

COPY initializer.sh .

RUN chmod +x initializer.sh

EXPOSE 8000

ENTRYPOINT ["./initializer.sh"]
```

También creamos el archivo initializer.sh que contiene lo siguiente:

```jsx
#!/bin/bash

gunicorn --bind 0.0.0.0 api.main:app -w 2 -k uvicorn.workers.UvicornWorker
```

Gunicor se usa para que los medios que hace que FastAPI sea sincrónico puedan ejecutarse de manera paralela en varios workers.

Finalmente, guardamos los cambios y commitiamos como “Adding Dockerfile”

# Continuous Integration/Deployment/Training

## ****Presentacion de Github Actions y Continuous Machine Learning****

Ahora falta automatizar las acciones de testing, la creación de imágenes. La herramienta para esto es GitHub Actions y para las partes de ML vamos a usar Continuos Machine Learning.

Links de referencia: 

[GitHub Actions documentation - GitHub Docs](https://docs.github.com/en/actions)

[https://github.com/iterative/cml](https://github.com/iterative/cml)

GitHub Actions son workflows que estan corriendo en la pestaña de Actions en nuestro repositorio.

La estructura de una Actions es:

- name: es para todo el workflow
- on: es el evento que activa esta action
- jobs: son los trabajos que se van a ejecutar dentro de este workflow
    - workflow-part: es el nombre dle job dentro del workflow
        - run on: es el SO donde se va a ejecutar
        - env: donde podemos setear variables de ambiente
            - GITHUB_SECRET: es para guardar variables secretas
        - steps: son los pasos que van ejecutandose dentro de este job
            - name: nombre del paso
            - uses: accion que utiliza
            - name: otra acción
            - run: ejecutamos los pasos dentro del script

Ahora, como podemos ejecutar muchas automatizaciones que ya fueron hechas y que son open source, entonces podemos usar Continuos Machine Learning que nos sirve para publicar métricas y reportes para ver cómo se está comportando nuestro modelo.

## ****Desarrollo de workflow para testing****

Vamos a desarrollar el workflow que va a testear nuestra API usando Github Actions.

Creamos una nueva rama:

```jsx
git checkout -b workflow_testing_api
```

Creamos una carpeta llamada `.github` y dentro de esta creamos una que se llama `workflow`. Allí creamos un archivo testing.yaml

```jsx
name: Testing API
on: [push, pull_request]
jobs:
  testing-api:
    runs-on: ubuntu-latest
    env:
      SERVICE_ACCOUNT_KEY: ${{ secrets.SERVICE_ACCOUNT_KEY }}
    steps:
      - name: Checkout the repo
        uses: actions/checkout@v3
      - name: Creating and activating virtualenv
        run: |
          pip3 install virtualenv
          virtualenv venv
          source venv/bin/activate
      - name: Installing dependencies
        run: |
          pip install dvc[gs]
          pip install -r requirements_test.txt
      - name: Test API
        run: |
          export GOOGLE_APPLICATION_CREDENTIALS=$(python utilities/setter.py)
          dvc pull model/model.pkl -r model-tracker
          pytest tests.py
```

En este caso, en el ambiente vamos a guardar como variable de entorno a las credenciales para conectar con GCP.

En los pasos, primero nos tremos el repositorio. Luego ejecuta una action que nos trae el repositorio.

El segundo paso es crear y activar el ambiente virtual con los comandos a ejecutar.

El tercer paso instala dependiencias.

Por último paso, exportamos las credenciales de GCP. Para setear el path vamos a hacer un script para setearlo (setter.py que ubicaremos en utilities)

Para crear este archivo `setter`, primero creamos una carpeta `utilities` y luego ese archivo. El contenido de este archivo permite transformar el contenido de la variable Service_Account_Key en algo que pueda ser interpretado google_aplication_credentials, que sería un path a un JSON.

```jsx
import os
from base64 import b64decode

def main():
    key = os.environ.get('SERVICE_ACCOUNT_KEY')
    with open('path.json','w') as json_file:
        json_file.write(b64decode(key).decode())
    print(os.path.realpath('path.json'))

if __name__ == '__main__':
    main()
```

Ahora, desde GCP vamos a crear una credencial que nos permite conectarnos a Cloud Run, nuestro servicio para despliegue, a Cloud Storage, nuestro servicio de almacenamiento y Artifact Registry, nuestro almacenamiento de contenedores.

Para ello, vamos a API & Service y luego Credentials. Ahí, creamos una nueva Service Account y la vamos a llamar `deployer-and-storage-acces`, continuar. 

Buscamos Cloud Storage y buscamos el permiso de Administrados. Luego buscamos Cloud Run y buscamos Admin. Luego buscamos Artifact Registry también administrador. Finalmente buscamos Service Account y en este caso debe ser Service Account User. Continuar. Terminar.

Ahora entramos a `deployer-and-storage-acces`, ahí dentro creamos una nueva key del tipo JSON y lo descargamos y lo guardamos en la raíz. En consola de powershell en la misma ubicación donde esta el archivo hacemos:

```jsx
[Convert]::ToBase64String([System.IO.File]::ReadAllBytes("nombre del archivo"))
```

Esto genera le base 64 del archivo. Lo copiamos y tenemos que ir a Github.

Vamos a Settings, luego Secrets and variables, luego a Actions, luego a New repository secret. Colocamos como nombre SERVICE_ACCOUNT_KEY y como secreto el base64 que copiamos. Creamos.

Ahora podemos hacer un push de nuestra rama workflow_testing_api a Github.

Commitiamos los últimos cambios como “Adding workflow for testing and add utilities” y hacemos el primer push con todas las ramas creadas.

```jsx
git push -all origin
```

Ahora, desde la rama workflow_testing_api en Github, vamos a actions y se puede ver el proceso de la misma.

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2013.png)

A continuación, se implementa el workflow de entrenamiento continuo.

## ****Desarrollo de workflow para Continuous Training utilizando CML****

Hacemos una nueva rama:

```jsx
git checkout -b workflow_continuos_training
```

Luego en la carpeta de `.github/workflows`creamos el archivo `continuos_training.yaml`. Este archivo es similar al de testing, pero con algunas diferencias en función de los tiggers para este caso.

```jsx
name: Continuous Training
on:
  push:
    branches:
      - workflow_continuos_training # va el nombre de la rama para que se dipare.
  schedule:
    - cron: '0 */6 * * *'
  workflow_dispatch:
    inputs:
      reason:
        description: Why to run this?
        required: false
        default: running CT
jobs:
  continuous-training:
    runs-on: ubuntu-latest
    permissions: # Set permissions to do git push
      contents: write
      pull-requests: write 
      issues: read
      packages: none
    steps:
      - name: Checkout repo
        uses: actions/checkout@v3
      - name: Set up Node 16
        uses: actions/setup-node@v1
        with:
          node-version: '16'
      - name: Train model
        env: 
          SERVICE_ACCOUNT_KEY: ${{ secrets.SERVICE_ACCOUNT_KEY }}
        run: |
           pip3 install virtualenv
           virtualenv venv
           source venv/bin/activate
           pip install -r src/requirements.txt
           export GOOGLE_APPLICATION_CREDENTIALS=$(python utilities/setter.py)
           dvc pull model/model.pkl.dvc -r model-track
           dvc unprotect model/model.pkl
           dvc repro -f 
           echo "Training Completed"
           dvc add model/model.pkl -r model-track --to-remote
           dvc push model/model.pkl.dvc -r model-track
      - name: Commit .dvc file changes
        run: |
          git config --local user.email "carlapezzone@gmail.com"
          git config --local user.name "github-actions[bot]"
          git add model/model.pkl.dvc
          git commit -m "Updating model serialization"
      - uses: ad-m/github-push-action@master
        with:
            github_token: ${{ secrets.GITHUB_TOKEN }}
            branch: ${{ github.ref }}
      - uses: iterative/setup-cml@v1
      - name: Push metrics
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          cat report.txt >> report.md 
          echo "![](./prediction_behavior.png)" >> report.md
          cml comment create report.md
```

**schedule** es para que Github Action corra el workflow con el crum '0 */6 * * *’, esto significa que se ejecuta cada 6 horas.

**workflow_dispatch** es por si queremos correrlo manualmente.

El workflow lo que hace es correr cada 6 horas, las tareas de ejecutar la action del repo, entrena el modelo, trakea el modelo y publica las métricas del reporte e imágen.

Guardamos los cambio y commitiamos “Adding continuos trainig workflow” y pusheamos.

Vemos que se ejecuta solo la accion de Testing:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2014.png)

Esto es porque se ejecuta cada 6 horas o hasta que el dispatch lo dispare. Para eso, se agrega push branches al comienzo, para que con un push se dispare.

> NOTA: si da un error de permiso desde le repo en [bot], hay que ir a Configuración en Github, a Actions y luego buscar y tildar lo siguiente:
> 
> 
> ![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2015.png)
> 

Finalmente, se ejecuta todo el workflow y se ve así:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2016.png)

Ahora, si tocamos en el nombre del commit, y vamos hacia abajo vemos el reporte y el gráfico:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2017.png)

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2018.png)

Con esto se termina el workflow del entrenamiento continuo de nuestro modelo, en el que se va a actualizar el serial de las versiones de nuestro modelo con DVC y también nos da un reporte con algunas métricas sobre el perform de los datos que vayan entrando.

## ****Creacion de Servicio en la nube para el despliegue y desarrollo de workflow para CI/CD****

Vamos a realizar el Continuous integration y Continuos deployment (CI/CD) como último workflow para poner en producción nuestro modelo. Vamos a utilizar GistHub Actions y un servicio de Cloud Run en GCP.

Desde GCP vamos a crear un nuevo servicio de Cloud Run, Create Service. 

Elegimos una imagen base porque nuestra imágen la vamos a generar con nuestro CI/CD. Para ello En Deploy one revision, ir a SELECT, luego a la derecha, en Container registry, en Demo conteiners seleccionar hello.

Ponemos un nombre al servicio: deployment-intro-ml-service.

En autoscaling, colocar 1 en mínimo y 10 en máximo. Esto dependerá del proyecto.

En Authentication, seleccionar Allow unauthenticated incocations.

En advance setting, el contenedor va a tener el puerto 8000 porque es el default de FastAPI.

La memoria será de 1 GB.

En maximun requests per container poner 10.

Al crear, comienza a desplegar el servicio de Cloud Run en base al hello. Pero lo que necesitamos de aca es el nombre del servicio y la región que la vamos a colocar en Github, además de colocar el registry name, que es el lugar donde va a estar alocado nuestras imágenes:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2019.png)

Una vez en el Github, vamos a settings y vamos a crear cuatro Secrets desde New repository secret:

- **REGION** que es us-central1
- **REGISTRY_NAME** que es gcr.io/mlops-fundamentals-399913/intro-deployment-ml que es el registry de google/el id del proyecto/proyecto de Github todo en minúsculas
- **SERVICE_NAME** que es el nombre del servicio de Cloud Run
- **PROJECT_ID** que es el ID del proyecto en Google Cloud (mlops-fundamentals-399913 en mi caso)

Ahora, vamos al código. Creamos una nueva rama:

```jsx
git checkout -b workflow_CI_CD
```

Lo primero que vamos a hacer es eliminar esta parte de nuestro archivo continuous_training.yaml, ya que eso sólo nos servía para ver que todo esté funcionando al hacer push:

```jsx
push:
    branches:
      - workflow_continuos_training # va el nombre de la rama para que se dipare.
```

Ahora creamos en `.github/workflows`un nuevo archivo llamado `ci_cd.yaml` que tiene la siguiente estructura:

```jsx
name: Continuous Integration/Continuous Deployment
on: [push]
jobs:
  ci_cd:
    runs-on: ubuntu-latest
    env:
      REGISTRY_NAME: ${{ secrets.REGISTRY_NAME }}
      REGION: ${{ secrets.REGION }}
      PROJECT_ID: ${{ secrets.PROJECT_ID }}
      SERVICE_NAME: ${{ secrets.SERVICE_NAME }}
      SERVICE_ACCOUNT_KEY: ${{ secrets.SERVICE_ACCOUNT_KEY }}
    steps:
      - name: Checkout Repo
        uses: actions/checkout@v3
      - name: Set environment and bring the model
        run: |
          pip3 install virtualenv
          virtualenv venv
          source venv/bin/activate
          pip install dvc[gs]
          export GOOGLE_APPLICATION_CREDENTIALS=$(python utilities/setter.py)
          dvc pull model/model.pkl.dvc -r model-track
      - name: Set up GCLOUD SDK
        uses: google-github-actions/setup-gcloud@master
        with:
          service_account_key: ${{ secrets.SERVICE_ACCOUNT_KEY }}
          project_id: ${{ secrets.PROJECT_ID }}
      - name: Build and Push
        run: |
          docker build . -t $REGISTRY_NAME:$GITHUB_SHA
          gcloud auth configure-docker -q
          sudo -u $USER docker push $REGISTRY_NAME:$GITHUB_SHA
      - name: Deploy to Cloud Run
        run: |
          gcloud run services update $SERVICE_NAME --region=$REGION --image=$REGISTRY_NAME:$GITHUB_SHA
```

on: [push] es para ver como funciona. Luego se cambia para que se actualice con los push a main.

Agregamos los cambios y commitiamos como “Adding CI/CD workflow”. Vemos que las Actions funcionen bien.

> NOTA: si en este punto vemos que los Test no pasan, pero antes sí pasaban, esto puede ser porque con distintas pruebas se cambió el md5 y size del modelo cuando fuimos haciendo los push en el branch de workflow_continuos_training. Para corregir esto, ir a Actions, luego al workflow de Testing API y buscar el primer Testing erroneo. 
Entrar al commit de ese primer error y ver cuál era el md5 y size que sí funcionaban y cambiarlo en nuestro archivo `model.pkl.dvc`y actualizar el ropo. Con esto debería funcionar
> 

Cuando todo funcione, podemos volver a GCP en Cloud Run y abrir el enlace a nuestra aplicación:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2020.png)

Ahora podemos ver la API deployada en GCP y podemos probarla.

Para finalizar, necesitamos quitar [push] de `ci_cd.yaml`y agregar lo siguiente para que haga el despliegue con cada push a main. También aprovechamos para que se haga el entrenamiento continuo.

```jsx
on:
 push:
    branches:
     - main 
 workflow_run:
   workflows: ["Continuous Training"]
   branches: [main]
   types:
     - completed
 workflow_dispatch:
   inputs:
     reason:
       description: Why to run this?
       required: false
       default: running CI/CD
```

Comittiamos los cambios como “Last configuration for production”.

En Github hacemos el merge a la rama main, desde la rama de workflow_CI_CD. Revisamos que todo este bien:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2021.png)

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2022.png)

Desde Código, verificando que estemos en la rama workflow_CI_CD ir a Comapare & pull request:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2023.png)

Revisar que el pull se este haciendo a la rama main y agregar un comentario:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2024.png)

Crear el pull request y esperar que se ejecuten las Actions. Una vez que se ejecutan hacer el Merge pull request y confirmarlo:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2025.png)

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2026.png)

Si vamos a código, vemos que todo esta en main ahora.

Recordar que:

- El testing se va a hacer cada push y pull request
- El entrenamiento continuo se va a hacer por dispath o cada 6 horas
- El CI/CD se va a hacer por cada push a la rama main, es decir, con cada pull request a main o cuando un entrenamiento continuo termine

Si vamos a las Actions vemos que todo se ejecutó correctamente:

![Untitled](Curso%20de%20Introduccio%CC%81n%20al%20Despliegue%20de%20Modelos%20de%204f1e73d12bdb40e2b48367047e21a566/Untitled%2027.png)

Con esto ya esta nuestro modelo en producción con un entrenamiento continuo.

# Sigue aprendiendo

## ****Otras maneras de hacer despliegues de modelos****

Para Demos de modelos pueden usarse Gradio y Streamlit, que tienen servicios en la nube para inferencias de usuario pequeños, no recurrentes.

[Gradio](https://gradio.app/)

[Streamlit • A faster way to build and share data apps](https://streamlit.io/)

Con servidores pre construidos como Tensorflow Serving y Torch Serve en donde ya el servicio esta contruido y el modelo es lo que debe ser cargado y esto genera un endpoint:

[TorchServe — PyTorch/Serve master documentation](https://pytorch.org/serve/)

Plataformas de desarrollo como Tensorflow Extended y mlflow que nos permiten osquertar de una manera mas sencilla con nuestro código en Python, todo el pipeline de entrenamiento

[MLflow - A platform for the machine learning lifecycle](https://mlflow.org/)

Plataformas en Kubernetes como Kubeflow y Seldon que gestinan los pipeline desde traer los datos, entrenarlos, mantener un modelo y generar endpoints.

[Kubeflow](https://www.kubeflow.org/)